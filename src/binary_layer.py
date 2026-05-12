"""
binary_layer.py — First binary conv layer: 24-channel input → N integer score maps.

Takes the 24 binary planes produced by the RGB bit-plane decomposition
(R bits 0-7, G bits 0-7, B bits 0-7) and applies N XNOR-Popcount filters,
each with shape [n_ch, kH, kW] of ±1 weights.

The output before thresholding is an integer score map per filter:
  score range = [-n_ch * kH * kW, +n_ch * kH * kW]
  positive → filter pattern matches; negative → anti-matches

After thresholding (score > threshold → 1, else 0) the output is binary,
ready to be packed into uint64 and fed into the next layer.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_wrapper as kw


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


class BinaryConvLayer:
    """
    XNOR-Popcount convolution layer with configurable ±1 weights.

    Parameters
    ----------
    n_filters : number of output feature maps (e.g. 64)
    n_ch      : number of input channels packed per pixel (max 64, e.g. 24 for RGB×8)
    kH, kW    : kernel height and width (must be odd)
    seed      : RNG seed for weight initialisation (default 42)
    """

    def __init__(self, n_filters: int, n_ch: int, kH: int, kW: int, seed: int = 42):
        n_filters = _positive_int("n_filters", n_filters)
        n_ch = _positive_int("n_ch", n_ch)
        kH = _positive_int("kH", kH)
        kW = _positive_int("kW", kW)
        if n_ch > 64:
            raise ValueError(f"n_ch={n_ch} exceeds uint64 capacity of 64")
        if kH % 2 == 0 or kW % 2 == 0:
            raise ValueError("kH and kW must be odd")

        self.n_filters = n_filters
        self.n_ch = n_ch
        self.kH = kH
        self.kW = kW
        self.max_score = n_ch * kH * kW  # theoretical max per filter per pixel

        # Initialise with random ±1 weights
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 2, size=(n_filters, n_ch, kH, kW), dtype=np.int8)
        self.weights = np.where(raw == 0, np.int8(-1), np.int8(1))
        self._weights_packed = None

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Replace all filter weights.

        Parameters
        ----------
        weights : [n_filters, n_ch, kH, kW] int8 or int, values must be +1 or -1
        """
        weights = np.asarray(weights, dtype=np.int8)
        expected = (self.n_filters, self.n_ch, self.kH, self.kW)
        if weights.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {weights.shape}")
        if not np.all((weights == 1) | (weights == -1)):
            raise ValueError("All weight values must be +1 or -1")
        self.weights = weights.copy()
        self._weights_packed = None

    def set_filter(self, filter_idx: int, weights: np.ndarray) -> None:
        """
        Replace the weights of a single filter.

        Parameters
        ----------
        filter_idx : which filter to update (0-indexed)
        weights    : [n_ch, kH, kW] int8, values ±1
        """
        if isinstance(filter_idx, bool) or not isinstance(filter_idx, (int, np.integer)):
            raise TypeError(f"filter_idx must be an integer, got {type(filter_idx).__name__}")
        filter_idx = int(filter_idx)
        if not 0 <= filter_idx < self.n_filters:
            raise IndexError(f"filter_idx must be in [0, {self.n_filters}), got {filter_idx}")
        weights = np.asarray(weights, dtype=np.int8)
        expected = (self.n_ch, self.kH, self.kW)
        if weights.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {weights.shape}")
        if not np.all((weights == 1) | (weights == -1)):
            raise ValueError("All weight values must be +1 or -1")
        self.weights[filter_idx] = weights
        self._weights_packed = None

    # ------------------------------------------------------------------
    # Packing helpers (binary → uint64)
    # ------------------------------------------------------------------

    def pack_input(self, planes: np.ndarray) -> np.ndarray:
        """
        Pack [n_ch, H, W] uint8 planes into [H, W] uint64.

        Bit i of each uint64 = channel i's activation at that pixel position.
        Any nonzero input value is treated as bit 1.

        Parameters
        ----------
        planes : [n_ch, H, W] uint8, usually values 0 or 1

        Returns
        -------
        packed : [H, W] uint64
        """
        planes = np.asarray(planes)
        if planes.ndim != 3:
            raise ValueError(f"Expected [n_ch, H, W] input, got shape {planes.shape}")
        if planes.shape[0] != self.n_ch:
            raise ValueError(f"Expected {self.n_ch} channels, got {planes.shape[0]}")
        # Delegate to C: auto-vectorised loop under -O3 -march=native.
        return kw.pack_channels_to_u64(planes, self.n_ch)

    def pack_all_weights(self) -> np.ndarray:
        """
        Pack all filter weights into [n_filters, kH, kW] uint64 in one shot.

        Bit i at position (f, kr, kc) = 1 if weights[f, i, kr, kc] > 0.

        Returns
        -------
        packed : [n_filters, kH, kW] uint64
        """
        if self._weights_packed is None:
            bits = (self.weights > 0).astype(np.uint64)
            shifts = np.arange(self.n_ch, dtype=np.uint64).reshape(1, -1, 1, 1)
            self._weights_packed = np.bitwise_or.reduce(bits << shifts, axis=1)
        return self._weights_packed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_packed(self, input_packed: np.ndarray, zero_padding: bool = False) -> np.ndarray:
        """
        Apply all filters to an already packed [H, W] uint64 input.

        This is the same native kernel used by forward(), exposed so live
        benchmarks can time packing and XNOR-popcount separately.
        Set zero_padding=True to match PyTorch Conv2d padding semantics.
        """
        input_packed = np.asarray(input_packed)
        if input_packed.ndim != 2:
            raise ValueError(f"Expected [H, W] packed input, got shape {input_packed.shape}")

        conv = kw.xnor_multi_filter_conv_zero_pad if zero_padding else kw.xnor_multi_filter_conv
        return conv(
            input_packed,
            self.pack_all_weights(),
            self.kH,
            self.kW,
            self.n_ch,
            self.n_filters,
        )

    def forward(self, planes: np.ndarray, zero_padding: bool = False) -> np.ndarray:
        """
        Apply all n_filters XNOR-Popcount convolutions to the input planes.

        Uses a single C call (xnor_multi_filter_conv) — no Python loop over
        filters, no repeated ctypes overhead.

        Parameters
        ----------
        planes : [n_ch, H, W] uint8, values 0 or 1

        Returns
        -------
        scores : [n_filters, H, W] int32
                 Range: [-n_ch*kH*kW, +n_ch*kH*kW]
                 Larger positive score = stronger match to that filter pattern.
        """
        planes = np.asarray(planes)
        if planes.ndim != 3:
            raise ValueError(f"Expected [n_ch, H, W] input, got shape {planes.shape}")
        if planes.shape[0] != self.n_ch:
            raise ValueError(f"Expected {self.n_ch} input channels, got {planes.shape[0]}")

        input_packed = self.pack_input(planes)
        return self.forward_packed(input_packed, zero_padding=zero_padding)

    def apply_threshold(self, scores: np.ndarray, threshold: int = 0) -> np.ndarray:
        """
        Binarize integer score maps with a per-layer threshold.

        Parameters
        ----------
        scores    : [n_filters, H, W] int32 — output of forward()
        threshold : fire if score > threshold (default 0)
                    Increase to require stronger pattern matches before firing.

        Returns
        -------
        binary : [n_filters, H, W] uint8, values 0 or 1
                 Ready to be packed into uint64 for the next layer.
        """
        scores = np.asarray(scores)
        if scores.ndim != 3:
            raise ValueError(f"Expected [n_filters, H, W] scores, got shape {scores.shape}")
        if scores.shape[0] != self.n_filters:
            raise ValueError(f"Expected {self.n_filters} score maps, got {scores.shape[0]}")
        # Delegate to C: avoids creating the large boolean intermediate array
        # that numpy's comparison operator would allocate.
        return kw.threshold_i32_to_u8(scores, threshold)
