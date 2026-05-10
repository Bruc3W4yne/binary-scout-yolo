"""
Scout feature extraction helpers.

The first feature mode is deliberately simple: per-tile RGB and bitplane
statistics. It is a C-free baseline for validating tile labels and scout
training before the binary-XNOR feature path is added.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from binary_layer import BinaryConvLayer
from preprocess import Tile, rgb_to_bitplanes

BITPLANE_STATS_DIM = 30
BINARY_XNOR_DIM = 64


def load_resized_rgb(path: Path, img_size: int = 640) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def bitplane_stats_features(rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")

    planes = rgb_to_bitplanes(rgb).astype(np.float32)
    rgb_f = rgb.astype(np.float32) / 255.0
    features = np.empty((len(tiles), BITPLANE_STATS_DIM), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        bit_patch = planes[:, tile.y1:tile.y2, tile.x1:tile.x2]
        rgb_patch = rgb_f[tile.y1:tile.y2, tile.x1:tile.x2, :]
        if bit_patch.size == 0 or rgb_patch.size == 0:
            raise ValueError(f"empty tile patch: {tile.xyxy()}")

        features[idx, :24] = bit_patch.mean(axis=(1, 2))
        features[idx, 24:27] = rgb_patch.mean(axis=(0, 1))
        features[idx, 27:30] = rgb_patch.std(axis=(0, 1))

    return features


def binary_xnor_features(
    rgb: np.ndarray,
    tiles: list[Tile],
    n_filters: int = BINARY_XNOR_DIM,
    kernel_size: int = 3,
    threshold: int = 0,
    seed: int = 42,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")

    planes = rgb_to_bitplanes(rgb)
    layer = BinaryConvLayer(
        n_filters=n_filters,
        n_ch=planes.shape[0],
        kH=kernel_size,
        kW=kernel_size,
        seed=seed,
    )
    binary_maps = layer.apply_threshold(layer.forward(planes), threshold=threshold).astype(np.float32)
    features = np.empty((len(tiles), n_filters), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        patch = binary_maps[:, tile.y1:tile.y2, tile.x1:tile.x2]
        if patch.size == 0:
            raise ValueError(f"empty tile patch: {tile.xyxy()}")
        features[idx] = patch.mean(axis=(1, 2))

    return features
