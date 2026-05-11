"""
Scout feature extraction helpers.

The main scout feature mode is deliberately small: per-tile RGB and bitplane
statistics, optionally extended with tile-position features. The native
binary-XNOR path is kept as a separate feature extractor.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from binary_layer import BinaryConvLayer
from preprocess import Tile, rgb_to_bitplanes

BITPLANE_STATS_DIM = 30
SPATIAL_FEATURE_DIM = 8
BINARY_STATS_SPATIAL_DIM = BITPLANE_STATS_DIM + SPATIAL_FEATURE_DIM
BINARY_XNOR_DIM = 64
_DEFAULT_BLOCK = 80
_DEFAULT_TILE = 160
_DEFAULT_IMAGE = 640
_DEFAULT_BLOCKS = _DEFAULT_IMAGE // _DEFAULT_BLOCK
_DEFAULT_BLOCK_PIXELS = _DEFAULT_BLOCK * _DEFAULT_BLOCK
_U8_VALUES = np.arange(256, dtype=np.float32) / 255.0
_U8_SQ_VALUES = _U8_VALUES * _U8_VALUES
_U8_BITS = ((np.arange(256, dtype=np.uint16)[:, None] >> np.arange(8, dtype=np.uint16)) & 1).astype(np.float32)


def _as_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")
    return rgb


def _rect_means_chw(values: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"values must have shape [C, H, W], got {values.shape}")

    if not tiles:
        return np.empty((0, values.shape[0]), dtype=np.float32)

    x1 = np.asarray([tile.x1 for tile in tiles], dtype=np.int32)
    y1 = np.asarray([tile.y1 for tile in tiles], dtype=np.int32)
    x2 = np.asarray([tile.x2 for tile in tiles], dtype=np.int32)
    y2 = np.asarray([tile.y2 for tile in tiles], dtype=np.int32)
    areas = ((x2 - x1) * (y2 - y1)).astype(np.float32)
    if np.any(areas <= 0):
        raise ValueError("all tiles must have positive area")

    integral = values.cumsum(axis=1).cumsum(axis=2)
    integral = np.pad(integral, ((0, 0), (1, 0), (1, 0)))
    sums = integral[:, y2, x2] - integral[:, y1, x2] - integral[:, y2, x1] + integral[:, y1, x1]
    return (sums / areas).T.astype(np.float32)


def load_resized_rgb(path: Path, img_size: int = 640) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((img_size, img_size), Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def _uses_default_grid(rgb: np.ndarray, tiles: list[Tile]) -> bool:
    if rgb.shape != (_DEFAULT_IMAGE, _DEFAULT_IMAGE, 3):
        return False

    for tile in tiles:
        if tile.x2 - tile.x1 != _DEFAULT_TILE or tile.y2 - tile.y1 != _DEFAULT_TILE:
            return False
        if tile.x1 % _DEFAULT_BLOCK or tile.y1 % _DEFAULT_BLOCK:
            return False
        last_start = _DEFAULT_IMAGE - _DEFAULT_TILE
        if not (0 <= tile.x1 <= last_start and 0 <= tile.y1 <= last_start):
            return False
    return True


def _bitplane_stats_features_default_grid(rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    if not tiles:
        return np.empty((0, BITPLANE_STATS_DIM), dtype=np.float32)

    pixels = (
        np.ascontiguousarray(rgb)
        .reshape(_DEFAULT_BLOCKS, _DEFAULT_BLOCK, _DEFAULT_BLOCKS, _DEFAULT_BLOCK, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(_DEFAULT_BLOCKS, _DEFAULT_BLOCKS, _DEFAULT_BLOCK_PIXELS, 3)
    )
    bit_means = np.empty((_DEFAULT_BLOCKS, _DEFAULT_BLOCKS, 24), dtype=np.float32)
    rgb_means = np.empty((_DEFAULT_BLOCKS, _DEFAULT_BLOCKS, 3), dtype=np.float32)
    rgb_sq_means = np.empty((_DEFAULT_BLOCKS, _DEFAULT_BLOCKS, 3), dtype=np.float32)
    for row in range(_DEFAULT_BLOCKS):
        for col in range(_DEFAULT_BLOCKS):
            for channel in range(3):
                counts = np.bincount(pixels[row, col, :, channel], minlength=256).astype(np.float32)
                bit_means[row, col, channel * 8 : (channel + 1) * 8] = counts @ _U8_BITS / _DEFAULT_BLOCK_PIXELS
                rgb_means[row, col, channel] = counts @ _U8_VALUES / _DEFAULT_BLOCK_PIXELS
                rgb_sq_means[row, col, channel] = counts @ _U8_SQ_VALUES / _DEFAULT_BLOCK_PIXELS

    rows = np.asarray([tile.y1 // _DEFAULT_BLOCK for tile in tiles], dtype=np.intp)
    cols = np.asarray([tile.x1 // _DEFAULT_BLOCK for tile in tiles], dtype=np.intp)

    def tile_average(values: np.ndarray) -> np.ndarray:
        return (
            values[rows, cols]
            + values[rows + 1, cols]
            + values[rows, cols + 1]
            + values[rows + 1, cols + 1]
        ) * 0.25

    features = np.empty((len(tiles), BITPLANE_STATS_DIM), dtype=np.float32)
    features[:, :24] = tile_average(bit_means)
    tile_rgb_means = tile_average(rgb_means)
    tile_rgb_sq_means = tile_average(rgb_sq_means)
    features[:, 24:27] = tile_rgb_means
    features[:, 27:30] = np.sqrt(np.maximum(tile_rgb_sq_means - tile_rgb_means * tile_rgb_means, 0.0))
    return features


def _bitplane_stats_features_integral(rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    planes = rgb_to_bitplanes(rgb).astype(np.float32)
    rgb_f = rgb.astype(np.float32) / 255.0
    features = np.empty((len(tiles), BITPLANE_STATS_DIM), dtype=np.float32)

    rgb_chw = rgb_f.transpose(2, 0, 1)
    rgb_means = _rect_means_chw(rgb_chw, tiles)
    rgb_sq_means = _rect_means_chw(rgb_chw * rgb_chw, tiles)
    features[:, :24] = _rect_means_chw(planes, tiles)
    features[:, 24:27] = rgb_means
    features[:, 27:30] = np.sqrt(np.maximum(rgb_sq_means - rgb_means * rgb_means, 0.0))

    return features


def bitplane_stats_features(rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    rgb = _as_rgb_u8(rgb)
    if _uses_default_grid(rgb, tiles):
        return _bitplane_stats_features_default_grid(rgb, tiles)
    return _bitplane_stats_features_integral(rgb, tiles)


def spatial_tile_features(tiles: list[Tile], img_size: int = 640) -> np.ndarray:
    if not tiles:
        return np.empty((0, SPATIAL_FEATURE_DIM), dtype=np.float32)

    max_row = max(tile.row for tile in tiles) or 1
    max_col = max(tile.col for tile in tiles) or 1
    features = np.empty((len(tiles), SPATIAL_FEATURE_DIM), dtype=np.float32)

    for idx, tile in enumerate(tiles):
        cx = ((tile.x1 + tile.x2) * 0.5) / img_size
        cy = ((tile.y1 + tile.y2) * 0.5) / img_size
        features[idx] = [
            tile.row / max_row,
            tile.col / max_col,
            cx,
            cy,
            abs(cx - 0.5),
            abs(cy - 0.5),
            float(tile.x1 == 0 or tile.x2 == img_size),
            float(tile.y1 == 0 or tile.y2 == img_size),
        ]

    return features


class BinaryXnorExtractor:
    def __init__(
        self,
        n_filters: int = BINARY_XNOR_DIM,
        kernel_size: int = 3,
        threshold: int = 0,
        seed: int = 42,
        input_channels: int = 24,
    ):
        self.threshold = int(threshold)
        self.seed = int(seed)
        self.layer = BinaryConvLayer(
            n_filters=n_filters,
            n_ch=input_channels,
            kH=kernel_size,
            kW=kernel_size,
            seed=seed,
        )

    def weight_hash(self) -> str:
        weights = np.ascontiguousarray(self.layer.weights, dtype=np.int8)
        return hashlib.sha256(weights.tobytes()).hexdigest()

    def metadata(self) -> dict:
        return {
            "binary_filters": int(self.layer.n_filters),
            "binary_kernel_size": int(self.layer.kH),
            "binary_threshold": int(self.threshold),
            "binary_seed": int(self.seed),
            "binary_input_channels": int(self.layer.n_ch),
            "binary_weight_sha256": self.weight_hash(),
        }

    def features(self, rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
        rgb = _as_rgb_u8(rgb)

        planes = rgb_to_bitplanes(rgb)
        if planes.shape[0] != self.layer.n_ch:
            raise ValueError(f"expected {self.layer.n_ch} bitplanes, got {planes.shape[0]}")
        scores = self.layer.forward(planes)
        binary_maps = self.layer.apply_threshold(scores, threshold=self.threshold).astype(np.float32)
        return _rect_means_chw(binary_maps, tiles)


def binary_xnor_features(
    rgb: np.ndarray,
    tiles: list[Tile],
    n_filters: int = BINARY_XNOR_DIM,
    kernel_size: int = 3,
    threshold: int = 0,
    seed: int = 42,
) -> np.ndarray:
    extractor = BinaryXnorExtractor(
        n_filters=n_filters,
        kernel_size=kernel_size,
        threshold=threshold,
        seed=seed,
    )
    return extractor.features(rgb, tiles)
