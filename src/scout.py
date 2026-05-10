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
SPATIAL_FEATURE_DIM = 8
BINARY_STATS_SPATIAL_DIM = BITPLANE_STATS_DIM + SPATIAL_FEATURE_DIM
BINARY_XNOR_DIM = 64


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


def bitplane_stats_features(rgb: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")

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
    return _rect_means_chw(binary_maps, tiles)
