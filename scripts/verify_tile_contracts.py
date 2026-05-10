"""
Synthetic verifier for preprocessing contracts.

This does not require VisDrone. It tests the coordinate, parser, tiling,
bitplane, selected-area, and split rules the full pipeline will rely on.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import (  # noqa: E402
    Box,
    deterministic_split,
    label_tiles,
    make_tiles,
    parse_visdrone_annotations,
    rgb_to_bitplanes,
    scale_boxes_to_resized,
    selected_area_fraction,
)


def assert_close(actual: float, expected: float, name: str, eps: float = 1e-9) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def reconstruct_rgb(planes: np.ndarray) -> np.ndarray:
    if planes.shape[0] != 24:
        raise AssertionError(f"expected 24 bitplanes, got {planes.shape}")

    _, height, width = planes.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        for bit in range(8):
            rgb[:, :, channel] |= (planes[channel * 8 + bit].astype(np.uint8) << bit)
    return rgb


def verify_grid() -> None:
    tiles = make_tiles(640, 160, 80)
    if len(tiles) != 49:
        raise AssertionError(f"expected 49 tiles, got {len(tiles)}")
    if tiles[0].xyxy() != [0, 0, 160, 160]:
        raise AssertionError(f"bad first tile: {tiles[0].xyxy()}")
    if tiles[-1].xyxy() != [480, 480, 640, 640]:
        raise AssertionError(f"bad last tile: {tiles[-1].xyxy()}")

    covered = np.zeros((640, 640), dtype=np.bool_)
    for tile in tiles:
        covered[tile.y1:tile.y2, tile.x1:tile.x2] = True
    if not bool(covered.all()):
        raise AssertionError("tile grid does not cover all pixels")


def verify_tile_labels() -> None:
    tiles = make_tiles(640, 160, 80)
    boundary_box = Box(159, 79, 161, 81, class_id=0, category_id=1)
    labels = label_tiles(tiles, [boundary_box])

    if labels[0]["box_indices"]:
        raise AssertionError("center x=160 must not be inside first tile [0,0,160,160]")
    if not any(0 in label["box_indices"] for label in labels[1:]):
        raise AssertionError("center x=160 should be covered by a later overlapping tile")


def verify_parser_and_scaling() -> None:
    annotation = "\n".join(
        [
            "10,20,30,40,1,1,0,0",
            "100,100,20,20,1,10,0,0",
            "0,0,50,50,0,1,0,0",
            "0,0,50,50,1,0,0,0",
            "0,0,50,50,1,11,0,0",
            "0,0,-1,50,1,1,0,0",
            "malformed",
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        ann_path = Path(tmp) / "sample.txt"
        ann_path.write_text(annotation)
        boxes, stats = parse_visdrone_annotations(ann_path)

    if len(boxes) != 2 or stats["valid"] != 2:
        raise AssertionError(f"expected 2 valid boxes, got boxes={len(boxes)} stats={stats}")
    if stats["ignored"] != 2 or stats["skipped_category"] != 1:
        raise AssertionError(f"bad skip stats: {stats}")
    if stats["invalid_box"] != 1 or stats["malformed"] != 1:
        raise AssertionError(f"bad invalid/malformed stats: {stats}")

    scaled = scale_boxes_to_resized([boxes[0]], orig_size=(100, 200), img_size=640)
    if len(scaled) != 1:
        raise AssertionError("scaled valid box was dropped")
    expected = [64.0, 64.0, 256.0, 192.0]
    for actual, want in zip(scaled[0].xyxy(), expected):
        assert_close(actual, want, "scaled box")


def verify_bitplanes() -> None:
    rgb = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[127, 128, 255], [16, 32, 64]],
        ],
        dtype=np.uint8,
    )
    planes = rgb_to_bitplanes(rgb)
    if planes.shape != (24, 2, 2) or planes.dtype != np.uint8:
        raise AssertionError(f"unexpected bitplane shape/dtype: {planes.shape} {planes.dtype}")
    if not np.array_equal(reconstruct_rgb(planes), rgb):
        raise AssertionError("bitplanes do not reconstruct original RGB")


def verify_selected_area() -> None:
    tiles = make_tiles(640, 160, 80)
    area = selected_area_fraction([tiles[0], tiles[1]], img_size=640)
    expected = (160 * 160 + 160 * 160 - 80 * 160) / (640 * 640)
    assert_close(area, expected, "selected area fraction")


def verify_split() -> None:
    stems = [f"img_{i:03d}" for i in range(10)]
    train_a, val_a = deterministic_split(stems, val_frac=0.2, seed=42)
    train_b, val_b = deterministic_split(reversed(stems), val_frac=0.2, seed=42)
    if (train_a, val_a) != (train_b, val_b):
        raise AssertionError("deterministic split is not stable")
    if set(train_a) & set(val_a):
        raise AssertionError("train/val split overlaps")
    if len(val_a) != 2:
        raise AssertionError(f"expected 2 val stems, got {len(val_a)}")


def main() -> int:
    checks = [
        ("tile_grid_49_full_coverage", verify_grid),
        ("center_in_tile_labels", verify_tile_labels),
        ("visdrone_parser_and_scaling", verify_parser_and_scaling),
        ("rgb_bitplanes_roundtrip", verify_bitplanes),
        ("selected_area_union", verify_selected_area),
        ("deterministic_split", verify_split),
    ]

    print("=== verify_tile_contracts.py ===")
    try:
        for name, fn in checks:
            fn()
            print(f"PASS  {name}")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print()
    print("All tile-contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
