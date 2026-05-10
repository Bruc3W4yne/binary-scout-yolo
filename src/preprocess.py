"""
Shared VisDrone and tile-grid preprocessing contracts.

This module intentionally avoids model code. It defines the small set of data
shapes the scout, router, and detector pipeline will agree on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

VALID_VISDRONE_CATEGORY_IDS = set(range(1, 11))
IGNORED_VISDRONE_CATEGORY_IDS = {0, 11}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    category_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Tile:
    tile_id: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    def xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def contains_center(self, box: Box) -> bool:
        cx, cy = box.center
        return self.x1 <= cx < self.x2 and self.y1 <= cy < self.y2


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def list_visdrone_stems(data_root: Path) -> list[str]:
    image_dir = data_root / "images"
    return [path.stem for path in sorted(image_dir.glob("*.jpg"))]


def parse_visdrone_annotations(path: Path) -> tuple[list[Box], dict[str, int]]:
    stats = {
        "rows": 0,
        "valid": 0,
        "ignored": 0,
        "skipped_category": 0,
        "invalid_box": 0,
        "malformed": 0,
    }
    boxes: list[Box] = []

    if not path.exists():
        return boxes, stats

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        stats["rows"] += 1
        parts = line.split(",")
        if len(parts) < 6:
            stats["malformed"] += 1
            continue

        try:
            x1, y1, w, h = (float(parts[i]) for i in range(4))
            score = int(float(parts[4]))
            category_id = int(float(parts[5]))
        except ValueError:
            stats["malformed"] += 1
            continue

        if score == 0 or category_id == 0:
            stats["ignored"] += 1
            continue
        if category_id not in VALID_VISDRONE_CATEGORY_IDS:
            stats["skipped_category"] += 1
            continue
        if w <= 0 or h <= 0:
            stats["invalid_box"] += 1
            continue

        boxes.append(
            Box(
                x1=x1,
                y1=y1,
                x2=x1 + w,
                y2=y1 + h,
                class_id=category_id - 1,
                category_id=category_id,
            )
        )
        stats["valid"] += 1

    return boxes, stats


def scale_boxes_to_resized(
    boxes: Iterable[Box],
    orig_size: tuple[int, int],
    img_size: int,
) -> list[Box]:
    orig_w, orig_h = orig_size
    if orig_w < 1 or orig_h < 1:
        raise ValueError(f"original image size must be positive, got {orig_size}")

    sx = img_size / orig_w
    sy = img_size / orig_h
    scaled: list[Box] = []

    for box in boxes:
        x1 = min(max(box.x1 * sx, 0.0), float(img_size))
        y1 = min(max(box.y1 * sy, 0.0), float(img_size))
        x2 = min(max(box.x2 * sx, 0.0), float(img_size))
        y2 = min(max(box.y2 * sy, 0.0), float(img_size))
        if x2 <= x1 or y2 <= y1:
            continue
        scaled.append(Box(x1, y1, x2, y2, box.class_id, box.category_id))

    return scaled


def tile_starts(img_size: int, tile_size: int, stride: int) -> list[int]:
    if img_size < 1:
        raise ValueError(f"img_size must be positive, got {img_size}")
    if tile_size < 1 or tile_size > img_size:
        raise ValueError(f"tile_size must be in [1, img_size], got {tile_size}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")

    last = img_size - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(img_size: int = 640, tile_size: int = 160, stride: int = 80) -> list[Tile]:
    starts_x = tile_starts(img_size, tile_size, stride)
    starts_y = tile_starts(img_size, tile_size, stride)
    tiles: list[Tile] = []

    for row, y1 in enumerate(starts_y):
        for col, x1 in enumerate(starts_x):
            tiles.append(
                Tile(
                    tile_id=len(tiles),
                    row=row,
                    col=col,
                    x1=x1,
                    y1=y1,
                    x2=x1 + tile_size,
                    y2=y1 + tile_size,
                )
            )

    return tiles


def tile_grid_metadata(img_size: int = 640, tile_size: int = 160, stride: int = 80) -> dict:
    starts_x = tile_starts(img_size, tile_size, stride)
    starts_y = tile_starts(img_size, tile_size, stride)
    return {
        "img_size": img_size,
        "tile_size": tile_size,
        "stride": stride,
        "starts_x": starts_x,
        "starts_y": starts_y,
        "n_rows": len(starts_y),
        "n_cols": len(starts_x),
        "n_tiles": len(starts_x) * len(starts_y),
        "coordinate_convention": "x1_y1_inclusive_x2_y2_exclusive",
    }


def label_tiles(tiles: Iterable[Tile], boxes: list[Box]) -> list[dict]:
    labels: list[dict] = []
    for tile in tiles:
        box_indices = [idx for idx, box in enumerate(boxes) if tile.contains_center(box)]
        labels.append(
            {
                "tile": tile,
                "label": int(bool(box_indices)),
                "box_indices": box_indices,
                "classes": [boxes[idx].class_id for idx in box_indices],
                "visdrone_category_ids": [boxes[idx].category_id for idx in box_indices],
                "n_objects": len(box_indices),
            }
        )
    return labels


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            clean = {
                key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
                for key, value in record.items()
            }
            handle.write(json.dumps(clean, separators=(",", ":")) + "\n")
