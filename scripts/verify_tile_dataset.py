"""
Verify JSONL tile datasets produced by scripts/make_tile_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import make_tiles  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL file: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_no}: invalid JSON") from exc
    return records


def verify_split(records: list[dict], split: str, tiles, args: argparse.Namespace) -> dict:
    groups = defaultdict(list)
    positives = 0

    tile_by_id = {tile.tile_id: tile for tile in tiles}
    for record in records:
        if record.get("split") != split:
            raise AssertionError(f"record split mismatch: expected {split}, got {record.get('split')}")
        if record.get("img_size") != args.expect_img_size:
            raise AssertionError(f"bad img_size in {record.get('stem')}: {record.get('img_size')}")
        if record.get("resize") != f"stretch_{args.expect_img_size}":
            raise AssertionError(f"bad resize in {record.get('stem')}: {record.get('resize')}")

        tile_id = record.get("tile_id")
        if tile_id not in tile_by_id:
            raise AssertionError(f"unknown tile_id={tile_id}")
        if record.get("tile") != tile_by_id[tile_id].xyxy():
            raise AssertionError(f"bad tile coords for tile_id={tile_id}: {record.get('tile')}")
        if record.get("tile_row") != tile_by_id[tile_id].row:
            raise AssertionError(f"bad tile_row for tile_id={tile_id}")
        if record.get("tile_col") != tile_by_id[tile_id].col:
            raise AssertionError(f"bad tile_col for tile_id={tile_id}")

        n_objects = record.get("n_objects")
        if record.get("label") not in (0, 1):
            raise AssertionError(f"label must be 0 or 1, got {record.get('label')}")
        if record["label"] != int(n_objects > 0):
            raise AssertionError("label does not match n_objects")
        if len(record.get("box_indices", [])) != n_objects:
            raise AssertionError("box_indices length does not match n_objects")
        if len(record.get("classes", [])) != n_objects:
            raise AssertionError("classes length does not match n_objects")
        if len(record.get("visdrone_category_ids", [])) != n_objects:
            raise AssertionError("category ID length does not match n_objects")

        positives += record["label"]
        groups[record["stem"]].append(record)

    for stem, image_records in groups.items():
        if len(image_records) != args.expect_tiles_per_image:
            raise AssertionError(
                f"{split}/{stem}: expected {args.expect_tiles_per_image} records, got {len(image_records)}"
            )
        tile_ids = sorted(record["tile_id"] for record in image_records)
        if tile_ids != list(range(args.expect_tiles_per_image)):
            raise AssertionError(f"{split}/{stem}: non-contiguous tile IDs")

    return {"images": len(groups), "records": len(records), "positive_tiles": positives}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, default=ROOT / "data" / "tile_dataset")
    parser.add_argument("--expect-img-size", type=int, default=640)
    parser.add_argument("--expect-tile-size", type=int, default=160)
    parser.add_argument("--expect-stride", type=int, default=80)
    parser.add_argument("--expect-tiles-per-image", type=int, default=49)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== verify_tile_dataset.py ===")

    try:
        grid_path = args.tile_dir / "tile_grid.json"
        if not grid_path.exists():
            raise FileNotFoundError(f"missing tile grid: {grid_path}")
        grid = json.loads(grid_path.read_text())
        expected_grid = {
            "img_size": args.expect_img_size,
            "tile_size": args.expect_tile_size,
            "stride": args.expect_stride,
            "n_tiles": args.expect_tiles_per_image,
        }
        for key, expected in expected_grid.items():
            if grid.get(key) != expected:
                raise AssertionError(f"tile_grid[{key}] expected {expected}, got {grid.get(key)}")

        tiles = make_tiles(args.expect_img_size, args.expect_tile_size, args.expect_stride)
        train_records = read_jsonl(args.tile_dir / "train_tiles.jsonl")
        val_records = read_jsonl(args.tile_dir / "val_tiles.jsonl")
        train = verify_split(train_records, "train", tiles, args)
        val = verify_split(val_records, "val", tiles, args)

        train_stems = {record["stem"] for record in train_records}
        val_stems = {record["stem"] for record in val_records}
        if train_stems & val_stems:
            raise AssertionError("train/val stems overlap")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS  train {train}")
    print(f"PASS  val   {val}")
    print("PASS  train/val disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
