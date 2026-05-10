"""
Create JSONL tile-occupancy labels from VisDrone DET annotations.

Outputs:
  data/tile_dataset/tile_grid.json
  data/tile_dataset/train_tiles.jsonl
  data/tile_dataset/val_tiles.jsonl
  data/tile_dataset/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import (  # noqa: E402
    deterministic_split,
    label_tiles,
    list_visdrone_stems,
    load_image_size,
    make_tiles,
    parse_visdrone_annotations,
    scale_boxes_to_resized,
    tile_grid_metadata,
    write_json,
    write_jsonl,
)


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def limited(stems: list[str], max_images: int) -> list[str]:
    return stems[:max_images] if max_images else stems


def resolve_splits(args: argparse.Namespace) -> dict[str, tuple[Path, list[str]]]:
    train_stems = list_visdrone_stems(args.data_root)
    if not train_stems:
        raise FileNotFoundError(f"no train images found in {args.data_root / 'images'}")

    if args.split_mode in {"auto", "official"} and args.val_root.exists():
        val_stems = list_visdrone_stems(args.val_root)
        if not val_stems:
            raise FileNotFoundError(f"no val images found in {args.val_root / 'images'}")
        return {
            "train": (args.data_root, limited(train_stems, args.max_images)),
            "val": (args.val_root, limited(val_stems, args.max_images)),
        }

    if args.split_mode == "official":
        raise FileNotFoundError(f"--split-mode official requires --val-root: {args.val_root}")

    stems = limited(train_stems, args.max_images)
    train, val = deterministic_split(stems, val_frac=args.val_frac, seed=args.seed)
    return {
        "train": (args.data_root, train),
        "val": (args.data_root, val),
    }


def make_records_for_image(
    split: str,
    data_root: Path,
    stem: str,
    args: argparse.Namespace,
    tiles,
) -> tuple[list[dict], dict]:
    image_path = data_root / "images" / f"{stem}.jpg"
    ann_path = data_root / "annotations" / f"{stem}.txt"
    if not image_path.exists():
        raise FileNotFoundError(f"missing image: {image_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"missing annotation: {ann_path}")

    orig_size = load_image_size(image_path)
    boxes, parse_stats = parse_visdrone_annotations(ann_path)
    boxes = scale_boxes_to_resized(boxes, orig_size=orig_size, img_size=args.img_size)
    labels = label_tiles(tiles, boxes)

    records = []
    for item in labels:
        tile = item["tile"]
        records.append(
            {
                "stem": stem,
                "split": split,
                "source_split": data_root.name,
                "image_path": repo_path(image_path),
                "annotation_path": repo_path(ann_path),
                "orig_size": list(orig_size),
                "img_size": args.img_size,
                "resize": f"stretch_{args.img_size}",
                "tile_id": tile.tile_id,
                "tile_row": tile.row,
                "tile_col": tile.col,
                "tile": tile.xyxy(),
                "label": item["label"],
                "box_indices": item["box_indices"],
                "classes": item["classes"],
                "visdrone_category_ids": item["visdrone_category_ids"],
                "n_objects": item["n_objects"],
                "n_boxes": len(boxes),
            }
        )

    summary = {
        "valid_boxes": len(boxes),
        "positive_tiles": sum(record["label"] for record in records),
        "parse_stats": parse_stats,
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-train")
    parser.add_argument("--val-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-val")
    parser.add_argument("--split-mode", choices=["auto", "official", "internal"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tile_dataset")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--positive-rule", choices=["center_in_tile"], default="center_in_tile")
    parser.add_argument("--max-images", type=int, default=0, help="Images per split for official mode; total before split for internal mode.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tiles = make_tiles(args.img_size, args.tile_size, args.stride)
    splits = resolve_splits(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "tile_grid.json", tile_grid_metadata(args.img_size, args.tile_size, args.stride))

    summary = {
        "img_size": args.img_size,
        "resize": f"stretch_{args.img_size}",
        "tile_size": args.tile_size,
        "stride": args.stride,
        "tiles_per_image": len(tiles),
        "positive_rule": args.positive_rule,
        "split_mode": args.split_mode,
        "splits": {},
    }

    try:
        for split, (data_root, stems) in splits.items():
            records = []
            split_summary = {
                "source_root": repo_path(data_root),
                "images": len(stems),
                "records": 0,
                "positive_tiles": 0,
                "valid_boxes": 0,
            }

            for stem in stems:
                image_records, image_summary = make_records_for_image(split, data_root, stem, args, tiles)
                records.extend(image_records)
                split_summary["positive_tiles"] += image_summary["positive_tiles"]
                split_summary["valid_boxes"] += image_summary["valid_boxes"]

            split_summary["records"] = len(records)
            summary["splits"][split] = split_summary
            write_jsonl(args.out_dir / f"{split}_tiles.jsonl", records)

        write_json(args.out_dir / "summary.json", summary)

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== make_tile_dataset.py ===")
    print(f"PASS  out_dir={args.out_dir}")
    print(f"PASS  tiles_per_image={len(tiles)}")
    for split, stats in summary["splits"].items():
        print(
            f"PASS  {split}: images={stats['images']} records={stats['records']} "
            f"positive_tiles={stats['positive_tiles']} valid_boxes={stats['valid_boxes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
