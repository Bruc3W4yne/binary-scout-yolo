"""
Verify that a VisDrone DET split has the expected image/annotation structure.

This script does not create derived data. It only checks local dataset roots and
annotation parsing before tile generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_image_size, parse_visdrone_annotations  # noqa: E402


def verify_root(root: Path, max_images: int) -> dict:
    image_dir = root / "images"
    ann_dir = root / "annotations"
    if not image_dir.exists():
        raise FileNotFoundError(f"missing image directory: {image_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"missing annotation directory: {ann_dir}")

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"no .jpg files found in {image_dir}")

    if max_images:
        image_paths = image_paths[:max_images]

    totals = {
        "images_checked": 0,
        "missing_annotations": 0,
        "valid_boxes": 0,
        "ignored_rows": 0,
        "skipped_category_rows": 0,
        "invalid_box_rows": 0,
        "malformed_rows": 0,
    }

    for image_path in image_paths:
        ann_path = ann_dir / f"{image_path.stem}.txt"
        if not ann_path.exists():
            totals["missing_annotations"] += 1
            continue

        width, height = load_image_size(image_path)
        if width < 1 or height < 1:
            raise AssertionError(f"invalid image size for {image_path}: {(width, height)}")

        _, stats = parse_visdrone_annotations(ann_path)
        totals["images_checked"] += 1
        totals["valid_boxes"] += stats["valid"]
        totals["ignored_rows"] += stats["ignored"]
        totals["skipped_category_rows"] += stats["skipped_category"]
        totals["invalid_box_rows"] += stats["invalid_box"]
        totals["malformed_rows"] += stats["malformed"]

    if totals["missing_annotations"]:
        raise AssertionError(f"missing annotations for {totals['missing_annotations']} checked images")
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-train")
    parser.add_argument("--val-root", type=Path, default=ROOT / "data" / "VisDrone2019-DET-val")
    parser.add_argument("--split-mode", choices=["auto", "official", "internal"], default="auto")
    parser.add_argument("--max-images", type=int, default=25, help="Images to check per split; 0 checks all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=== verify_visdrone_dataset.py ===")

    try:
        train = verify_root(args.data_root, args.max_images)
        print(f"PASS  train {args.data_root}")
        print(f"      {train}")

        if args.split_mode in {"official", "auto"} and args.val_root.exists():
            val = verify_root(args.val_root, args.max_images)
            print(f"PASS  val   {args.val_root}")
            print(f"      {val}")
        elif args.split_mode == "official":
            raise FileNotFoundError(f"--split-mode official requires --val-root: {args.val_root}")
        else:
            print("PASS  official val split not required for this mode")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
