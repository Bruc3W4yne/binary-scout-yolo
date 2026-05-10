"""
Extract scout features for a tile JSONL split.

Current feature mode:
  bitplane-stats -> 30 float32 features per tile
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import Tile  # noqa: E402
from scout import (  # noqa: E402
    BINARY_XNOR_DIM,
    BITPLANE_STATS_DIM,
    binary_xnor_features,
    bitplane_stats_features,
    load_resized_rgb,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing tile JSONL: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def tile_from_record(record: dict) -> Tile:
    x1, y1, x2, y2 = record["tile"]
    return Tile(
        tile_id=int(record["tile_id"]),
        row=int(record["tile_row"]),
        col=int(record["tile_col"]),
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
    )


def grouped_by_stem(records: list[dict]) -> OrderedDict[str, list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(record["stem"], []).append(record)
    return groups


def extract_features(records: list[dict], args: argparse.Namespace) -> dict[str, np.ndarray]:
    groups = grouped_by_stem(records)
    if args.max_images:
        groups = OrderedDict(list(groups.items())[:args.max_images])

    features = []
    labels = []
    stems = []
    tile_ids = []
    n_objects = []
    n_boxes = []
    box_indices_json = []

    for stem, image_records in groups.items():
        image_records = sorted(image_records, key=lambda item: item["tile_id"])
        rgb = load_resized_rgb(resolve_path(image_records[0]["image_path"]), img_size=args.img_size)
        tiles = [tile_from_record(record) for record in image_records]
        if args.feature_mode == "bitplane-stats":
            image_features = bitplane_stats_features(rgb, tiles)
        else:
            image_features = binary_xnor_features(
                rgb,
                tiles,
                n_filters=args.binary_filters,
                kernel_size=args.binary_kernel_size,
                threshold=args.binary_threshold,
                seed=args.seed,
            )

        features.append(image_features)
        labels.extend(int(record["label"]) for record in image_records)
        stems.extend(stem for _ in image_records)
        tile_ids.extend(int(record["tile_id"]) for record in image_records)
        n_objects.extend(int(record["n_objects"]) for record in image_records)
        n_boxes.extend(int(record["n_boxes"]) for record in image_records)
        box_indices_json.extend(json.dumps(record["box_indices"]) for record in image_records)

    if not features:
        raise ValueError("no features extracted; input JSONL is empty")

    return {
        "features": np.vstack(features).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "stems": np.asarray(stems),
        "tile_ids": np.asarray(tile_ids, dtype=np.int32),
        "n_objects": np.asarray(n_objects, dtype=np.int16),
        "n_boxes": np.asarray(n_boxes, dtype=np.int16),
        "box_indices_json": np.asarray(box_indices_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, default=ROOT / "data" / "tile_dataset")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tile_features")
    parser.add_argument("--feature-mode", choices=["bitplane-stats", "binary-xnor"], default="bitplane-stats")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--binary-filters", type=int, default=BINARY_XNOR_DIM)
    parser.add_argument("--binary-kernel-size", type=int, default=3)
    parser.add_argument("--binary-threshold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = read_jsonl(args.tile_dir / f"{args.split}_tiles.jsonl")
        arrays = extract_features(records, args)
        feature_dim = BITPLANE_STATS_DIM if args.feature_mode == "bitplane-stats" else args.binary_filters

        metadata = {
            "feature_mode": args.feature_mode,
            "feature_dim": feature_dim,
            "split": args.split,
            "img_size": args.img_size,
            "n_tiles": int(arrays["features"].shape[0]),
            "requires_c_kernel": args.feature_mode == "binary-xnor",
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{args.feature_mode.replace('-', '_')}_{args.split}.npz"
        np.savez_compressed(out_path, **arrays)

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== extract_tile_features.py ===")
    print(f"PASS  out={out_path}")
    print(f"PASS  features={arrays['features'].shape}")
    print(f"PASS  positives={int(arrays['labels'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
