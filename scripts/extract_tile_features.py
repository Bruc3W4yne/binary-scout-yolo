"""
Extract scout features for a tile JSONL split.

Feature modes:
  bitplane-stats     -> 30 float32 features per tile
  binary-xnor        -> C-kernel binary convolution summaries per tile
  binary-xnor-hybrid -> binary-XNOR summaries plus tile-position features
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
    BinaryXnorExtractor,
    SPATIAL_FEATURE_DIM,
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
    total_images = len(groups)
    stem_to_id = {stem: idx for idx, stem in enumerate(groups)}

    features = []
    labels = []
    image_ids = []
    tile_ids = []
    n_objects = []
    n_boxes = []
    box_indices = []
    box_offsets = [0]
    binary_extractor = None
    if args.feature_mode.startswith("binary-xnor"):
        binary_extractor = BinaryXnorExtractor(
            n_filters=args.binary_filters,
            kernel_size=args.binary_kernel_size,
            threshold=args.binary_threshold,
            seed=args.seed,
        )

    for image_idx, (stem, image_records) in enumerate(groups.items(), start=1):
        image_records = sorted(image_records, key=lambda item: item["tile_id"])
        rgb = load_resized_rgb(resolve_path(image_records[0]["image_path"]), img_size=args.img_size)
        tiles = [tile_from_record(record) for record in image_records]
        if args.feature_mode == "bitplane-stats":
            image_features = bitplane_stats_features(rgb, tiles)
        elif args.feature_mode == "binary-xnor":
            if binary_extractor is None:
                raise RuntimeError("binary extractor was not initialized")
            image_features = binary_extractor.features(rgb, tiles)
        elif args.feature_mode == "binary-xnor-hybrid":
            if binary_extractor is None:
                raise RuntimeError("binary extractor was not initialized")
            image_features = binary_extractor.hybrid_features(rgb, tiles)
        else:
            raise ValueError(f"unsupported feature mode: {args.feature_mode}")

        features.append(image_features)
        for record in image_records:
            indices = [int(value) for value in record["box_indices"]]
            labels.append(int(record["label"]))
            image_ids.append(stem_to_id[stem])
            tile_ids.append(int(record["tile_id"]))
            n_objects.append(int(record["n_objects"]))
            n_boxes.append(int(record["n_boxes"]))
            box_indices.extend(indices)
            box_offsets.append(len(box_indices))
        if args.progress_every and (image_idx == total_images or image_idx % args.progress_every == 0):
            print(f"progress {image_idx}/{total_images} images", file=sys.stderr, flush=True)

    if not features:
        raise ValueError("no features extracted; input JSONL is empty")

    arrays = {
        "features": np.vstack(features).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "image_ids": np.asarray(image_ids, dtype=np.uint32),
        "image_stems": np.asarray(list(groups.keys())),
        "tile_ids": np.asarray(tile_ids, dtype=np.uint8),
        "n_objects": np.asarray(n_objects, dtype=np.uint16),
        "n_boxes": np.asarray(n_boxes, dtype=np.uint16),
        "box_offsets": np.asarray(box_offsets, dtype=np.uint32),
        "box_indices": np.asarray(box_indices, dtype=np.uint16),
    }
    if binary_extractor is not None:
        arrays["binary_metadata_json"] = np.asarray(json.dumps(binary_extractor.metadata()))
    return arrays


def output_path(args: argparse.Namespace) -> Path:
    mode = args.feature_mode.replace("-", "_")
    limit = f"_n{args.max_images}" if args.max_images else ""
    return args.out_dir / f"{mode}_{args.split}{limit}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, default=ROOT / "data" / "tile_dataset")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "tile_features")
    parser.add_argument(
        "--feature-mode",
        choices=["bitplane-stats", "binary-xnor", "binary-xnor-hybrid"],
        default="bitplane-stats",
    )
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--binary-filters", type=int, default=BINARY_XNOR_DIM)
    parser.add_argument("--binary-kernel-size", type=int, default=3)
    parser.add_argument("--binary-threshold", type=int, default=0)
    parser.add_argument("--compress", action="store_true", help="write a smaller but slower compressed NPZ")
    parser.add_argument("--progress-every", type=int, default=500, help="print progress every N images; 0 disables it")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = read_jsonl(args.tile_dir / f"{args.split}_tiles.jsonl")
        arrays = extract_features(records, args)
        if args.feature_mode == "bitplane-stats":
            feature_dim = BITPLANE_STATS_DIM
        elif args.feature_mode == "binary-xnor-hybrid":
            feature_dim = args.binary_filters + SPATIAL_FEATURE_DIM
        else:
            feature_dim = args.binary_filters

        metadata = {
            "feature_mode": args.feature_mode,
            "feature_dim": feature_dim,
            "schema_version": 2,
            "split": args.split,
            "img_size": args.img_size,
            "n_tiles": int(arrays["features"].shape[0]),
            "n_images": int(len(arrays["image_stems"])),
            "max_images": int(args.max_images),
            "requires_c_kernel": args.feature_mode.startswith("binary-xnor"),
        }
        if "binary_metadata_json" in arrays:
            metadata.update(json.loads(str(arrays["binary_metadata_json"].item())))
        if args.feature_mode == "binary-xnor-hybrid":
            metadata["hybrid_features"] = [
                "binary_xnor_tile_summaries",
                "row_norm",
                "col_norm",
                "center_x",
                "center_y",
                "abs_center_x",
                "abs_center_y",
                "touches_x_border",
                "touches_y_border",
            ]
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))

        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_path(args)
        save = np.savez_compressed if args.compress else np.savez
        save(out_path, **arrays)

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
