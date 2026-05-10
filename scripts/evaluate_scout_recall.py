"""
Evaluate top-K object-center recall from tile scout scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_feature_file(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing feature file: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def load_scout_scores(feature_data: dict[str, np.ndarray], checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    features = torch.from_numpy(feature_data["features"].astype(np.float32))
    mean = checkpoint["mean"].float()
    std = checkpoint["std"].float()
    model = torch.nn.Linear(int(checkpoint["feature_dim"]), 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model((features - mean) / std).squeeze(1)
    return logits.numpy()


def load_random_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n).astype(np.float32)


def load_oracle_scores(feature_data: dict[str, np.ndarray]) -> np.ndarray:
    return feature_data["n_objects"].astype(np.float32)


def parse_box_indices(raw: str) -> list[int]:
    return [int(value) for value in json.loads(str(raw))]


def feature_stems(feature_data: dict[str, np.ndarray]) -> np.ndarray:
    if "stems" in feature_data:
        return feature_data["stems"].astype(str)
    return feature_data["image_stems"].astype(str)[feature_data["image_ids"].astype(np.int64)]


def tile_box_indices(feature_data: dict[str, np.ndarray], idx: int) -> list[int]:
    if "box_indices_json" in feature_data:
        return parse_box_indices(feature_data["box_indices_json"][idx])
    offsets = feature_data["box_offsets"].astype(np.int64)
    flat = feature_data["box_indices"].astype(np.int64)
    return [int(value) for value in flat[offsets[idx] : offsets[idx + 1]]]


def evaluate_topk(feature_data: dict[str, np.ndarray], scores: np.ndarray, top_k_values: list[int]) -> dict:
    groups = defaultdict(list)
    stems = feature_stems(feature_data)
    tile_ids = feature_data["tile_ids"].astype(np.int32)
    n_boxes = feature_data["n_boxes"].astype(np.int32)

    for idx, stem in enumerate(stems):
        groups[stem].append(idx)

    results = {}
    for top_k in top_k_values:
        covered_total = 0
        object_total = 0
        positive_tile_total = 0
        selected_positive_tiles = 0

        for indices in groups.values():
            image_n_boxes = int(max(n_boxes[idx] for idx in indices))
            if image_n_boxes == 0:
                continue

            ranked = sorted(indices, key=lambda idx: (-float(scores[idx]), int(tile_ids[idx])))
            selected = ranked[: min(top_k, len(ranked))]
            covered = set()
            positives = 0
            selected_positives = 0

            for idx in indices:
                if tile_box_indices(feature_data, idx):
                    positives += 1
            for idx in selected:
                box_indices = tile_box_indices(feature_data, idx)
                if box_indices:
                    selected_positives += 1
                covered.update(box_indices)

            object_total += image_n_boxes
            covered_total += len(covered)
            positive_tile_total += positives
            selected_positive_tiles += selected_positives

        results[str(top_k)] = {
            "object_total": object_total,
            "covered_objects": covered_total,
            "object_recall": covered_total / object_total if object_total else 0.0,
            "positive_tile_total": positive_tile_total,
            "selected_positive_tiles": selected_positive_tiles,
            "positive_tile_recall": (
                selected_positive_tiles / positive_tile_total if positive_tile_total else 0.0
            ),
        }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "tile_features" / "bitplane_stats_val.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "scout" / "scout_bitplane_stats.pt")
    parser.add_argument("--mode", choices=["scout", "random", "oracle"], default="scout")
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_feature_file(args.features)
        if args.mode == "scout":
            scores = load_scout_scores(data, args.checkpoint)
        elif args.mode == "oracle":
            scores = load_oracle_scores(data)
        else:
            scores = load_random_scores(len(data["labels"]), args.seed)
        results = {
            "mode": args.mode,
            "features": str(args.features),
            "checkpoint": str(args.checkpoint) if args.mode == "scout" else None,
            "top_k": evaluate_topk(data, scores, args.top_k_values),
        }

        out = args.out
        if out is None:
            name = "bitplane_stats" if args.mode == "scout" else args.mode
            out = ROOT / "data" / f"results_scout_recall_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== evaluate_scout_recall.py ===")
    print(f"PASS  out={out}")
    for top_k, row in results["top_k"].items():
        print(f"PASS  top_k={top_k} object_recall={row['object_recall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
