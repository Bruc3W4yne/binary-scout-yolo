"""
Evaluate top-K object-center recall from tile scout scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from routing import (  # noqa: E402
    feature_groups,
    feature_stems,
    heuristic_scores,
    image_box_count,
    oracle_count_scores,
    prior_by_tile,
    prior_scores,
    random_scores,
    select_oracle_greedy_indices,
    select_topk_indices,
    selected_area_for_tile_ids,
    tile_box_indices,
)


def load_feature_file(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing feature file: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def build_model(feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if hidden_dim <= 0:
        return torch.nn.Linear(feature_dim, 1)
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def load_scout_scores(feature_data: dict[str, np.ndarray], checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    features = torch.from_numpy(feature_data["features"].astype(np.float32))
    mean = checkpoint["mean"].float()
    std = checkpoint["std"].float()
    model = build_model(int(checkpoint["feature_dim"]), int(checkpoint.get("hidden_dim", 0)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model((features - mean) / std).squeeze(1)
    return logits.numpy()


def load_random_scores(n: int, seed: int) -> np.ndarray:
    return random_scores(n, seed)


def load_oracle_scores(feature_data: dict[str, np.ndarray]) -> np.ndarray:
    return oracle_count_scores(feature_data)


def feature_img_size(feature_data: dict[str, np.ndarray]) -> int:
    if "metadata_json" not in feature_data:
        return 640
    raw = feature_data["metadata_json"]
    text = raw.item() if hasattr(raw, "item") else raw
    try:
        return int(json.loads(str(text)).get("img_size", 640))
    except Exception:
        return 640


def feature_artifact_name(feature_data: dict[str, np.ndarray], fallback: str) -> str:
    if "metadata_json" in feature_data:
        raw = feature_data["metadata_json"]
        text = raw.item() if hasattr(raw, "item") else raw
        try:
            fallback = str(json.loads(str(text)).get("feature_mode", fallback))
        except Exception:
            pass
    return fallback.replace("-", "_").replace("/", "_")


def evaluate_topk(
    feature_data: dict[str, np.ndarray],
    scores: np.ndarray | None,
    top_k_values: list[int],
    selector: str,
    max_images: int = 0,
) -> dict:
    groups = feature_groups(feature_data, max_images=max_images)
    tile_ids = feature_data["tile_ids"].astype(np.int32)
    count_scores = oracle_count_scores(feature_data)
    img_size = feature_img_size(feature_data)

    results = {}
    for top_k in top_k_values:
        image_total = 0
        tile_total = 0
        selected_tile_total = 0
        covered_total = 0
        object_total = 0
        positive_tile_total = 0
        selected_positive_tiles = 0
        area_total = 0.0

        for indices in groups.values():
            image_n_boxes = image_box_count(feature_data, indices)
            if image_n_boxes == 0:
                continue

            if selector == "oracle-greedy":
                box_lookup = {idx: tile_box_indices(feature_data, idx) for idx in indices}
                selected = select_oracle_greedy_indices(indices, top_k, tile_ids, box_lookup, count_scores)
            else:
                if scores is None:
                    raise ValueError(f"selector={selector} requires scores")
                selected = select_topk_indices(indices, scores, tile_ids, top_k)

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

            image_total += 1
            tile_total += len(indices)
            selected_tile_total += len(selected)
            object_total += image_n_boxes
            covered_total += len(covered)
            positive_tile_total += positives
            selected_positive_tiles += selected_positives
            area_total += selected_area_for_tile_ids(tile_ids[selected], img_size=img_size)

        results[str(top_k)] = {
            "images": image_total,
            "tile_total": tile_total,
            "selected_tile_total": selected_tile_total,
            "mean_selected_tiles": selected_tile_total / image_total if image_total else 0.0,
            "selected_area_fraction": area_total / image_total if image_total else 0.0,
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


def scores_for_mode(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray | None, Path | None]:
    if args.mode == "scout":
        return load_scout_scores(data, args.checkpoint), None
    if args.mode == "random":
        return load_random_scores(len(data["labels"]), seed), None
    if args.mode in {"oracle", "oracle-count"}:
        return load_oracle_scores(data), None
    if args.mode == "heuristic":
        return heuristic_scores(data["features"]), None
    if args.mode == "prior":
        if args.prior_features is None:
            raise ValueError("--mode prior requires --prior-features from the train split")
        prior_data = load_feature_file(args.prior_features)
        prior = prior_by_tile(prior_data["tile_ids"], prior_data["labels"])
        return prior_scores(data["tile_ids"], prior), args.prior_features
    if args.mode == "oracle-greedy":
        return None, None
    raise ValueError(f"unsupported mode: {args.mode}")


def aggregate_trials(trials: list[dict]) -> dict:
    if len(trials) == 1:
        return trials[0]

    top_k_values = trials[0].keys()
    fixed_keys = {"images", "tile_total", "object_total", "positive_tile_total"}
    aggregated = {}
    for top_k in top_k_values:
        rows = [trial[top_k] for trial in trials]
        out = {"trials": len(rows)}
        for key in rows[0]:
            values = [float(row[key]) for row in rows]
            if key in fixed_keys:
                out[key] = int(values[0])
            else:
                out[key] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        aggregated[top_k] = out
    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "tile_features" / "bitplane_stats_val.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "scout" / "scout_bitplane_stats.pt")
    parser.add_argument("--mode", choices=["scout", "random", "prior", "heuristic", "oracle", "oracle-count", "oracle-greedy"], default="scout")
    parser.add_argument("--prior-features", type=Path, default=None)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", "--trials", dest="random_trials", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images in the feature cache")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_feature_file(args.features)
        n_trials = args.random_trials if args.mode == "random" else 1
        if n_trials < 1:
            raise ValueError("--random-trials must be positive")

        trial_results = []
        prior_features = None
        for trial_idx in range(n_trials):
            scores, prior_features = scores_for_mode(args, data, args.seed + trial_idx)
            trial_results.append(evaluate_topk(data, scores, args.top_k_values, args.mode, args.max_images))

        results = {
            "mode": args.mode,
            "features": str(args.features),
            "checkpoint": str(args.checkpoint) if args.mode == "scout" else None,
            "prior_features": str(prior_features) if prior_features else None,
            "seed": args.seed,
            "random_trials": n_trials,
            "max_images": args.max_images,
            "top_k": aggregate_trials(trial_results),
        }

        out = args.out
        if out is None:
            name = feature_artifact_name(data, "scout") if args.mode == "scout" else args.mode.replace("-", "_")
            out = ROOT / "data" / f"results_scout_recall_{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== evaluate_scout_recall.py ===")
    print(f"PASS  out={out}")
    for top_k, row in results["top_k"].items():
        suffix = f" +/- {row['object_recall_std']:.3f}" if "object_recall_std" in row else ""
        print(f"PASS  top_k={top_k} object_recall={row['object_recall']:.3f}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
