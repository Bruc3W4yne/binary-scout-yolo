"""
Evaluate live learned-heatmap tile routing without running YOLO.

This uses the tile JSONL as ground truth for object-center coverage, but the
learned scorer only sees the image pixels at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from heatmap_scout import live_heatmap_outputs, load_live_checkpoint  # noqa: E402
from preprocess import selected_area_fraction  # noqa: E402
from routing import (  # noqa: E402
    heuristic_scores,
    prior_from_records,
    select_records_by_heatmap_coverage,
    select_records_by_scores,
    select_records_oracle_greedy,
)
from scout import bitplane_stats_features  # noqa: E402
from xnor_heatmap_scout import LITE_SCOUT_IMAGE_SIZE, live_xnor_heatmap_outputs, load_xnor_live_checkpoint  # noqa: E402

XNOR_HEATMAP_MODES = {"xnor-heatmap-live", "xnor-heatmap-320-live", "learned-xnor-heatmap-live"}
HEATMAP_POLICIES = {"heatmap-coverage", "adaptive-heatmap-coverage"}


def xnor_scout_image_size(mode: str) -> int:
    return LITE_SCOUT_IMAGE_SIZE if mode == "xnor-heatmap-320-live" else 640


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def grouped_by_stem(records: list[dict], max_images: int = 0) -> OrderedDict[str, list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(record["stem"], []).append(record)
    return OrderedDict(list(groups.items())[:max_images]) if max_images else groups


def tile_from_record(record: dict):
    from preprocess import Tile

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


def load_resized_rgb(record: dict) -> np.ndarray:
    with Image.open(resolve_path(record["image_path"])) as image:
        image = image.convert("RGB").resize((int(record.get("img_size", 640)),) * 2, Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def torch_device_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def record_box_indices(record: dict) -> list[int]:
    return [int(value) for value in record.get("box_indices", [])]


def image_box_count(records: list[dict]) -> int:
    if not records:
        return 0
    if "n_boxes" in records[0]:
        return int(records[0]["n_boxes"])
    covered = set()
    for record in records:
        covered.update(record_box_indices(record))
    return len(covered)


def summarize_ms(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return out


def learned_scores(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
    device: torch.device,
    xnor: bool = False,
) -> tuple[dict[int, float], np.ndarray, dict[str, float], str]:
    if xnor:
        scores, heatmap, timing, route = live_xnor_heatmap_outputs(rgb, records, checkpoint)
    else:
        scores, heatmap, timing, route = live_heatmap_outputs(rgb, records, checkpoint, device=device)
    stem = records[0]["stem"]
    by_tile = {tile_id: score for (score_stem, tile_id), score in scores.items() if score_stem == stem}
    timing["scout_total_ms"] = sum(float(value) for value in timing.values())
    return by_tile, heatmap, timing, route


def heuristic_tile_scores(rgb: np.ndarray, records: list[dict]) -> tuple[dict[int, float], dict[str, float], str]:
    tiles = [tile_from_record(record) for record in sorted(records, key=lambda item: int(item["tile_id"]))]
    started = time.perf_counter()
    features = bitplane_stats_features(rgb, tiles)
    scores = heuristic_scores(features)
    timing = {"scout_heuristic_ms": (time.perf_counter() - started) * 1000.0}
    return {tile.tile_id: float(score) for tile, score in zip(tiles, scores)}, timing, "heuristic"


def select_records(
    records: list[dict],
    mode: str,
    top_k: int,
    scores: dict[int, float] | None,
    rng: random.Random,
    args: argparse.Namespace,
    heatmap: np.ndarray | None = None,
) -> list[dict]:
    records = sorted(records, key=lambda item: int(item["tile_id"]))
    if mode == "random":
        shuffled = records[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:top_k], key=lambda item: int(item["tile_id"]))
    if mode == "oracle-greedy":
        return select_records_oracle_greedy(records, top_k)
    if mode == "oracle-count":
        ranked = sorted(records, key=lambda item: (-int(item["n_objects"]), -int(item["label"]), int(item["tile_id"])))
        return sorted(ranked[:top_k], key=lambda item: int(item["tile_id"]))
    if args.selection_policy in HEATMAP_POLICIES:
        if heatmap is None:
            raise ValueError(f"--selection-policy {args.selection_policy} requires a heatmap mode")
        return select_records_by_heatmap_coverage(
            records,
            heatmap,
            top_k,
            policy=args.selection_policy,
            min_k=args.min_k,
            max_k=args.max_k or None,
            mass_threshold=args.mass_threshold,
            score_threshold=args.score_threshold,
        )
    if scores is None:
        raise ValueError(f"mode={mode} requires scores")
    return select_records_by_scores(
        records,
        scores,
        top_k,
        policy=args.selection_policy,
        min_k=args.min_k,
        max_k=args.max_k or None,
        tile_nms_iou=args.tile_nms_iou,
        mmr_lambda=args.mmr_lambda,
        score_threshold=args.score_threshold,
    )


def evaluate_groups(
    groups: OrderedDict[str, list[dict]],
    args: argparse.Namespace,
    checkpoint: dict | None,
    prior_scores: dict[int, float] | None,
    seed: int,
) -> tuple[dict, list[dict[str, float]], str | None]:
    rng = random.Random(seed)
    metrics = {
        str(top_k): {
            "images": 0,
            "tile_total": 0,
            "selected_tile_total": 0,
            "object_total": 0,
            "covered_objects": 0,
            "positive_tile_total": 0,
            "selected_positive_tiles": 0,
            "area_total": 0.0,
        }
        for top_k in args.top_k_values
    }
    timing_rows: list[dict[str, float]] = []
    route = None

    for records in groups.values():
        if image_box_count(records) == 0:
            continue

        scores = None
        heatmap = None
        rgb = None
        if args.mode in {"learned-heatmap", "learned-heatmap-live", "heuristic", *XNOR_HEATMAP_MODES}:
            rgb = load_resized_rgb(records[0])
        if args.mode in {"learned-heatmap", "learned-heatmap-live", *XNOR_HEATMAP_MODES}:
            if checkpoint is None:
                raise ValueError("--mode learned-heatmap requires --checkpoint")
            scores, heatmap, timing, route = learned_scores(
                rgb,
                records,
                checkpoint,
                args.scout_device,
                xnor=args.mode in XNOR_HEATMAP_MODES,
            )
            timing_rows.append(timing)
        elif args.mode == "heuristic":
            scores, timing, route = heuristic_tile_scores(rgb, records)
            timing_rows.append(timing)
        elif args.mode == "prior":
            if prior_scores is None:
                raise ValueError("--mode prior requires --prior-tile-jsonl")
            scores = {int(record["tile_id"]): prior_scores.get(int(record["tile_id"]), 0.0) for record in records}

        object_n = image_box_count(records)
        positive_n = sum(1 for record in records if record_box_indices(record))

        for top_k in args.top_k_values:
            selected = select_records(records, args.mode, top_k, scores, rng, args, heatmap)
            covered = set()
            selected_positive = 0
            selected_tiles = []
            for record in selected:
                box_indices = record_box_indices(record)
                selected_positive += int(bool(box_indices))
                covered.update(box_indices)
                selected_tiles.append(tile_from_record(record))

            row = metrics[str(top_k)]
            row["images"] += 1
            row["tile_total"] += len(records)
            row["selected_tile_total"] += len(selected)
            row["object_total"] += object_n
            row["covered_objects"] += len(covered)
            row["positive_tile_total"] += positive_n
            row["selected_positive_tiles"] += selected_positive
            row["area_total"] += selected_area_fraction(selected_tiles, img_size=int(records[0].get("img_size", 640)))

    for row in metrics.values():
        images = int(row["images"])
        objects = int(row["object_total"])
        positives = int(row["positive_tile_total"])
        row["mean_selected_tiles"] = row["selected_tile_total"] / images if images else 0.0
        row["selected_area_fraction"] = row.pop("area_total") / images if images else 0.0
        row["object_recall"] = row["covered_objects"] / objects if objects else 0.0
        row["positive_tile_recall"] = row["selected_positive_tiles"] / positives if positives else 0.0

    return metrics, timing_rows, route


def aggregate_trials(trials: list[dict]) -> dict:
    if len(trials) == 1:
        return trials[0]
    out = {}
    for top_k in trials[0]:
        rows = [trial[top_k] for trial in trials]
        out[top_k] = {}
        for key in rows[0]:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            if key in {"images", "tile_total", "object_total", "positive_tile_total"}:
                out[top_k][key] = int(values[0])
            else:
                out[top_k][key] = float(values.mean())
                out[top_k][f"{key}_std"] = float(values.std())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-jsonl", type=Path, default=ROOT / "data" / "tile_dataset" / "val_tiles.jsonl")
    parser.add_argument("--prior-tile-jsonl", type=Path, default=ROOT / "data" / "tile_dataset" / "train_tiles.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=[
            "learned-heatmap",
            "learned-heatmap-live",
            "xnor-heatmap-live",
            "xnor-heatmap-320-live",
            "learned-xnor-heatmap-live",
            "heuristic",
            "random",
            "prior",
            "oracle-count",
            "oracle-greedy",
        ],
        default="learned-heatmap",
    )
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[8, 12])
    parser.add_argument(
        "--selection-policy",
        choices=[
            "topk",
            "tile-nms",
            "mmr",
            "heatmap-coverage",
            "adaptive-mmr",
            "adaptive-heatmap-coverage",
        ],
        default="topk",
    )
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=0, help="0 means use each top-k value")
    parser.add_argument("--tile-nms-iou", type=float, default=0.3)
    parser.add_argument("--mmr-lambda", type=float, default=0.75)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mass-threshold", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", "--trials", dest="random_trials", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.random_trials < 1:
            raise ValueError("--random-trials must be positive")
        if any(top_k < 1 for top_k in args.top_k_values):
            raise ValueError("--top-k-values must be positive")
        if args.min_k < 1:
            raise ValueError("--min-k must be positive")
        if args.max_k < 0:
            raise ValueError("--max-k must be nonnegative")
        heatmap_modes = {"learned-heatmap", "learned-heatmap-live", *XNOR_HEATMAP_MODES}
        if args.selection_policy in HEATMAP_POLICIES and args.mode not in heatmap_modes:
            raise ValueError(f"--selection-policy {args.selection_policy} requires a heatmap mode")

        records = read_jsonl(args.tile_jsonl)
        groups = grouped_by_stem(records, args.max_images)
        if not groups:
            raise ValueError("no images selected")

        args.scout_device = torch_device_arg(args.device)
        checkpoint = None
        if args.mode in {"learned-heatmap", "learned-heatmap-live", *XNOR_HEATMAP_MODES}:
            if args.checkpoint is None:
                raise ValueError("--mode learned-heatmap requires --checkpoint")
            checkpoint = (
                load_xnor_live_checkpoint(args.checkpoint, xnor_scout_image_size(args.mode))
                if args.mode in XNOR_HEATMAP_MODES
                else load_live_checkpoint(args.checkpoint, args.scout_device)
            )

        priors = None
        if args.mode == "prior":
            priors = prior_from_records(read_jsonl(args.prior_tile_jsonl))

        trial_results = []
        all_timings = []
        route = None
        n_trials = args.random_trials if args.mode == "random" else 1
        for trial_idx in range(n_trials):
            metrics, timings, route = evaluate_groups(groups, args, checkpoint, priors, args.seed + trial_idx)
            trial_results.append(metrics)
            all_timings.extend(timings)

        results = {
            "mode": args.mode,
            "route": route,
            "tile_jsonl": str(args.tile_jsonl),
            "prior_tile_jsonl": str(args.prior_tile_jsonl) if args.mode == "prior" else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "device": str(args.scout_device),
            "seed": args.seed,
            "random_trials": n_trials,
            "max_images": args.max_images,
            "selection_policy": args.selection_policy,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "tile_nms_iou": args.tile_nms_iou,
            "mmr_lambda": args.mmr_lambda,
            "score_threshold": args.score_threshold,
            "mass_threshold": args.mass_threshold,
            "top_k": aggregate_trials(trial_results),
            "timing_ms": summarize_ms(all_timings),
        }

        out = args.out
        if out is None:
            suffix = f"_n{args.max_images}" if args.max_images else ""
            policy = "" if args.selection_policy == "topk" else f"_{args.selection_policy}"
            out = ROOT / "data" / f"results_heatmap_scout_recall_{args.mode.replace('-', '_')}{policy}{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== evaluate_heatmap_scout_recall.py ===")
    print(f"PASS  out={out}")
    for top_k, row in results["top_k"].items():
        suffix = f" +/- {row['object_recall_std']:.3f}" if "object_recall_std" in row else ""
        print(f"PASS  top_k={top_k} object_recall={row['object_recall']:.3f}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
