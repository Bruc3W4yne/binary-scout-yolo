"""
Run YOLO on full images or routed VisDrone tiles.

This is the detector/router smoke benchmark, not final mAP evaluation. It
reports class-agnostic recall against VisDrone boxes so the pipeline can be
tested before any VisDrone-specific YOLO fine-tuning exists.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import Detection, match_recall, nms, offset_detection  # noqa: E402
from preprocess import (  # noqa: E402
    Tile,
    parse_visdrone_annotations,
    scale_boxes_to_resized,
    selected_area_fraction,
)
from routing import (  # noqa: E402
    heuristic_scores,
    prior_from_records,
    select_records_by_scores,
    select_records_oracle_greedy,
)
from scout import SPATIAL_FEATURE_DIM, bitplane_stats_features, load_resized_rgb, spatial_tile_features  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL: {path}")
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


def grouped_by_stem(records: list[dict], max_images: int) -> OrderedDict[str, list[dict]]:
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for record in records:
        groups.setdefault(record["stem"], []).append(record)
    if max_images:
        groups = OrderedDict(list(groups.items())[:max_images])
    return groups


def yolo_device_arg(device: str):
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return 0 if device == "cuda" else "cpu"


def build_scout_model(feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if hidden_dim <= 0:
        return torch.nn.Linear(feature_dim, 1)
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def detections_from_result(result, source_tile_id: int | None) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
    return [
        Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            score=float(score),
            class_id=int(class_id),
            source_tile_id=source_tile_id,
        )
        for box, score, class_id in zip(xyxy, scores, classes)
    ]


def load_scout_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_scout_model(int(checkpoint["feature_dim"]), int(checkpoint.get("hidden_dim", 0)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["model"] = model
    return checkpoint


def score_features(features: np.ndarray, checkpoint: dict) -> np.ndarray:
    features_t = torch.from_numpy(features.astype(np.float32))
    mean = checkpoint["mean"].float()
    std = checkpoint["std"].float()
    with torch.no_grad():
        return checkpoint["model"]((features_t - mean) / std).squeeze(1).numpy()


def load_scout_scores(feature_path: Path, checkpoint_path: Path) -> dict[tuple[str, int], float]:
    data = np.load(feature_path)
    scores = score_features(data["features"], load_scout_checkpoint(checkpoint_path))

    if "stems" in data.files:
        stems = data["stems"].astype(str)
    else:
        stems = data["image_stems"].astype(str)[data["image_ids"].astype(np.int64)]
    tile_ids = data["tile_ids"].astype(np.int32)
    return {
        (str(stem), int(tile_id)): float(score)
        for stem, tile_id, score in zip(stems, tile_ids, scores)
    }


def live_bitplane_scores(rgb: np.ndarray, records: list[dict], checkpoint: dict) -> dict[tuple[str, int], float]:
    tiles = [tile_from_record(record) for record in sorted(records, key=lambda item: int(item["tile_id"]))]
    features = bitplane_stats_features(rgb, tiles)
    feature_dim = int(checkpoint["feature_dim"])
    if feature_dim == features.shape[1] + SPATIAL_FEATURE_DIM:
        features = np.hstack([features, spatial_tile_features(tiles, img_size=rgb.shape[0])]).astype(np.float32)
    elif feature_dim != features.shape[1]:
        raise ValueError(f"live scout cannot produce checkpoint feature_dim={feature_dim}")
    scores = score_features(features, checkpoint)
    stem = records[0]["stem"]
    return {
        (stem, tile.tile_id): float(score)
        for tile, score in zip(tiles, scores)
    }


def select_tile_records(
    stem: str,
    records: list[dict],
    args: argparse.Namespace,
    rng: random.Random,
    scout_scores: dict[tuple[str, int], float] | None,
    tile_scores: dict[int, float] | None = None,
) -> list[dict]:
    records = sorted(records, key=lambda item: int(item["tile_id"]))
    if args.selector == "all":
        return records
    if args.top_k < 1:
        raise ValueError("--top-k must be positive for selected-tile modes")
    if args.selector == "random":
        shuffled = records[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[: args.top_k], key=lambda item: int(item["tile_id"]))
    if args.selector in {"oracle", "oracle-count"}:
        ranked = sorted(
            records,
            key=lambda item: (-int(item["n_objects"]), -int(item["label"]), int(item["tile_id"])),
        )
        return sorted(ranked[: args.top_k], key=lambda item: int(item["tile_id"]))
    if args.selector == "oracle-greedy":
        return select_records_oracle_greedy(records, args.top_k)
    if args.selector == "prior":
        return select_records_by_scores(records, args.prior_scores, args.top_k)
    if args.selector == "heuristic":
        if tile_scores is None:
            raise ValueError("--selector heuristic requires live tile scores")
        return select_records_by_scores(records, tile_scores, args.top_k)
    if args.selector in {"scout", "scout-live"}:
        if scout_scores is None:
            raise ValueError(f"--selector {args.selector} requires scout scores")
        missing = [
            int(record["tile_id"])
            for record in records
            if (stem, int(record["tile_id"])) not in scout_scores
        ]
        if missing:
            raise ValueError(f"scout scores missing for {stem} tile_ids={missing[:5]}")
        ranked = sorted(
            records,
            key=lambda item: (-scout_scores[(stem, int(item["tile_id"]))], int(item["tile_id"])),
        )
        return sorted(ranked[: args.top_k], key=lambda item: int(item["tile_id"]))
    raise ValueError(f"unsupported selector: {args.selector}")


def predict_full(model, rgb: np.ndarray, args: argparse.Namespace, device) -> list[Detection]:
    result = model.predict(
        [rgb],
        imgsz=args.detector_imgsz,
        conf=args.conf,
        iou=args.yolo_iou,
        device=device,
        verbose=False,
        batch=1,
    )[0]
    return detections_from_result(result, source_tile_id=None)


def predict_tiles(
    model,
    rgb: np.ndarray,
    selected_records: list[dict],
    args: argparse.Namespace,
    device,
) -> tuple[list[Detection], list[Tile], float, float]:
    tiles = [tile_from_record(record) for record in selected_records]
    if not tiles:
        return [], [], 0.0, 0.0

    crops = [rgb[tile.y1 : tile.y2, tile.x1 : tile.x2, :] for tile in tiles]
    yolo_started = time.perf_counter()
    results = model.predict(
        crops,
        imgsz=args.detector_imgsz,
        conf=args.conf,
        iou=args.yolo_iou,
        device=device,
        verbose=False,
        batch=args.batch,
    )
    yolo_ms = (time.perf_counter() - yolo_started) * 1000.0

    merge_started = time.perf_counter()
    detections: list[Detection] = []
    for tile, result in zip(tiles, results):
        for det in detections_from_result(result, source_tile_id=tile.tile_id):
            shifted = offset_detection(det, tile, args.img_size)
            if shifted is not None:
                detections.append(shifted)
    merged = nms(detections, iou_threshold=args.merge_iou)
    merge_ms = (time.perf_counter() - merge_started) * 1000.0
    return merged, tiles, yolo_ms, merge_ms


def load_ground_truth(first_record: dict, img_size: int):
    ann_path = resolve_path(first_record["annotation_path"])
    boxes, _ = parse_visdrone_annotations(ann_path)
    return scale_boxes_to_resized(boxes, tuple(first_record["orig_size"]), img_size)


def run_one_image(model, stem: str, records: list[dict], args: argparse.Namespace, device, rng, scout_scores):
    started = time.perf_counter()
    first = records[0]
    rgb = load_resized_rgb(resolve_path(first["image_path"]), img_size=args.img_size)
    boxes = load_ground_truth(first, args.img_size)
    load_ms = (time.perf_counter() - started) * 1000.0
    scout_ms = 0.0
    merge_ms = 0.0

    if args.selector == "full":
        selected_records: list[dict] = []
        selected_tiles: list[Tile] = []
        yolo_started = time.perf_counter()
        detections = predict_full(model, rgb, args, device)
        yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        merge_started = time.perf_counter()
        detections = nms(detections, iou_threshold=args.merge_iou)
        merge_ms = (time.perf_counter() - merge_started) * 1000.0
        area_fraction = 1.0
    else:
        if args.selector == "scout-live":
            scout_started = time.perf_counter()
            scout_scores = live_bitplane_scores(rgb, records, args.scout_checkpoint)
            scout_ms = (time.perf_counter() - scout_started) * 1000.0
        tile_scores = None
        if args.selector == "heuristic":
            scout_started = time.perf_counter()
            tiles_for_scores = [
                tile_from_record(record)
                for record in sorted(records, key=lambda item: int(item["tile_id"]))
            ]
            scores = heuristic_scores(bitplane_stats_features(rgb, tiles_for_scores))
            tile_scores = {
                tile.tile_id: float(score)
                for tile, score in zip(tiles_for_scores, scores)
            }
            scout_ms = (time.perf_counter() - scout_started) * 1000.0
        selected_records = select_tile_records(stem, records, args, rng, scout_scores, tile_scores)
        detections, selected_tiles, yolo_ms, merge_ms = predict_tiles(model, rgb, selected_records, args, device)
        area_fraction = selected_area_fraction(selected_tiles, img_size=args.img_size)

    latency_ms = (time.perf_counter() - started) * 1000.0
    recall = match_recall(detections, boxes, iou_threshold=args.match_iou)
    row = {
        "stem": stem,
        "selector": args.selector,
        "top_k": None if args.selector in {"full", "all"} else args.top_k,
        "selected_tiles": len(selected_tiles),
        "selected_tile_ids": [tile.tile_id for tile in selected_tiles],
        "selected_area_fraction": area_fraction,
        "detections": len(detections),
        "latency_ms": latency_ms,
        "load_ms": load_ms,
        "scout_ms": scout_ms,
        "yolo_ms": yolo_ms,
        "merge_ms": merge_ms,
        **recall,
    }
    if args.save_detections:
        row["detection_boxes"] = [det.to_dict() for det in detections]
    return row


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    latencies = [float(row["latency_ms"]) for row in rows]
    gt_total = sum(int(row["gt_boxes"]) for row in rows)
    matched_total = sum(int(row["matched_gt"]) for row in rows)
    return {
        "selector": args.selector,
        "top_k": None if args.selector in {"full", "all"} else args.top_k,
        "images": len(rows),
        "gt_boxes": gt_total,
        "matched_gt": matched_total,
        "class_agnostic_recall": matched_total / gt_total if gt_total else 0.0,
        "mean_detections": float(np.mean([row["detections"] for row in rows])) if rows else 0.0,
        "mean_selected_tiles": float(np.mean([row["selected_tiles"] for row in rows])) if rows else 0.0,
        "mean_selected_area_fraction": (
            float(np.mean([row["selected_area_fraction"] for row in rows])) if rows else 0.0
        ),
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "phase_ms": {
            name: float(np.mean([row[name] for row in rows])) if rows else 0.0
            for name in ["load_ms", "scout_ms", "yolo_ms", "merge_ms"]
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, default=ROOT / "data" / "tile_dataset")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--selector",
        choices=[
            "full",
            "all",
            "random",
            "prior",
            "heuristic",
            "oracle",
            "oracle-count",
            "oracle-greedy",
            "scout",
            "scout-live",
        ],
        default="full",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=5, help="0 means all images")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prior-split", choices=["train", "val"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-detections", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO

        records = read_jsonl(args.tile_dir / f"{args.split}_tiles.jsonl")
        groups = grouped_by_stem(records, args.max_images)
        if not groups:
            raise ValueError("no images selected")

        scout_scores = None
        if args.selector == "scout":
            if args.features is None or args.checkpoint is None:
                raise ValueError("--selector scout requires --features and --checkpoint")
            scout_scores = load_scout_scores(args.features, args.checkpoint)
        if args.selector == "scout-live":
            if args.checkpoint is None:
                raise ValueError("--selector scout-live requires --checkpoint")
            args.scout_checkpoint = load_scout_checkpoint(args.checkpoint)
        if args.selector == "prior":
            args.prior_scores = prior_from_records(read_jsonl(args.tile_dir / f"{args.prior_split}_tiles.jsonl"))

        model = YOLO(args.weights)
        device = yolo_device_arg(args.device)
        rng = random.Random(args.seed)
        rows = [
            run_one_image(model, stem, image_records, args, device, rng, scout_scores)
            for stem, image_records in groups.items()
        ]
        result = {
            "config": {
                "weights": args.weights,
                "split": args.split,
                "selector": args.selector,
                "top_k": args.top_k,
                "max_images": args.max_images,
                "conf": args.conf,
                "yolo_iou": args.yolo_iou,
                "merge_iou": args.merge_iou,
                "match_iou": args.match_iou,
                "device": str(device),
                "prior_split": args.prior_split if args.selector == "prior" else None,
            },
            "summary": summarize(rows, args),
            "images": rows,
        }

        out = args.out
        if out is None:
            limit = f"_n{args.max_images}" if args.max_images else ""
            topk = f"_k{args.top_k}" if args.selector not in {"full", "all"} else ""
            out = ROOT / "data" / f"results_yolo_{args.selector}{topk}_{args.split}{limit}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== run_yolo_tiles.py ===")
    print(f"PASS  out={out}")
    print(f"PASS  selector={args.selector} images={len(rows)} device={device}")
    print(f"PASS  recall={result['summary']['class_agnostic_recall']:.3f}")
    print(f"PASS  latency_mean_ms={result['summary']['latency_ms']['mean']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
