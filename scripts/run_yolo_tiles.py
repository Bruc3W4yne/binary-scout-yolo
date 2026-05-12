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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import (  # noqa: E402
    Detection,
    match_recall,
    nms,
    offset_detection,
    project_detection_from_original_crop,
    tile_to_original_crop,
)
from heatmap_scout import (  # noqa: E402
    live_heatmap_scores,
    load_live_checkpoint as load_heatmap_checkpoint,
)
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
from scout import (  # noqa: E402
    BINARY_XNOR_DIM,
    SPATIAL_FEATURE_DIM,
    BinaryXnorExtractor,
    append_spatial_features,
    bitplane_stats_features,
    spatial_tile_features,
)


LIVE_SCOUT_PHASES = [
    "resize",
    "bitplane_stats",
    "bitplanes",
    "pack",
    "xnor_kernel",
    "threshold",
    "tile_summary",
    "spatial",
    "heatmap_preprocess",
    "heatmap_model",
    "heatmap_tile_score",
    "mlp",
    "topk",
    "total",
    "total_with_resize",
]


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


def git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def torch_device_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def device_name(device) -> str:
    if device != "cpu" and torch.cuda.is_available():
        index = int(device) if isinstance(device, int) else torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    return "cpu"


def is_live_scout_selector(selector: str) -> bool:
    return selector in {"scout-live", "binary-xnor-live", "learned-heatmap", "learned-heatmap-live"}


def is_heatmap_selector(selector: str) -> bool:
    return selector in {"learned-heatmap", "learned-heatmap-live"}


def sync_cuda(device) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_scout_model(feature_dim: int, hidden_dim: int) -> torch.nn.Module:
    if hidden_dim <= 0:
        return torch.nn.Linear(feature_dim, 1)
    return torch.nn.Sequential(
        torch.nn.Linear(feature_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def empty_live_scout_timing() -> dict[str, float]:
    return {f"scout_{phase}_ms": 0.0 for phase in LIVE_SCOUT_PHASES}


def timing_total(timing: dict[str, float]) -> float:
    return sum(
        float(timing.get(f"scout_{phase}_ms", 0.0))
        for phase in LIVE_SCOUT_PHASES
        if phase not in {"resize", "total", "total_with_resize"}
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


def binary_extractor_from_checkpoint(checkpoint: dict) -> BinaryXnorExtractor:
    metadata = checkpoint.get("feature_metadata") or {}
    feature_mode = str(checkpoint.get("feature_mode", "binary-xnor"))
    feature_dim = int(checkpoint["feature_dim"])
    has_spatial = feature_mode in {"binary-xnor-hybrid", "binary-xnor-spatial"}
    fallback_filters = feature_dim - SPATIAL_FEATURE_DIM if has_spatial else feature_dim
    fallback_filters = fallback_filters if fallback_filters > 0 else BINARY_XNOR_DIM
    return BinaryXnorExtractor(
        n_filters=int(metadata.get("binary_filters", fallback_filters)),
        kernel_size=int(metadata.get("binary_kernel_size", 3)),
        threshold=int(metadata.get("binary_threshold", 0)),
        seed=int(metadata.get("binary_seed", 42)),
        input_channels=int(metadata.get("binary_input_channels", 24)),
    )


def live_scout_scores(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
) -> tuple[dict[tuple[str, int], float], dict[str, float], str]:
    tiles = [tile_from_record(record) for record in sorted(records, key=lambda item: int(item["tile_id"]))]
    feature_mode = str(checkpoint.get("feature_mode", ""))
    timing = empty_live_scout_timing()
    feature_dim = int(checkpoint["feature_dim"])

    if feature_mode in {"bitplane-stats", "bitplane-stats-spatial"}:
        started = time.perf_counter()
        features = bitplane_stats_features(rgb, tiles)
        timing["scout_bitplane_stats_ms"] = (time.perf_counter() - started) * 1000.0
        if feature_dim == features.shape[1] + SPATIAL_FEATURE_DIM:
            started = time.perf_counter()
            features = append_spatial_features(features, tiles, img_size=rgb.shape[0])
            timing["scout_spatial_ms"] = (time.perf_counter() - started) * 1000.0
        elif feature_dim != features.shape[1]:
            raise ValueError(f"live scout cannot produce checkpoint feature_dim={feature_dim}")
    elif feature_mode in {"binary-xnor", "binary-xnor-hybrid", "binary-xnor-spatial"}:
        extractor = binary_extractor_from_checkpoint(checkpoint)
        features, raw_timing = extractor.features_with_timing(rgb, tiles)
        for name, value in raw_timing.items():
            timing[f"scout_{name}"] = value
        if feature_dim == features.shape[1] + SPATIAL_FEATURE_DIM:
            started = time.perf_counter()
            features = append_spatial_features(features, tiles, img_size=rgb.shape[0])
            timing["scout_spatial_ms"] = (time.perf_counter() - started) * 1000.0
        elif feature_dim != features.shape[1]:
            raise ValueError(f"live binary scout cannot produce checkpoint feature_dim={feature_dim}")
    else:
        raise ValueError(f"unsupported live scout feature_mode={feature_mode!r}")

    started = time.perf_counter()
    scores = score_features(features, checkpoint)
    timing["scout_mlp_ms"] = (time.perf_counter() - started) * 1000.0
    stem = records[0]["stem"]
    score_lookup = {(stem, tile.tile_id): float(score) for tile, score in zip(tiles, scores)}
    return score_lookup, timing, feature_mode


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
    if args.selector == "scout" or is_live_scout_selector(args.selector):
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
    sync_cuda(device)
    started = time.perf_counter()
    result = model.predict(
        [rgb],
        imgsz=args.detector_imgsz,
        conf=args.conf,
        iou=args.yolo_iou,
        device=device,
        verbose=False,
        batch=1,
    )[0]
    sync_cuda(device)
    yolo_ms = (time.perf_counter() - started) * 1000.0
    return detections_from_result(result, source_tile_id=None), yolo_ms


def predict_tiles(
    model,
    rgb: np.ndarray,
    original_rgb: np.ndarray | None,
    selected_records: list[dict],
    args: argparse.Namespace,
    device,
) -> tuple[list[Detection], list[Tile], float, float]:
    tiles = [tile_from_record(record) for record in selected_records]
    if not tiles:
        return [], [], 0.0, 0.0

    orig_size = tuple(selected_records[0]["orig_size"])
    if args.crop_source == "original":
        if original_rgb is None:
            raise ValueError("--crop-source original requires the original RGB image")
        crop_boxes = [tile_to_original_crop(tile, orig_size, args.img_size) for tile in tiles]
        crops = [original_rgb[y1:y2, x1:x2, :] for x1, y1, x2, y2 in crop_boxes]
    else:
        crop_boxes = [None for _ in tiles]
        crops = [rgb[tile.y1 : tile.y2, tile.x1 : tile.x2, :] for tile in tiles]

    sync_cuda(device)
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
    sync_cuda(device)
    yolo_ms = (time.perf_counter() - yolo_started) * 1000.0

    merge_started = time.perf_counter()
    detections: list[Detection] = []
    for tile, crop_box, result in zip(tiles, crop_boxes, results):
        for det in detections_from_result(result, source_tile_id=tile.tile_id):
            if crop_box is None:
                shifted = offset_detection(det, tile, args.img_size)
            else:
                shifted = project_detection_from_original_crop(
                    det,
                    crop_box,
                    orig_size,
                    args.img_size,
                    tile.tile_id,
                )
            if shifted is not None:
                detections.append(shifted)
    merged = nms(detections, iou_threshold=args.merge_iou)
    merge_ms = (time.perf_counter() - merge_started) * 1000.0
    return merged, tiles, yolo_ms, merge_ms


def load_ground_truth(first_record: dict, img_size: int):
    ann_path = resolve_path(first_record["annotation_path"])
    boxes, _ = parse_visdrone_annotations(ann_path)
    return scale_boxes_to_resized(boxes, tuple(first_record["orig_size"]), img_size)


def load_rgb_timed(path: Path, img_size: int, keep_original: bool) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    started = time.perf_counter()
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_rgb = np.asarray(image, dtype=np.uint8) if keep_original else None
        image_load_ms = (time.perf_counter() - started) * 1000.0

        resize_started = time.perf_counter()
        image = image.resize((img_size, img_size), Image.LANCZOS)
        rgb = np.asarray(image, dtype=np.uint8)
    resize_ms = (time.perf_counter() - resize_started) * 1000.0
    return rgb, original_rgb, image_load_ms, resize_ms


def run_one_image(model, stem: str, records: list[dict], args: argparse.Namespace, device, rng, scout_scores):
    wall_started = time.perf_counter()
    first = records[0]

    rgb, original_rgb, image_load_ms, resize_preprocess_ms = load_rgb_timed(
        resolve_path(first["image_path"]),
        img_size=args.img_size,
        keep_original=args.crop_source == "original" and args.selector != "full",
    )

    gt_started = time.perf_counter()
    boxes = load_ground_truth(first, args.img_size)
    gt_parse_ms = (time.perf_counter() - gt_started) * 1000.0
    scout_ms = 0.0
    merge_nms_ms = 0.0
    live_scout_timing = empty_live_scout_timing()
    scout_route = None

    if args.selector == "full":
        selected_records: list[dict] = []
        selected_tiles: list[Tile] = []
        detections, yolo_ms = predict_full(model, rgb, args, device)
        detector_calls = 1
        merge_started = time.perf_counter()
        detections = nms(detections, iou_threshold=args.merge_iou)
        merge_nms_ms = (time.perf_counter() - merge_started) * 1000.0
        area_fraction = 1.0
    else:
        if is_live_scout_selector(args.selector):
            if is_heatmap_selector(args.selector):
                scout_scores, live_scout_timing, scout_route = live_heatmap_scores(
                    rgb,
                    records,
                    args.scout_checkpoint,
                    device=args.scout_device,
                )
            else:
                scout_scores, live_scout_timing, scout_route = live_scout_scores(rgb, records, args.scout_checkpoint)
            if args.selector == "binary-xnor-live" and not str(scout_route).startswith("binary-xnor"):
                raise ValueError("--selector binary-xnor-live requires a binary-xnor checkpoint")
            if is_heatmap_selector(args.selector) and not str(scout_route).startswith("learned-heatmap"):
                raise ValueError(f"--selector {args.selector} requires a learned heatmap checkpoint")
            live_scout_timing["scout_resize_ms"] = resize_preprocess_ms
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
        select_started = time.perf_counter()
        selected_records = select_tile_records(stem, records, args, rng, scout_scores, tile_scores)
        if is_live_scout_selector(args.selector):
            live_scout_timing["scout_topk_ms"] = (time.perf_counter() - select_started) * 1000.0
            scout_ms = timing_total(live_scout_timing)
            live_scout_timing["scout_total_ms"] = scout_ms
            live_scout_timing["scout_total_with_resize_ms"] = scout_ms + resize_preprocess_ms
        detections, selected_tiles, yolo_ms, merge_nms_ms = predict_tiles(
            model,
            rgb,
            original_rgb,
            selected_records,
            args,
            device,
        )
        detector_calls = len(selected_tiles)
        area_fraction = selected_area_fraction(selected_tiles, img_size=args.img_size)

    match_started = time.perf_counter()
    recall = match_recall(
        detections,
        boxes,
        iou_threshold=args.match_iou,
        small_area=args.small_area,
        medium_area=args.medium_area,
    )
    match_eval_ms = (time.perf_counter() - match_started) * 1000.0
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    pipeline_ms = image_load_ms + resize_preprocess_ms + scout_ms + yolo_ms + merge_nms_ms
    row = {
        "image_id": stem,
        "stem": stem,
        "selector": args.selector,
        "scout_route": scout_route,
        "seed": args.seed,
        "crop_source": args.crop_source,
        "top_k": None if args.selector in {"full", "all"} else args.top_k,
        "selected_tiles": len(selected_tiles),
        "selected_tile_ids": [tile.tile_id for tile in selected_tiles],
        "selected_area_fraction": area_fraction,
        "detector_calls": detector_calls,
        "detections": len(detections),
        "latency_ms": pipeline_ms,
        "image_load_ms": image_load_ms,
        "resize_preprocess_ms": resize_preprocess_ms,
        "gt_parse_ms": gt_parse_ms,
        "scout_ms": scout_ms,
        "yolo_ms": yolo_ms,
        "merge_nms_ms": merge_nms_ms,
        "match_eval_ms": match_eval_ms,
        "pipeline_ms_excl_gt": pipeline_ms,
        "wall_ms": wall_ms,
        "load_ms": image_load_ms + resize_preprocess_ms + gt_parse_ms,
        "merge_ms": merge_nms_ms,
        **recall,
        **live_scout_timing,
    }
    if is_live_scout_selector(args.selector):
        row["scout_timing_ms"] = {
            key.replace("scout_", ""): value
            for key, value in live_scout_timing.items()
            if float(value) != 0.0 or key in {"scout_total_ms", "scout_total_with_resize_ms"}
        }
    if args.save_detections:
        row["detection_boxes"] = [det.to_dict() for det in detections]
    return row


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def summarize_ms(rows: list[dict], key: str) -> dict:
    values = [float(row.get(key, 0.0)) for row in rows]
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def bucket_summary(rows: list[dict], name: str) -> dict:
    gt = sum(int(row.get(f"{name}_gt_boxes", 0)) for row in rows)
    matched = sum(int(row.get(f"{name}_matched_gt", 0)) for row in rows)
    return {
        "gt_boxes": gt,
        "matched_gt": matched,
        "object_recall": matched / gt if gt else 0.0,
    }


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    gt_total = sum(int(row["gt_boxes"]) for row in rows)
    matched_total = sum(int(row["matched_gt"]) for row in rows)
    detection_total = sum(int(row["detections"]) for row in rows)
    matched_detection_total = sum(int(row.get("matched_detections", 0)) for row in rows)
    false_positive_total = sum(int(row.get("false_positives", 0)) for row in rows)
    precision = matched_detection_total / detection_total if detection_total else 0.0
    recall = matched_total / gt_total if gt_total else 0.0
    phase_names = [
        "image_load_ms",
        "resize_preprocess_ms",
        "gt_parse_ms",
        "scout_ms",
        *[f"scout_{phase}_ms" for phase in LIVE_SCOUT_PHASES],
        "yolo_ms",
        "merge_nms_ms",
        "match_eval_ms",
        "pipeline_ms_excl_gt",
        "wall_ms",
    ]
    return {
        "selector": args.selector,
        "crop_source": args.crop_source,
        "top_k": None if args.selector in {"full", "all"} else args.top_k,
        "images": len(rows),
        "gt_boxes": gt_total,
        "matched_gt": matched_total,
        "detections": detection_total,
        "matched_detections": matched_detection_total,
        "false_positives": false_positive_total,
        "class_agnostic_precision": precision,
        "class_agnostic_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "class_agnostic_recall": recall,
        **{
            f"{name}_{field}": value
            for name in ("small", "medium", "large")
            for field, value in bucket_summary(rows, name).items()
        },
        "mean_detections": float(np.mean([row["detections"] for row in rows])) if rows else 0.0,
        "mean_detector_calls": float(np.mean([row["detector_calls"] for row in rows])) if rows else 0.0,
        "mean_selected_tiles": float(np.mean([row["selected_tiles"] for row in rows])) if rows else 0.0,
        "mean_selected_area_fraction": (
            float(np.mean([row["selected_area_fraction"] for row in rows])) if rows else 0.0
        ),
        "latency_ms": summarize_ms(rows, "latency_ms"),
        "phase_ms": {name: summarize_ms(rows, name) for name in phase_names},
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
            "binary-xnor-live",
            "learned-heatmap",
            "learned-heatmap-live",
        ],
        default="full",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=5, help="0 means all images")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--crop-source",
        choices=["resized", "original"],
        default="resized",
        help="crop tiled detector inputs from the resized canvas or the original image",
    )
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--small-area", type=float, default=32.0 * 32.0)
    parser.add_argument("--medium-area", type=float, default=96.0 * 96.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prior-split", choices=["train", "val"], default="train")
    parser.add_argument("--warmup-images", type=int, default=1, help="YOLO warmup images to run before measured rows")
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
        if is_live_scout_selector(args.selector):
            if args.checkpoint is None:
                raise ValueError(f"--selector {args.selector} requires --checkpoint")
            args.scout_device = torch_device_arg(args.device)
            args.scout_checkpoint = (
                load_heatmap_checkpoint(args.checkpoint, args.scout_device)
                if is_heatmap_selector(args.selector)
                else load_scout_checkpoint(args.checkpoint)
            )
        if args.selector == "prior":
            args.prior_scores = prior_from_records(read_jsonl(args.tile_dir / f"{args.prior_split}_tiles.jsonl"))

        model = YOLO(args.weights)
        device = yolo_device_arg(args.device)
        if args.warmup_images < 0:
            raise ValueError("--warmup-images must be nonnegative")
        if args.warmup_images:
            warmup_rng = random.Random(args.seed)
            for stem, image_records in list(groups.items())[: args.warmup_images]:
                run_one_image(model, stem, image_records, args, device, warmup_rng, scout_scores)

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
                "crop_source": args.crop_source,
                "top_k": args.top_k,
                "max_images": args.max_images,
                "conf": args.conf,
                "yolo_iou": args.yolo_iou,
                "merge_iou": args.merge_iou,
                "match_iou": args.match_iou,
                "small_area": args.small_area,
                "medium_area": args.medium_area,
                "device": str(device),
                "device_name": device_name(device),
                "git_commit": git_commit(),
                "prior_split": args.prior_split if args.selector == "prior" else None,
                "warmup_images": args.warmup_images,
                "features": str(args.features) if args.features else None,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "scout_route": (
                    str(args.scout_checkpoint.get("route", args.scout_checkpoint.get("feature_mode")))
                    if is_live_scout_selector(args.selector)
                    else None
                ),
                "command": "python " + " ".join(sys.argv),
            },
            "summary": summarize(rows, args),
            "images": rows,
        }

        out = args.out
        if out is None:
            limit = f"_n{args.max_images}" if args.max_images else ""
            topk = f"_k{args.top_k}" if args.selector not in {"full", "all"} else ""
            crop = "_original" if args.crop_source == "original" else ""
            out = ROOT / "data" / f"results_yolo_{args.selector}{topk}{crop}_{args.split}{limit}.json"
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
