"""
Tile routing helpers shared by recall evaluation and YOLO routing.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from typing import Iterable

import numpy as np

from preprocess import Tile, make_tiles, selected_area_fraction


def feature_stems(feature_data: dict[str, np.ndarray]) -> np.ndarray:
    if "stems" in feature_data:
        return feature_data["stems"].astype(str)
    return feature_data["image_stems"].astype(str)[feature_data["image_ids"].astype(np.int64)]


def feature_groups(feature_data: dict[str, np.ndarray], max_images: int = 0) -> OrderedDict[str, list[int]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for idx, stem in enumerate(feature_stems(feature_data)):
        groups.setdefault(str(stem), []).append(idx)
    return OrderedDict(list(groups.items())[:max_images]) if max_images else groups


def _parse_json_list(raw) -> list[int]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return [int(value) for value in json.loads(str(raw))]


def tile_box_indices(feature_data: dict[str, np.ndarray], idx: int) -> list[int]:
    if "box_indices_json" in feature_data:
        return _parse_json_list(feature_data["box_indices_json"][idx])

    offsets = feature_data["box_offsets"].astype(np.int64)
    flat = feature_data["box_indices"].astype(np.int64)
    return [int(value) for value in flat[offsets[idx] : offsets[idx + 1]]]


def image_box_count(feature_data: dict[str, np.ndarray], indices: Iterable[int]) -> int:
    indices = list(indices)
    if not indices:
        return 0
    if "n_boxes" in feature_data:
        return int(np.max(feature_data["n_boxes"].astype(np.int64)[indices]))
    covered = set()
    for idx in indices:
        covered.update(tile_box_indices(feature_data, idx))
    return len(covered)


def random_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n).astype(np.float32)


def oracle_count_scores(feature_data: dict[str, np.ndarray]) -> np.ndarray:
    return feature_data["n_objects"].astype(np.float32)


def prior_by_tile(tile_ids: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    counts: dict[int, int] = {}
    positives: dict[int, int] = {}
    for tile_id, label in zip(tile_ids.astype(np.int32), labels.astype(np.uint8)):
        tid = int(tile_id)
        counts[tid] = counts.get(tid, 0) + 1
        positives[tid] = positives.get(tid, 0) + int(label > 0)
    return {tile_id: positives[tile_id] / counts[tile_id] for tile_id in counts}


def prior_scores(tile_ids: np.ndarray, prior: dict[int, float]) -> np.ndarray:
    return np.asarray([prior.get(int(tile_id), 0.0) for tile_id in tile_ids], dtype=np.float32)


def heuristic_scores(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {features.shape}")
    if features.shape[1] >= 30:
        bitplanes = np.clip(features[:, :24], 0.0, 1.0)
        texture = np.mean(bitplanes * (1.0 - bitplanes), axis=1)
        rgb_std = np.mean(np.maximum(features[:, 27:30], 0.0), axis=1)
        return (texture + 0.5 * rgb_std).astype(np.float32)
    return np.mean(features, axis=1).astype(np.float32)


def ranked_indices(indices: Iterable[int], scores: np.ndarray, tile_ids: np.ndarray) -> list[int]:
    return sorted(indices, key=lambda idx: (-float(scores[idx]), int(tile_ids[idx])))


def select_topk_indices(indices: Iterable[int], scores: np.ndarray, tile_ids: np.ndarray, top_k: int) -> list[int]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = ranked_indices(indices, scores, tile_ids)
    return ranked[: min(top_k, len(ranked))]


def select_oracle_greedy_indices(
    indices: Iterable[int],
    top_k: int,
    tile_ids: np.ndarray,
    box_lookup: dict[int, list[int]],
    count_scores: np.ndarray | None = None,
) -> list[int]:
    return sorted(
        oracle_greedy_order_indices(indices, top_k, tile_ids, box_lookup, count_scores),
        key=lambda idx: int(tile_ids[idx]),
    )


def oracle_greedy_order_indices(
    indices: Iterable[int],
    top_k: int,
    tile_ids: np.ndarray,
    box_lookup: dict[int, list[int]],
    count_scores: np.ndarray | None = None,
) -> list[int]:
    if top_k < 1:
        raise ValueError("top_k must be positive")

    remaining = set(int(idx) for idx in indices)
    selected: list[int] = []
    covered: set[int] = set()

    while remaining and len(selected) < top_k:
        def key(idx: int) -> tuple[int, float, int]:
            boxes = set(box_lookup.get(idx, []))
            new = len(boxes - covered)
            count = float(count_scores[idx]) if count_scores is not None else float(len(boxes))
            return (-new, -count, int(tile_ids[idx]))

        best = min(remaining, key=key)
        selected.append(best)
        covered.update(box_lookup.get(best, []))
        remaining.remove(best)

    return selected


def record_box_indices(record: dict) -> list[int]:
    return [int(value) for value in record.get("box_indices", [])]


def record_tile_xyxy(record: dict) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = record["tile"]
    return float(x1), float(y1), float(x2), float(y2)


def tile_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = record_tile_xyxy(a)
    bx1, by1, bx2, by2 = record_tile_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def pairwise_overlap_stats(records: list[dict]) -> dict[str, float]:
    overlaps = [
        tile_iou(left, right)
        for idx, left in enumerate(records)
        for right in records[idx + 1 :]
    ]
    if not overlaps:
        return {
            "selected_pair_count": 0,
            "selected_iou_mean": 0.0,
            "selected_iou_max": 0.0,
            "selected_iou_over_0_1": 0,
            "selected_iou_over_0_3": 0,
        }
    return {
        "selected_pair_count": len(overlaps),
        "selected_iou_mean": float(np.mean(overlaps)),
        "selected_iou_max": float(np.max(overlaps)),
        "selected_iou_over_0_1": int(sum(value > 0.1 for value in overlaps)),
        "selected_iou_over_0_3": int(sum(value > 0.3 for value in overlaps)),
    }


def coverage_stats(records: list[dict], selected: list[dict]) -> dict[str, float]:
    all_boxes: set[int] = set()
    for record in records:
        all_boxes.update(record_box_indices(record))
    if not all_boxes and records:
        all_boxes = set(range(max(int(record.get("n_boxes", 0)) for record in records)))

    covered: set[int] = set()
    duplicate_coverage = 0
    selected_positive = 0
    for record in selected:
        boxes = set(record_box_indices(record))
        selected_positive += int(bool(boxes))
        duplicate_coverage += len(boxes & covered)
        covered.update(boxes)

    positive_total = sum(1 for record in records if record_box_indices(record))
    object_total = len(all_boxes)
    return {
        "gt_object_coverage": len(covered) / object_total if object_total else 0.0,
        "gt_objects_covered": len(covered),
        "gt_objects_total": object_total,
        "positive_tile_recall": selected_positive / positive_total if positive_total else 0.0,
        "positive_tiles_selected": selected_positive,
        "positive_tiles_total": positive_total,
        "duplicate_coverage_count": duplicate_coverage,
    }


def prior_from_records(records: Iterable[dict]) -> dict[int, float]:
    tile_ids = []
    labels = []
    for record in records:
        tile_ids.append(int(record["tile_id"]))
        labels.append(int(record["label"]))
    return prior_by_tile(np.asarray(tile_ids, dtype=np.int32), np.asarray(labels, dtype=np.uint8))


def _score(records_scores: dict[int, float], record: dict) -> float:
    return float(records_scores.get(int(record["tile_id"]), 0.0))


def _rank_records(records: list[dict], scores: dict[int, float]) -> list[dict]:
    return sorted(records, key=lambda item: (-_score(scores, item), int(item["tile_id"])))


def _normalized_scores(records: list[dict], scores: dict[int, float]) -> dict[int, float]:
    values = np.asarray([_score(scores, record) for record in records], dtype=np.float32)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    denom = max(hi - lo, 1e-6)
    return {int(record["tile_id"]): (_score(scores, record) - lo) / denom for record in records}


def _fill_to_limit(selected: list[dict], ranked: list[dict], limit: int) -> list[dict]:
    seen = {int(record["tile_id"]) for record in selected}
    for record in ranked:
        if len(selected) >= limit:
            break
        if int(record["tile_id"]) not in seen:
            selected.append(record)
            seen.add(int(record["tile_id"]))
    return selected


def select_records_by_scores(
    records: list[dict],
    scores: dict[int, float],
    top_k: int,
    policy: str = "topk",
    min_k: int = 1,
    max_k: int | None = None,
    tile_nms_iou: float = 0.3,
    mmr_lambda: float = 0.75,
    score_threshold: float | None = None,
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    records = sorted(records, key=lambda item: int(item["tile_id"]))
    ranked = _rank_records(records, scores)
    limit = min(max_k or top_k, len(ranked))
    min_k = min(max(int(min_k), 1), limit)

    if policy == "topk":
        return sorted(ranked[: min(top_k, len(ranked))], key=lambda item: int(item["tile_id"]))

    if policy == "tile-nms":
        selected: list[dict] = []
        for record in ranked:
            if len(selected) >= limit:
                break
            if all(tile_iou(record, kept) <= tile_nms_iou for kept in selected):
                selected.append(record)
        selected = _fill_to_limit(selected, ranked, limit)
        return sorted(selected, key=lambda item: int(item["tile_id"]))

    if policy not in {"mmr", "adaptive-mmr"}:
        raise ValueError(f"unsupported score selection policy: {policy}")

    norm = _normalized_scores(records, scores)
    remaining = ranked[:]
    selected = []
    while remaining and len(selected) < limit:
        def key(record: dict) -> tuple[float, int]:
            tile_id = int(record["tile_id"])
            overlap = max((tile_iou(record, kept) for kept in selected), default=0.0)
            value = mmr_lambda * norm[tile_id] - (1.0 - mmr_lambda) * overlap
            return value, -tile_id

        best = max(remaining, key=key)
        if policy == "adaptive-mmr" and len(selected) >= min_k and score_threshold is not None:
            if norm[int(best["tile_id"])] < score_threshold:
                break
        selected.append(best)
        remaining.remove(best)

    selected = _fill_to_limit(selected, ranked, min_k) if len(selected) < min_k else selected
    return sorted(selected, key=lambda item: int(item["tile_id"]))


def _tile_slice(record: dict, heatmap_shape: tuple[int, int]) -> tuple[slice, slice]:
    height, width = heatmap_shape
    x1, y1, x2, y2 = record_tile_xyxy(record)
    image_size = float(record.get("img_size", 640))
    sx = width / image_size
    sy = height / image_size
    left = max(0, min(width - 1, int(math.floor(x1 * sx))))
    top = max(0, min(height - 1, int(math.floor(y1 * sy))))
    right = max(left + 1, min(width, int(math.ceil(x2 * sx))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 * sy))))
    return slice(top, bottom), slice(left, right)


def _positive_heatmap_mass(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"heatmap must be 2D after squeeze, got {values.shape}")
    mass = np.maximum(values, 0.0)
    if float(mass.sum()) > 0.0:
        return mass
    shifted = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    shifted -= float(shifted.min())
    return shifted


def select_records_by_heatmap_coverage(
    records: list[dict],
    heatmap: np.ndarray,
    top_k: int,
    policy: str = "heatmap-coverage",
    min_k: int = 1,
    max_k: int | None = None,
    mass_threshold: float | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if policy not in {"heatmap-coverage", "adaptive-heatmap-coverage"}:
        raise ValueError(f"unsupported heatmap selection policy: {policy}")

    records = sorted(records, key=lambda item: int(item["tile_id"]))
    limit = min(max_k or top_k, len(records))
    min_k = min(max(int(min_k), 1), limit)
    mass = _positive_heatmap_mass(heatmap)
    total_mass = float(mass.sum())
    if total_mass <= 0.0:
        return records[:min_k if policy.startswith("adaptive") else limit]

    covered = np.zeros(mass.shape, dtype=bool)
    remaining = records[:]
    selected: list[dict] = []
    initial_best = None

    while remaining and len(selected) < limit:
        scored = []
        for record in remaining:
            ys, xs = _tile_slice(record, mass.shape)
            gain = float(mass[ys, xs][~covered[ys, xs]].sum())
            total = float(mass[ys, xs].sum())
            scored.append((gain, total, -int(record["tile_id"]), record, ys, xs))
        gain, total, _, record, ys, xs = max(scored, key=lambda item: (item[0], item[1], item[2]))
        if initial_best is None:
            initial_best = max(gain, 1e-6)
        if policy == "adaptive-heatmap-coverage" and len(selected) >= min_k:
            covered_mass = float(mass[covered].sum())
            if mass_threshold is not None and covered_mass / total_mass >= mass_threshold:
                break
            if score_threshold is not None and gain / initial_best < score_threshold:
                break
        selected.append(record)
        covered[ys, xs] = True
        remaining.remove(record)

    selected = _fill_to_limit(selected, records, min_k) if len(selected) < min_k else selected
    return sorted(selected, key=lambda item: int(item["tile_id"]))


def select_records_oracle_greedy(records: list[dict], top_k: int) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")

    by_idx = {idx: record for idx, record in enumerate(records)}
    tile_ids = np.asarray([int(record["tile_id"]) for record in records], dtype=np.int32)
    box_lookup = {idx: record_box_indices(record) for idx, record in by_idx.items()}
    counts = np.asarray([int(record["n_objects"]) for record in records], dtype=np.float32)
    selected = select_oracle_greedy_indices(by_idx.keys(), top_k, tile_ids, box_lookup, counts)
    return [by_idx[idx] for idx in selected]


def selected_area_for_tile_ids(tile_ids: Iterable[int], img_size: int = 640) -> float:
    tiles_by_id = {tile.tile_id: tile for tile in make_tiles(img_size=img_size)}
    tiles: list[Tile] = []
    for tile_id in tile_ids:
        try:
            tiles.append(tiles_by_id[int(tile_id)])
        except KeyError as exc:
            raise ValueError(f"tile_id={tile_id} is not in the default tile grid") from exc
    return selected_area_fraction(tiles, img_size=img_size)
