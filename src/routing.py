"""
Tile routing helpers shared by recall evaluation and YOLO routing.
"""

from __future__ import annotations

import json
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

    return sorted(selected, key=lambda idx: int(tile_ids[idx]))


def record_box_indices(record: dict) -> list[int]:
    return [int(value) for value in record.get("box_indices", [])]


def prior_from_records(records: Iterable[dict]) -> dict[int, float]:
    tile_ids = []
    labels = []
    for record in records:
        tile_ids.append(int(record["tile_id"]))
        labels.append(int(record["label"]))
    return prior_by_tile(np.asarray(tile_ids, dtype=np.int32), np.asarray(labels, dtype=np.uint8))


def select_records_by_scores(records: list[dict], scores: dict[int, float], top_k: int) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = sorted(records, key=lambda item: (-scores.get(int(item["tile_id"]), 0.0), int(item["tile_id"])))
    return sorted(ranked[: min(top_k, len(ranked))], key=lambda item: int(item["tile_id"]))


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
