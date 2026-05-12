"""
Verify selector and routing contracts without VisDrone or YOLO.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_scout_recall import evaluate_topk, feature_artifact_name  # noqa: E402
from run_yolo_tiles import is_heatmap_selector, is_live_scout_selector, select_tile_records  # noqa: E402
from routing import (  # noqa: E402
    feature_groups,
    heuristic_scores,
    prior_by_tile,
    prior_scores,
    random_scores,
    select_oracle_greedy_indices,
    select_records_oracle_greedy,
    selected_area_for_tile_ids,
    tile_box_indices,
)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def feature_fixture() -> dict[str, np.ndarray]:
    box_offsets = np.asarray([0, 1, 3, 4, 4], dtype=np.uint32)
    box_indices = np.asarray([0, 0, 1, 2], dtype=np.uint16)
    return {
        "features": np.zeros((4, 38), dtype=np.float32),
        "labels": np.asarray([1, 1, 1, 0], dtype=np.uint8),
        "image_ids": np.zeros(4, dtype=np.uint32),
        "image_stems": np.asarray(["image_a"]),
        "tile_ids": np.asarray([0, 1, 2, 3], dtype=np.uint8),
        "n_objects": np.asarray([1, 2, 1, 0], dtype=np.uint16),
        "n_boxes": np.asarray([3, 3, 3, 3], dtype=np.uint16),
        "box_offsets": box_offsets,
        "box_indices": box_indices,
    }


def verify_feature_grouping_and_boxes() -> None:
    data = feature_fixture()
    expect(list(feature_groups(data).keys()) == ["image_a"], "feature groups should preserve stem order")
    expect(tile_box_indices(data, 1) == [0, 1], "box offset lookup failed")


def verify_random_prior_and_heuristic_scores() -> None:
    expect(np.array_equal(random_scores(5, 7), random_scores(5, 7)), "random scores must be seed-stable")

    prior = prior_by_tile(
        np.asarray([0, 1, 0, 1], dtype=np.uint8),
        np.asarray([0, 1, 1, 1], dtype=np.uint8),
    )
    scores = prior_scores(np.asarray([0, 1, 2], dtype=np.uint8), prior)
    expect(np.allclose(scores, [0.5, 1.0, 0.0]), f"bad prior scores: {scores}")

    features = np.zeros((2, 38), dtype=np.float32)
    features[0, :24] = 0.5
    expect(heuristic_scores(features)[0] > heuristic_scores(features)[1], "texture heuristic should prefer high bit entropy")


def verify_oracle_greedy() -> None:
    data = feature_fixture()
    indices = list(range(4))
    tile_ids = data["tile_ids"].astype(np.int32)
    box_lookup = {idx: tile_box_indices(data, idx) for idx in indices}
    selected = select_oracle_greedy_indices(indices, 2, tile_ids, box_lookup, data["n_objects"])
    expect(selected == [1, 2], f"greedy oracle should cover all boxes with tiles 1 and 2, got {selected}")

    records = [
        {"tile_id": 0, "n_objects": 1, "box_indices": [0]},
        {"tile_id": 1, "n_objects": 2, "box_indices": [0, 1]},
        {"tile_id": 2, "n_objects": 1, "box_indices": [2]},
        {"tile_id": 3, "n_objects": 0, "box_indices": []},
    ]
    record_selected = select_records_oracle_greedy(records, 2)
    expect([row["tile_id"] for row in record_selected] == [1, 2], "record greedy oracle mismatch")


def verify_recall_evaluation() -> None:
    data = feature_fixture()
    greedy = evaluate_topk(data, None, [1, 2], "oracle-greedy")
    expect(abs(greedy["1"]["object_recall"] - (2 / 3)) < 1e-9, f"bad K=1 recall: {greedy}")
    expect(greedy["2"]["object_recall"] == 1.0, f"bad K=2 recall: {greedy}")
    expect(greedy["2"]["selected_area_fraction"] > greedy["1"]["selected_area_fraction"], "area should grow with K")


def verify_selected_area() -> None:
    area = selected_area_for_tile_ids([0, 1], img_size=640)
    expect(0.09 < area < 0.10, f"unexpected selected area fraction: {area}")


def verify_feature_artifact_name() -> None:
    data = {"metadata_json": np.asarray('{"feature_mode": "binary-xnor"}')}
    expect(feature_artifact_name(data, "scout") == "binary_xnor", "feature artifact name should use metadata")
    expect(feature_artifact_name({}, "oracle-greedy") == "oracle_greedy", "fallback artifact name mismatch")


def verify_learned_heatmap_selector_aliases() -> None:
    expect(is_live_scout_selector("learned-heatmap"), "learned-heatmap should be a live scout selector")
    expect(is_live_scout_selector("learned-heatmap-live"), "learned-heatmap-live should be a live scout selector")
    expect(is_heatmap_selector("learned-heatmap-live"), "heatmap selector alias should be recognized")
    expect(is_live_scout_selector("xnor-heatmap-live"), "native XNOR heatmap should be a live scout selector")
    expect(is_heatmap_selector("xnor-heatmap-live"), "native XNOR heatmap should be a heatmap selector")
    expect(is_live_scout_selector("xnor-heatmap-320-live"), "lite native XNOR heatmap should be live")
    expect(is_heatmap_selector("xnor-heatmap-320-live"), "lite native XNOR heatmap should be heatmap")

    records = [
        {"stem": "image_a", "tile_id": 0},
        {"stem": "image_a", "tile_id": 1},
        {"stem": "image_a", "tile_id": 2},
    ]
    scores = {("image_a", 0): 0.1, ("image_a", 1): 0.9, ("image_a", 2): 0.8}
    selected = select_tile_records(
        "image_a",
        records,
        Namespace(selector="learned-heatmap-live", top_k=2),
        rng=None,
        scout_scores=scores,
    )
    expect([record["tile_id"] for record in selected] == [1, 2], "learned heatmap selector should use scout scores")


def main() -> int:
    checks = [
        ("feature_grouping_and_boxes", verify_feature_grouping_and_boxes),
        ("random_prior_and_heuristic_scores", verify_random_prior_and_heuristic_scores),
        ("oracle_greedy", verify_oracle_greedy),
        ("recall_evaluation", verify_recall_evaluation),
        ("selected_area", verify_selected_area),
        ("feature_artifact_name", verify_feature_artifact_name),
        ("learned_heatmap_selector_aliases", verify_learned_heatmap_selector_aliases),
    ]
    print("=== verify_routing.py ===")
    try:
        for name, fn in checks:
            fn()
            print(f"PASS  {name}")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
