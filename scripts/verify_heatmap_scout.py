"""
Verify learned heatmap scout contracts without VisDrone or YOLO.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from heatmap_scout import (  # noqa: E402
    HEATMAP_SIZE,
    IMAGE_SIZE,
    BinaryConv2d,
    HeatmapScout,
    export_binary_metadata,
    heatmap_scout_loss,
    heatmap_target_from_boxes,
    load_checkpoint,
    load_live_checkpoint,
    live_heatmap_scores,
    oracle_rank_tile_targets,
    rgb_to_msb_planes,
    save_checkpoint,
    select_topk_tile_ids,
    tile_logits_from_heatmap,
    tile_targets_from_boxes,
)
from preprocess import Box, make_tiles  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_msb_planes() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    planes = rgb_to_msb_planes(rgb)
    expect(planes.shape == (4, 8, 8), f"bad plane shape: {planes.shape}")
    expect(set(np.unique(planes).tolist()) == {-1.0, 1.0}, "planes must be in {-1,+1}")


def verify_target_rasterization() -> None:
    tiny = Box(100, 120, 106, 126, class_id=0, category_id=1)
    target = heatmap_target_from_boxes([tiny])
    expect(target.shape == (1, HEATMAP_SIZE, HEATMAP_SIZE), f"bad target shape: {target.shape}")
    expect(float(target.max()) == 1.0, "center Gaussian should have peak 1.0")
    expect(int((target[0] >= 0.5).sum()) >= 9, "tiny objects should mark at least a 3x3 footprint")


def verify_tile_targets() -> None:
    tiles = make_tiles(IMAGE_SIZE)
    center_box = Box(10, 10, 20, 20, class_id=0, category_id=1)
    mostly_inside_box = Box(145, 20, 195, 70, class_id=0, category_id=1)
    targets = tile_targets_from_boxes(tiles, [center_box, mostly_inside_box])
    expect(targets[0] == 1.0, "tile containing object center should be positive")
    expect(targets[1] == 1.0, "tile covering >=60% of a box should be positive")

    far_box = Box(600, 600, 620, 620, class_id=0, category_id=1)
    rank_targets, rank_weights = oracle_rank_tile_targets(tiles, [center_box, far_box], max_rank=18)
    expect(rank_targets[0] == 1.0 and rank_targets[48] == 1.0, "oracle-rank targets should mark useful greedy tiles")
    expect(float(rank_targets.sum()) == 2.0, "oracle-rank targets should not mark filler tiles positive")
    expect(rank_weights[0] == 3.0 and rank_weights[48] == 3.0, "early oracle-rank positives should be weighted")


def verify_model_shape_and_backprop() -> None:
    for variant in ("float", "ste"):
        model = HeatmapScout(variant)
        with torch.no_grad():
            logits = model(torch.ones(1, 4, IMAGE_SIZE, IMAGE_SIZE))
        expect(tuple(logits.shape) == (1, 1, HEATMAP_SIZE, HEATMAP_SIZE), f"{variant} bad full shape: {tuple(logits.shape)}")

        small = torch.randn(1, 4, 64, 64)
        heatmap = torch.zeros(1, 1, 8, 8)
        tile_targets = torch.zeros(1, len(make_tiles(64, 16, 8)))
        loss, parts = heatmap_scout_loss(
            model(small),
            heatmap,
            tile_targets,
            make_tiles(64, 16, 8),
            torch.tensor([10.0]),
            image_size=64,
            tile_sample_weight=torch.ones_like(tile_targets),
        )
        loss.backward()
        expect(float(loss.item()) > 0.0 and "tile_bce" in parts, f"{variant} loss/backprop failed")


def verify_tile_scoring_and_topk() -> None:
    tiles = make_tiles(IMAGE_SIZE)
    logits = torch.zeros(1, 1, HEATMAP_SIZE, HEATMAP_SIZE)
    logits[:, :, :20, :20] = 3.0
    scores = tile_logits_from_heatmap(logits, tiles).squeeze(0)
    expect(int(scores.argmax().item()) == 0, "top-left heatmap score should select tile 0")

    tied = np.ones(len(tiles), dtype=np.float32)
    expect(select_topk_tile_ids(tied, tiles, 4) == [0, 1, 2, 3], "top-K ties must be tile-id deterministic")


def verify_checkpoint_and_export() -> None:
    model = HeatmapScout("ste")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heatmap.pt"
        save_checkpoint(model, path, {"dataset": "synthetic"}, ROOT)
        loaded, checkpoint = load_checkpoint(path)
        expect(loaded.variant == "ste", "checkpoint variant drifted")
        expect(checkpoint["input"] == "grayscale_msb4", "checkpoint input metadata missing")

    metadata = export_binary_metadata(model)
    expect(metadata["variant"] == "ste", "export variant missing")
    expect(len(metadata["binary_layers"]) == 3, "STE model should export three binary conv layers")
    for layer in metadata["binary_layers"]:
        expect(set(layer["values"]) <= {-1, 1}, f"non-binary export values: {layer}")


def verify_live_scores() -> None:
    records = []
    for tile in make_tiles(IMAGE_SIZE):
        records.append(
            {
                "stem": "synthetic",
                "tile_id": tile.tile_id,
                "tile_row": tile.row,
                "tile_col": tile.col,
                "tile": tile.xyxy(),
            }
        )
    rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heatmap.pt"
        save_checkpoint(HeatmapScout("float"), path, {"dataset": "synthetic"}, ROOT)
        checkpoint = load_live_checkpoint(path)
        scores, timing, route = live_heatmap_scores(rgb, records, checkpoint)

    expect(route == "learned-heatmap-float", f"bad route: {route}")
    expect(len(scores) == len(records), "live scoring should produce one score per tile")
    expect("scout_heatmap_model_ms" in timing, "live heatmap timing missing model phase")


def main() -> int:
    checks = [
        ("msb_planes", verify_msb_planes),
        ("target_rasterization", verify_target_rasterization),
        ("tile_targets", verify_tile_targets),
        ("model_shape_and_backprop", verify_model_shape_and_backprop),
        ("tile_scoring_and_topk", verify_tile_scoring_and_topk),
        ("checkpoint_and_export", verify_checkpoint_and_export),
        ("live_scores", verify_live_scores),
    ]
    print("=== verify_heatmap_scout.py ===")
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
