from argparse import Namespace
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tile_contracts() -> None:
    checks = load_script("verify_tile_contracts")

    checks.verify_grid()
    checks.verify_tile_labels()
    checks.verify_parser_and_scaling()
    checks.verify_bitplanes()
    checks.verify_bitplane_stats_fast_path()
    checks.verify_selected_area()
    checks.verify_split()


def test_detector_contracts() -> None:
    checks = load_script("verify_detector_utils")

    checks.verify_iou_and_nms()
    checks.verify_tile_offset_and_clip()
    checks.verify_original_crop_mapping()
    checks.verify_original_crop_predict_path()
    checks.verify_match_recall()
    checks.verify_size_bucket_recall()
    checks.verify_yolo_timing_summary()


def test_routing_contracts() -> None:
    checks = load_script("verify_routing")

    checks.verify_feature_grouping_and_boxes()
    checks.verify_random_prior_and_heuristic_scores()
    checks.verify_oracle_greedy()
    checks.verify_recall_evaluation()
    checks.verify_selected_area()
    checks.verify_learned_heatmap_selector_aliases()


def test_heatmap_scout_contracts() -> None:
    checks = load_script("verify_heatmap_scout")

    checks.verify_msb_planes()
    checks.verify_target_rasterization()
    checks.verify_tile_targets()
    checks.verify_model_shape_and_backprop()
    checks.verify_tile_scoring_and_topk()
    checks.verify_checkpoint_and_export()
    checks.verify_live_scores()


def test_tile_grid_contract() -> None:
    verify_grid = load_script("verify_tile_grid").verify_grid

    meta = verify_grid(
        Namespace(
            img_size=640,
            tile_size=160,
            stride=80,
            expect_tiles=49,
            require_full_coverage=True,
        )
    )
    assert meta["n_tiles"] == 49
