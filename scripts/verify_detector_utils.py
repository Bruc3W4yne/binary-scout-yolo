"""
Verify detector utility contracts without loading YOLO.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from detector import Detection, box_iou, match_recall, nms, offset_detection  # noqa: E402
from preprocess import Box, Tile  # noqa: E402
from run_yolo_tiles import summarize  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_iou_and_nms() -> None:
    a = Detection(0, 0, 10, 10, 0.9, 0)
    b = Detection(1, 1, 11, 11, 0.8, 0)
    c = Detection(20, 20, 30, 30, 0.7, 0)
    expect(0.68 < box_iou(a.xyxy(), b.xyxy()) < 0.69, "unexpected IoU")
    kept = nms([b, c, a], iou_threshold=0.5)
    expect([det.score for det in kept] == [0.9, 0.7], "NMS should keep best overlap plus distant box")


def verify_tile_offset_and_clip() -> None:
    tile = Tile(tile_id=3, row=0, col=1, x1=80, y1=40, x2=240, y2=200)
    det = Detection(10, 20, 200, 180, 0.5, 2)
    shifted = offset_detection(det, tile, img_size=220)
    expect(shifted is not None, "shifted detection should survive clipping")
    expect(shifted.xyxy() == [90, 60, 220.0, 220.0], "unexpected clipped offset")
    expect(shifted.source_tile_id == 3, "tile id should be preserved")


def verify_match_recall() -> None:
    boxes = [
        Box(0, 0, 10, 10, class_id=0, category_id=1),
        Box(50, 50, 80, 80, class_id=1, category_id=2),
    ]
    detections = [
        Detection(1, 1, 11, 11, 0.9, 9),
        Detection(45, 45, 70, 70, 0.8, 9),
    ]
    result = match_recall(detections, boxes, iou_threshold=0.5)
    expect(result["gt_boxes"] == 2, "GT count mismatch")
    expect(result["matched_gt"] == 1, "only one detection should match at IoU 0.5")
    expect(result["recall"] == 0.5, "unexpected recall")


def verify_yolo_timing_summary() -> None:
    phases = {
        "image_load_ms": 1.0,
        "resize_preprocess_ms": 2.0,
        "gt_parse_ms": 3.0,
        "scout_ms": 4.0,
        "yolo_ms": 5.0,
        "merge_nms_ms": 6.0,
        "match_eval_ms": 7.0,
        "pipeline_ms_excl_gt": 18.0,
        "wall_ms": 28.0,
    }
    rows = [
        {
            "selected_tiles": 8,
            "selected_area_fraction": 0.25,
            "detector_calls": 8,
            "detections": 2,
            "gt_boxes": 2,
            "matched_gt": 1,
            "latency_ms": 18.0,
            **phases,
        },
        {
            "selected_tiles": 8,
            "selected_area_fraction": 0.50,
            "detector_calls": 4,
            "detections": 4,
            "gt_boxes": 2,
            "matched_gt": 2,
            "latency_ms": 20.0,
            **{key: value + 2.0 for key, value in phases.items()},
        },
    ]
    summary = summarize(rows, Namespace(selector="scout", top_k=8))
    expect(summary["class_agnostic_recall"] == 0.75, "summary recall should aggregate counts")
    expect(summary["mean_detector_calls"] == 6.0, "detector call summary mismatch")
    expect(summary["latency_ms"]["mean"] == 19.0, "latency mean should use pipeline latency")
    expect(summary["phase_ms"]["pipeline_ms_excl_gt"]["mean"] == 19.0, "pipeline phase mean mismatch")
    expect(summary["phase_ms"]["gt_parse_ms"]["mean"] == 4.0, "GT parse phase mean mismatch")


def main() -> int:
    try:
        verify_iou_and_nms()
        print("PASS  iou_and_nms")
        verify_tile_offset_and_clip()
        print("PASS  tile_offset_and_clip")
        verify_match_recall()
        print("PASS  match_recall")
        verify_yolo_timing_summary()
        print("PASS  yolo_timing_summary")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== verify_detector_utils.py ===")
    print("All detector utility checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
