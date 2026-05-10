"""
Verify detector utility contracts without loading YOLO.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from detector import Detection, box_iou, match_recall, nms, offset_detection  # noqa: E402
from preprocess import Box, Tile  # noqa: E402


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


def main() -> int:
    try:
        verify_iou_and_nms()
        print("PASS  iou_and_nms")
        verify_tile_offset_and_clip()
        print("PASS  tile_offset_and_clip")
        verify_match_recall()
        print("PASS  match_recall")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== verify_detector_utils.py ===")
    print("All detector utility checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
