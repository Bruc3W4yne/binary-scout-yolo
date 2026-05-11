"""
Verify detector utility contracts without loading YOLO.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from detector import (  # noqa: E402
    Detection,
    box_iou,
    match_recall,
    nms,
    offset_detection,
    project_detection_from_original_crop,
    tile_to_original_crop,
)
from preprocess import Box, Tile  # noqa: E402
from run_yolo_tiles import predict_tiles, summarize  # noqa: E402


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


def verify_original_crop_mapping() -> None:
    tile = Tile(tile_id=9, row=1, col=2, x1=160, y1=80, x2=320, y2=240)
    crop = tile_to_original_crop(tile, orig_size=(1280, 720), img_size=640)
    expect(crop == (320, 90, 640, 270), f"unexpected original crop: {crop}")

    det = Detection(32, 18, 96, 54, 0.7, 4)
    projected = project_detection_from_original_crop(det, crop, orig_size=(1280, 720), img_size=640, source_tile_id=9)
    expect(projected is not None, "projected detection should survive clipping")
    expect(projected.xyxy() == [176.0, 96.0, 208.0, 128.0], f"bad projected box: {projected.xyxy()}")
    expect(projected.source_tile_id == 9, "projected tile id should be set")

    identity_crop = tile_to_original_crop(tile, orig_size=(640, 640), img_size=640)
    identity = project_detection_from_original_crop(det, identity_crop, (640, 640), 640, source_tile_id=9)
    offset = offset_detection(det, tile, img_size=640)
    expect(identity == offset, "640x640 original mapping should match normal tile offset")


class FakeBoxes:
    def __init__(self, xyxy: list[list[float]]):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.ones(len(xyxy), dtype=torch.float32)
        self.cls = torch.zeros(len(xyxy), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class FakeResult:
    def __init__(self, xyxy: list[list[float]]):
        self.boxes = FakeBoxes(xyxy)


class FakeModel:
    def __init__(self) -> None:
        self.crop_shapes: list[tuple[int, int, int]] = []

    def predict(self, crops, **kwargs):
        self.crop_shapes = [tuple(crop.shape) for crop in crops]
        return [FakeResult([[32, 18, 96, 54]]) for _ in crops]


def verify_original_crop_predict_path() -> None:
    record = {
        "tile": [160, 80, 320, 240],
        "tile_id": 9,
        "tile_row": 1,
        "tile_col": 2,
        "orig_size": [1280, 720],
    }
    args = Namespace(
        crop_source="original",
        img_size=640,
        detector_imgsz=640,
        conf=0.25,
        yolo_iou=0.7,
        batch=1,
        merge_iou=0.5,
    )
    model = FakeModel()
    detections, tiles, _, _ = predict_tiles(
        model,
        rgb=np.zeros((640, 640, 3), dtype=np.uint8),
        original_rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        selected_records=[record],
        args=args,
        device="cpu",
    )
    expect(model.crop_shapes == [(180, 320, 3)], f"bad original crop shape: {model.crop_shapes}")
    expect([tile.tile_id for tile in tiles] == [9], "selected tile should round-trip")
    expect(len(detections) == 1, "fake model should produce one mapped detection")
    expect(detections[0].xyxy() == [176.0, 96.0, 208.0, 128.0], f"bad mapped detection: {detections[0].xyxy()}")


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
    expect(result["matched_detections"] == 1, "matched detection count mismatch")
    expect(result["false_positives"] == 1, "false-positive count mismatch")
    expect(result["precision"] == 0.5, "unexpected precision")
    expect(result["recall"] == 0.5, "unexpected recall")


def verify_size_bucket_recall() -> None:
    boxes = [
        Box(0, 0, 16, 16, class_id=0, category_id=1),
        Box(100, 100, 160, 160, class_id=1, category_id=2),
        Box(300, 300, 430, 430, class_id=2, category_id=3),
    ]
    detections = [
        Detection(0, 0, 16, 16, 0.9, 0),
        Detection(300, 300, 430, 430, 0.8, 2),
    ]
    result = match_recall(detections, boxes, iou_threshold=0.5)
    expect(result["small_gt_boxes"] == 1, "small GT count mismatch")
    expect(result["medium_gt_boxes"] == 1, "medium GT count mismatch")
    expect(result["large_gt_boxes"] == 1, "large GT count mismatch")
    expect(result["small_object_recall"] == 1.0, "small recall mismatch")
    expect(result["medium_object_recall"] == 0.0, "medium recall mismatch")
    expect(result["large_object_recall"] == 1.0, "large recall mismatch")


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
            "matched_detections": 1,
            "false_positives": 1,
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
            "matched_detections": 2,
            "false_positives": 2,
            "gt_boxes": 2,
            "matched_gt": 2,
            "latency_ms": 20.0,
            **{key: value + 2.0 for key, value in phases.items()},
        },
    ]
    summary = summarize(rows, Namespace(selector="scout", crop_source="resized", top_k=8))
    expect(summary["class_agnostic_recall"] == 0.75, "summary recall should aggregate counts")
    expect(summary["class_agnostic_precision"] == 0.5, "summary precision should aggregate counts")
    expect(summary["false_positives"] == 3, "summary false positives mismatch")
    expect(summary["mean_detector_calls"] == 6.0, "detector call summary mismatch")
    expect(summary["latency_ms"]["mean"] == 19.0, "latency mean should use pipeline latency")
    expect(summary["phase_ms"]["pipeline_ms_excl_gt"]["mean"] == 19.0, "pipeline phase mean mismatch")
    expect(summary["phase_ms"]["gt_parse_ms"]["mean"] == 4.0, "GT parse phase mean mismatch")
    expect("small_object_recall" in summary, "bucket recall should be summarized")


def main() -> int:
    try:
        verify_iou_and_nms()
        print("PASS  iou_and_nms")
        verify_tile_offset_and_clip()
        print("PASS  tile_offset_and_clip")
        verify_original_crop_mapping()
        print("PASS  original_crop_mapping")
        verify_original_crop_predict_path()
        print("PASS  original_crop_predict_path")
        verify_match_recall()
        print("PASS  match_recall")
        verify_size_bucket_recall()
        print("PASS  size_bucket_recall")
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
