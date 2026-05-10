"""
Small detection utilities shared by YOLO routing scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from preprocess import Box, Tile


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    source_tile_id: int | None = None

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict:
        return asdict(self)


def clip_detection(det: Detection, img_size: int) -> Detection | None:
    x1 = min(max(det.x1, 0.0), float(img_size))
    y1 = min(max(det.y1, 0.0), float(img_size))
    x2 = min(max(det.x2, 0.0), float(img_size))
    y2 = min(max(det.y2, 0.0), float(img_size))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(x1, y1, x2, y2, det.score, det.class_id, det.source_tile_id)


def offset_detection(det: Detection, tile: Tile, img_size: int) -> Detection | None:
    shifted = Detection(
        x1=det.x1 + tile.x1,
        y1=det.y1 + tile.y1,
        x2=det.x2 + tile.x1,
        y2=det.y2 + tile.y1,
        score=det.score,
        class_id=det.class_id,
        source_tile_id=tile.tile_id,
    )
    return clip_detection(shifted, img_size)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(box_iou(det.xyxy(), prev.xyxy()) < iou_threshold for prev in kept):
            kept.append(det)
    return kept


def match_recall(detections: list[Detection], boxes: list[Box], iou_threshold: float = 0.5) -> dict:
    matched: set[int] = set()
    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, box in enumerate(boxes):
            if idx in matched:
                continue
            iou = box_iou(det.xyxy(), box.xyxy())
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched.add(best_idx)

    total = len(boxes)
    return {
        "gt_boxes": total,
        "matched_gt": len(matched),
        "recall": len(matched) / total if total else 0.0,
    }
