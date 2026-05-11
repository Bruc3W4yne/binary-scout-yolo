"""
Small detection utilities shared by YOLO routing scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor

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


def tile_to_original_crop(tile: Tile, orig_size: tuple[int, int], img_size: int) -> tuple[int, int, int, int]:
    orig_w, orig_h = orig_size
    if orig_w < 1 or orig_h < 1:
        raise ValueError(f"original image size must be positive, got {orig_size}")
    if img_size < 1:
        raise ValueError(f"img_size must be positive, got {img_size}")

    sx = orig_w / img_size
    sy = orig_h / img_size
    x1 = max(0, min(orig_w, floor(tile.x1 * sx)))
    y1 = max(0, min(orig_h, floor(tile.y1 * sy)))
    x2 = max(0, min(orig_w, ceil(tile.x2 * sx)))
    y2 = max(0, min(orig_h, ceil(tile.y2 * sy)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"tile maps to an empty original crop: tile={tile.xyxy()} orig_size={orig_size}")
    return x1, y1, x2, y2


def project_detection_from_original_crop(
    det: Detection,
    crop_xyxy: tuple[int, int, int, int],
    orig_size: tuple[int, int],
    img_size: int,
    source_tile_id: int | None,
) -> Detection | None:
    orig_w, orig_h = orig_size
    if orig_w < 1 or orig_h < 1:
        raise ValueError(f"original image size must be positive, got {orig_size}")
    if img_size < 1:
        raise ValueError(f"img_size must be positive, got {img_size}")

    crop_x1, crop_y1, _, _ = crop_xyxy
    sx = img_size / orig_w
    sy = img_size / orig_h
    projected = Detection(
        x1=(det.x1 + crop_x1) * sx,
        y1=(det.y1 + crop_y1) * sy,
        x2=(det.x2 + crop_x1) * sx,
        y2=(det.y2 + crop_y1) * sy,
        score=det.score,
        class_id=det.class_id,
        source_tile_id=source_tile_id,
    )
    return clip_detection(projected, img_size)


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


def box_area(box: Box) -> float:
    return max(box.x2 - box.x1, 0.0) * max(box.y2 - box.y1, 0.0)


def match_recall(
    detections: list[Detection],
    boxes: list[Box],
    iou_threshold: float = 0.5,
    small_area: float = 32.0 * 32.0,
    medium_area: float = 96.0 * 96.0,
) -> dict:
    matched: set[int] = set()
    matched_detections = 0
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
            matched_detections += 1

    total = len(boxes)
    false_positives = max(len(detections) - matched_detections, 0)
    precision = matched_detections / len(detections) if detections else 0.0
    recall = len(matched) / total if total else 0.0
    result = {
        "gt_boxes": total,
        "matched_gt": len(matched),
        "matched_detections": matched_detections,
        "false_positives": false_positives,
        "precision": precision,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "recall": recall,
    }

    buckets = {
        "small": [idx for idx, box in enumerate(boxes) if box_area(box) < small_area],
        "medium": [idx for idx, box in enumerate(boxes) if small_area <= box_area(box) < medium_area],
        "large": [idx for idx, box in enumerate(boxes) if box_area(box) >= medium_area],
    }
    for name, indices in buckets.items():
        count = len(indices)
        matched_count = sum(1 for idx in indices if idx in matched)
        result[f"{name}_gt_boxes"] = count
        result[f"{name}_matched_gt"] = matched_count
        result[f"{name}_object_recall"] = matched_count / count if count else 0.0

    return result
