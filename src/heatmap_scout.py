from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from preprocess import Box, Tile, make_tiles, parse_visdrone_annotations, scale_boxes_to_resized
from routing import oracle_greedy_order_indices

IMAGE_SIZE = 640
HEATMAP_SIZE = 80
MSB_BITS = (7, 6, 5, 4)
SCOUT_VARIANTS = {"float", "ste"}


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).to(grad.dtype)


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return _SignSTE.apply(x)


class BinaryConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(ste_sign(x), ste_sign(self.weight), padding=self.padding)


class HeatmapBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, variant: str):
        super().__init__()
        if variant not in SCOUT_VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        self.conv = (
            BinaryConv2d(in_channels, out_channels, 3)
            if variant == "ste"
            else torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.Hardtanh(-1.0, 1.0) if variant == "ste" else torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class HeatmapScout(torch.nn.Module):
    def __init__(self, variant: str = "float"):
        super().__init__()
        if variant not in SCOUT_VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")
        self.variant = variant
        self.blocks = torch.nn.Sequential(
            HeatmapBlock(4, 32, variant),
            HeatmapBlock(32, 64, variant),
            HeatmapBlock(64, 64, variant),
        )
        self.head = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


def load_resized_rgb(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def rgb_to_msb_planes(rgb: np.ndarray, bits: tuple[int, ...] = MSB_BITS) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")
    gray = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    ).astype(np.uint8)
    planes = [((gray >> bit) & 1).astype(np.float32) * 2.0 - 1.0 for bit in bits]
    return np.stack(planes, axis=0)


def load_scaled_boxes(annotation_path: Path, orig_size: tuple[int, int], image_size: int = IMAGE_SIZE) -> list[Box]:
    boxes, _ = parse_visdrone_annotations(annotation_path)
    return scale_boxes_to_resized(boxes, orig_size, image_size)


def _cell_bounds(lo: float, hi: float, heatmap_size: int) -> tuple[int, int]:
    start = max(0, min(heatmap_size - 1, int(math.floor(lo))))
    end = max(start + 1, min(heatmap_size, int(math.ceil(hi))))
    return start, end


def heatmap_target_from_boxes(
    boxes: list[Box],
    image_size: int = IMAGE_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
) -> np.ndarray:
    target = np.zeros((1, heatmap_size, heatmap_size), dtype=np.float32)
    scale = heatmap_size / image_size
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)

    for box in boxes:
        x1 = box.x1 * scale
        y1 = box.y1 * scale
        x2 = box.x2 * scale
        y2 = box.y2 * scale
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(x2 - x1, 1e-3)
        h = max(y2 - y1, 1e-3)

        fx1, fx2 = _cell_bounds(x1, x2, heatmap_size)
        fy1, fy2 = _cell_bounds(y1, y2, heatmap_size)
        if fx2 - fx1 < 3:
            center = int(round(cx))
            fx1 = max(0, min(heatmap_size - 3, center - 1))
            fx2 = min(heatmap_size, fx1 + 3)
        if fy2 - fy1 < 3:
            center = int(round(cy))
            fy1 = max(0, min(heatmap_size - 3, center - 1))
            fy2 = min(heatmap_size, fy1 + 3)
        target[0, fy1:fy2, fx1:fx2] = np.maximum(target[0, fy1:fy2, fx1:fx2], 0.5)

        sigma = max(0.75, 0.25 * math.sqrt(w * h))
        radius = int(math.ceil(3.0 * sigma))
        gx1 = max(0, int(math.floor(cx)) - radius)
        gx2 = min(heatmap_size, int(math.floor(cx)) + radius + 1)
        gy1 = max(0, int(math.floor(cy)) - radius)
        gy2 = min(heatmap_size, int(math.floor(cy)) + radius + 1)
        if gx1 < gx2 and gy1 < gy2:
            dist2 = (xx[gy1:gy2, gx1:gx2] + 0.5 - cx) ** 2 + (yy[gy1:gy2, gx1:gx2] + 0.5 - cy) ** 2
            gaussian = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
            gaussian /= max(float(gaussian.max()), 1e-6)
            target[0, gy1:gy2, gx1:gx2] = np.maximum(target[0, gy1:gy2, gx1:gx2], gaussian)

    return target


def _intersection_area(tile: Tile, box: Box) -> float:
    x1 = max(float(tile.x1), box.x1)
    y1 = max(float(tile.y1), box.y1)
    x2 = min(float(tile.x2), box.x2)
    y2 = min(float(tile.y2), box.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def tile_targets_from_boxes(tiles: list[Tile], boxes: list[Box], cover_threshold: float = 0.6) -> np.ndarray:
    targets = np.zeros(len(tiles), dtype=np.float32)
    for idx, tile in enumerate(tiles):
        for box in boxes:
            box_area = max(1e-6, (box.x2 - box.x1) * (box.y2 - box.y1))
            if tile.contains_center(box) or _intersection_area(tile, box) / box_area >= cover_threshold:
                targets[idx] = 1.0
                break
    return targets


def oracle_rank_tile_targets(
    tiles: list[Tile],
    boxes: list[Box],
    max_rank: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rank < 1:
        raise ValueError("max_rank must be positive")

    targets = np.zeros(len(tiles), dtype=np.float32)
    weights = np.ones(len(tiles), dtype=np.float32)
    if not tiles or not boxes:
        return targets, weights

    tile_ids = np.asarray([tile.tile_id for tile in tiles], dtype=np.int32)
    box_lookup = {
        idx: [box_idx for box_idx, box in enumerate(boxes) if tile.contains_center(box)]
        for idx, tile in enumerate(tiles)
    }
    counts = np.asarray([len(box_lookup[idx]) for idx in range(len(tiles))], dtype=np.float32)
    selected = oracle_greedy_order_indices(
        range(len(tiles)),
        min(max_rank, len(tiles)),
        tile_ids,
        box_lookup,
        counts,
    )

    covered: set[int] = set()
    for rank, idx in enumerate(selected, start=1):
        new_boxes = set(box_lookup.get(idx, [])) - covered
        if not new_boxes:
            continue
        targets[idx] = 1.0
        if rank <= 6:
            weights[idx] = 3.0
        elif rank <= 12:
            weights[idx] = 2.0
        else:
            weights[idx] = 1.0
        covered.update(new_boxes)

    return targets, weights


def tile_slices(
    tiles: list[Tile],
    image_size: int = IMAGE_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
) -> list[tuple[slice, slice]]:
    scale = heatmap_size / image_size
    out = []
    for tile in tiles:
        x1 = max(0, min(heatmap_size - 1, int(math.floor(tile.x1 * scale))))
        y1 = max(0, min(heatmap_size - 1, int(math.floor(tile.y1 * scale))))
        x2 = max(x1 + 1, min(heatmap_size, int(math.ceil(tile.x2 * scale))))
        y2 = max(y1 + 1, min(heatmap_size, int(math.ceil(tile.y2 * scale))))
        out.append((slice(y1, y2), slice(x1, x2)))
    return out


def tile_logits_from_heatmap(
    logits: torch.Tensor,
    tiles: list[Tile],
    image_size: int = IMAGE_SIZE,
    top_fraction: float = 0.05,
) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}")
    heatmap_size = int(logits.shape[-1])
    scores = []
    for ys, xs in tile_slices(tiles, image_size=image_size, heatmap_size=heatmap_size):
        values = logits[:, 0, ys, xs].flatten(1)
        k = max(1, int(math.ceil(values.shape[1] * top_fraction)))
        scores.append(values.topk(k, dim=1).values.mean(dim=1))
    return torch.stack(scores, dim=1)


def select_topk_tile_ids(tile_scores: torch.Tensor | np.ndarray, tiles: list[Tile], top_k: int) -> list[int]:
    scores = np.asarray(tile_scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(tiles):
        raise ValueError(f"scores length {len(scores)} does not match tiles {len(tiles)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = sorted(zip(tiles, scores), key=lambda item: (-float(item[1]), int(item[0].tile_id)))
    return [tile.tile_id for tile, _ in ranked[: min(top_k, len(ranked))]]


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def heatmap_scout_loss(
    logits: torch.Tensor,
    heatmap_target: torch.Tensor,
    tile_target: torch.Tensor,
    tiles: list[Tile],
    pos_weight: torch.Tensor,
    image_size: int = IMAGE_SIZE,
    tile_sample_weight: torch.Tensor | None = None,
    dense_weight: float = 1.0,
    dice_weight: float = 0.2,
    tile_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = logits.sum() * 0.0
    dense = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            heatmap_target,
            pos_weight=pos_weight.to(logits.device),
        )
        if dense_weight
        else zero
    )
    dice = soft_dice_loss(logits, heatmap_target) if dice_weight else zero
    tile_logits = tile_logits_from_heatmap(logits, tiles, image_size=image_size)
    tile_loss = torch.nn.functional.binary_cross_entropy_with_logits(tile_logits, tile_target, reduction="none")
    if tile_sample_weight is not None:
        weight = tile_sample_weight.to(tile_loss.device, dtype=tile_loss.dtype)
        tile = (tile_loss * weight).sum() / weight.sum().clamp_min(1.0)
    else:
        tile = tile_loss.mean()
    total = dense_weight * dense + dice_weight * dice + tile_loss_weight * tile
    return total, {"dense_bce": float(dense.item()), "dice": float(dice.item()), "tile_bce": float(tile.item())}


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def checkpoint_payload(
    model: HeatmapScout,
    metadata: dict,
    root: Path,
) -> dict:
    return {
        "model_state_dict": model.cpu().state_dict(),
        "variant": model.variant,
        "input": "grayscale_msb4",
        "msb_bits": MSB_BITS,
        "image_size": IMAGE_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "tile_config": [asdict(tile) for tile in make_tiles()],
        "metadata": {**metadata, "git_commit": git_commit(root)},
    }


def save_checkpoint(model: HeatmapScout, path: Path, metadata: dict, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata, root), path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[HeatmapScout, dict]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = HeatmapScout(str(checkpoint["variant"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def export_binary_metadata(model: HeatmapScout) -> dict:
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, BinaryConv2d):
            weight = module.weight.detach().cpu()
            signed = torch.where(weight >= 0, torch.ones_like(weight), -torch.ones_like(weight))
            layers.append(
                {
                    "name": name,
                    "shape": list(signed.shape),
                    "values": sorted(int(value) for value in signed.unique().tolist()),
                }
            )
    return {"variant": model.variant, "binary_layers": layers}


def record_tile(record: dict) -> dict:
    x1, y1, x2, y2 = record["tile"]
    return {
        "tile_id": int(record["tile_id"]),
        "row": int(record["tile_row"]),
        "col": int(record["tile_col"]),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def load_live_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    model, checkpoint = load_checkpoint(path, map_location=device)
    checkpoint["model"] = model.to(device).eval()
    checkpoint["route"] = f"learned-heatmap-{checkpoint['variant']}"
    return checkpoint


@torch.no_grad()
def live_heatmap_outputs(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
    device: torch.device | str = "cpu",
) -> tuple[dict[tuple[str, int], float], np.ndarray, dict[str, float], str]:
    tiles = [Tile(**{k: int(v) for k, v in record_tile(record).items()}) for record in sorted(records, key=lambda item: int(item["tile_id"]))]
    model = checkpoint["model"].to(device).eval()
    timing: dict[str, float] = {}

    started = time.perf_counter()
    planes = rgb_to_msb_planes(rgb)
    tensor = torch.from_numpy(planes[None]).to(device)
    timing["scout_heatmap_preprocess_ms"] = (time.perf_counter() - started) * 1000.0

    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model(tensor)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    timing["scout_heatmap_model_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    logits_cpu = logits.cpu()
    scores = tile_logits_from_heatmap(logits_cpu, tiles).squeeze(0).numpy()
    timing["scout_heatmap_tile_score_ms"] = (time.perf_counter() - started) * 1000.0

    stem = records[0]["stem"]
    return (
        {(stem, tile.tile_id): float(score) for tile, score in zip(tiles, scores)},
        logits_cpu.numpy(),
        timing,
        str(checkpoint["route"]),
    )


@torch.no_grad()
def live_heatmap_scores(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
    device: torch.device | str = "cpu",
) -> tuple[dict[tuple[str, int], float], dict[str, float], str]:
    scores, _logits, timing, route = live_heatmap_outputs(rgb, records, checkpoint, device)
    return scores, timing, route


def write_export(model: HeatmapScout, out_path: Path) -> None:
    metadata = export_binary_metadata(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2) + "\n")
