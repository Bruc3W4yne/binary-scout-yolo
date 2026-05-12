from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import kernel_wrapper as kw
from binary_layer import BinaryConvLayer
from heatmap_scout import (
    BinaryConv2d,
    HeatmapScout,
    load_checkpoint,
    record_tile,
    rgb_to_msb_planes,
    tile_logits_from_heatmap,
)
from preprocess import Tile

DEFAULT_SCOUT_IMAGE_SIZE = 640
LITE_SCOUT_IMAGE_SIZE = 320


@dataclass
class _Block:
    conv: BinaryConvLayer
    bn_weight: np.ndarray
    bn_bias: np.ndarray
    bn_mean: np.ndarray
    bn_std: np.ndarray
    sign_threshold: np.ndarray
    sign_ge: np.ndarray
    sign_fixed: np.ndarray


def _signed_weights(module: BinaryConv2d) -> np.ndarray:
    weights = module.weight.detach().cpu().numpy()
    return np.where(weights >= 0.0, np.int8(1), np.int8(-1))


def _max_pool2x2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"expected even spatial dimensions for 2x2 pool, got {(h, w)}")
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _as_bits(x: np.ndarray) -> np.ndarray:
    return (x >= 0.0).astype(np.uint8, copy=False)


class NativeXnorHeatmapScout:
    def __init__(self, model: HeatmapScout):
        if model.variant != "ste":
            raise ValueError("native XNOR heatmap scout requires a STE checkpoint")
        default_threads = min(16, os.cpu_count() or 1)
        self.omp_threads = kw.set_omp_threads(int(os.environ.get("XNOR_SCOUT_THREADS", default_threads)))
        model = model.cpu().eval()
        self.blocks = [_build_block(block) for block in model.blocks]
        self.head_weight = model.head.weight.detach().cpu().numpy()[0, :, 0, 0].astype(np.float32)
        self.head_bias = float(model.head.bias.detach().cpu().numpy()[0]) if model.head.bias is not None else 0.0

    def logits(self, msb_planes: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        x = _as_bits(np.asarray(msb_planes, dtype=np.float32))
        timing = {"scout_pack_ms": 0.0, "scout_xnor_kernel_ms": 0.0, "scout_heatmap_postprocess_ms": 0.0}

        for idx, block in enumerate(self.blocks):
            started = time.perf_counter()
            packed = block.conv.pack_input(x)
            timing["scout_pack_ms"] += (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            scores = block.conv.forward_packed(packed, zero_padding=True)
            timing["scout_xnor_kernel_ms"] += (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            if idx < len(self.blocks) - 1:
                x = kw.bn_sign_pool2x2_i32_to_u8(
                    scores,
                    block.sign_threshold,
                    block.sign_ge,
                    block.sign_fixed,
                )
            else:
                scores = scores.astype(np.float32, copy=False)
                scores = (scores - block.bn_mean[:, None, None]) / block.bn_std[:, None, None]
                scores = scores * block.bn_weight[:, None, None] + block.bn_bias[:, None, None]
                x = _max_pool2x2(np.clip(scores, -1.0, 1.0))
            timing["scout_heatmap_postprocess_ms"] += (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        logits = np.tensordot(self.head_weight, x.astype(np.float32, copy=False), axes=(0, 0)) + self.head_bias
        timing["scout_heatmap_head_ms"] = (time.perf_counter() - started) * 1000.0
        return logits[None, None].astype(np.float32, copy=False), timing


def _build_block(block: torch.nn.Module) -> _Block:
    conv = block.conv
    bn = block.bn
    if not isinstance(conv, BinaryConv2d):
        raise ValueError("native XNOR heatmap scout can only export BinaryConv2d blocks")
    weights = _signed_weights(conv)
    layer = BinaryConvLayer(
        n_filters=weights.shape[0],
        n_ch=weights.shape[1],
        kH=weights.shape[2],
        kW=weights.shape[3],
    )
    layer.set_weights(weights)
    bn_weight = bn.weight.detach().cpu().numpy().astype(np.float32)
    bn_bias = bn.bias.detach().cpu().numpy().astype(np.float32)
    bn_mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    bn_std = np.sqrt(bn.running_var.detach().cpu().numpy().astype(np.float32) + float(bn.eps))
    scale = bn_weight / bn_std
    fixed = np.full(scale.shape, 255, dtype=np.uint8)
    stable = np.abs(scale) >= 1e-12
    fixed[~stable] = (bn_bias[~stable] >= 0.0).astype(np.uint8)
    thresholds = np.zeros_like(scale, dtype=np.float32)
    thresholds[stable] = bn_mean[stable] - bn_bias[stable] / scale[stable]

    return _Block(
        conv=layer,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
        bn_mean=bn_mean,
        bn_std=bn_std,
        sign_threshold=thresholds,
        sign_ge=(scale >= 0.0).astype(np.uint8),
        sign_fixed=fixed,
    )


def _resize_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    if rgb.shape[0] == image_size and rgb.shape[1] == image_size:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)


def load_xnor_live_checkpoint(path: Path, scout_image_size: int = DEFAULT_SCOUT_IMAGE_SIZE) -> dict:
    model, checkpoint = load_checkpoint(path, map_location="cpu")
    if scout_image_size < 64 or scout_image_size % 8:
        raise ValueError("scout_image_size must be >=64 and divisible by 8")
    checkpoint["model"] = NativeXnorHeatmapScout(model)
    checkpoint["scout_image_size"] = int(scout_image_size)
    checkpoint["route"] = "xnor-heatmap-live" if scout_image_size == DEFAULT_SCOUT_IMAGE_SIZE else f"xnor-heatmap-{scout_image_size}-live"
    return checkpoint


def live_xnor_heatmap_scores(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
) -> tuple[dict[tuple[str, int], float], dict[str, float], str]:
    scores, _logits, timing, route = live_xnor_heatmap_outputs(rgb, records, checkpoint)
    return scores, timing, route


def live_xnor_heatmap_outputs(
    rgb: np.ndarray,
    records: list[dict],
    checkpoint: dict,
) -> tuple[dict[tuple[str, int], float], np.ndarray, dict[str, float], str]:
    tiles = [Tile(**{k: int(v) for k, v in record_tile(record).items()}) for record in sorted(records, key=lambda item: int(item["tile_id"]))]
    timing: dict[str, float] = {}
    source_image_size = int(records[0].get("img_size", rgb.shape[0]))
    scout_image_size = int(checkpoint.get("scout_image_size", DEFAULT_SCOUT_IMAGE_SIZE))

    started = time.perf_counter()
    scout_rgb = _resize_rgb(rgb, scout_image_size)
    timing["scout_heatmap_resize_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    planes = rgb_to_msb_planes(scout_rgb)
    timing["scout_heatmap_preprocess_ms"] = (time.perf_counter() - started) * 1000.0

    logits, model_timing = checkpoint["model"].logits(planes)
    timing.update(model_timing)

    started = time.perf_counter()
    scores = tile_logits_from_heatmap(torch.from_numpy(logits), tiles, image_size=source_image_size).squeeze(0).numpy()
    timing["scout_heatmap_tile_score_ms"] = (time.perf_counter() - started) * 1000.0

    stem = records[0]["stem"]
    return (
        {(stem, tile.tile_id): float(score) for tile, score in zip(tiles, scores)},
        logits,
        timing,
        str(checkpoint["route"]),
    )
