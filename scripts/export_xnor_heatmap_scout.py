"""
Export a trained STE heatmap scout into a compact native-XNOR artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from heatmap_scout import BinaryConv2d, load_checkpoint, rgb_to_msb_planes  # noqa: E402
from xnor_heatmap_scout import NativeXnorHeatmapScout  # noqa: E402


def binary_weights(module: BinaryConv2d) -> np.ndarray:
    return np.where(module.weight.detach().cpu().numpy() >= 0.0, np.int8(1), np.int8(-1))


def export_payload(checkpoint_path: Path) -> dict[str, np.ndarray | str]:
    model, checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if model.variant != "ste":
        raise ValueError("only STE heatmap checkpoints can be exported to XNOR")

    native = NativeXnorHeatmapScout(model)
    rng = np.random.default_rng(123)
    fixture_rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    fixture_planes = rgb_to_msb_planes(fixture_rgb)
    fixture_logits, _ = native.logits(fixture_planes)

    payload: dict[str, np.ndarray | str] = {
        "metadata_json": json.dumps(
            {
                "route": "xnor-heatmap-live",
                "source_checkpoint": str(checkpoint_path),
                "source_variant": model.variant,
                "input": checkpoint.get("input", "grayscale_msb4"),
                "msb_bits": list(checkpoint.get("msb_bits", [7, 6, 5, 4])),
                "kernel": 3,
                "padding": "torch_zero",
                "pool": "max2x2",
                "note": "Binary conv body is exported for native packed XNOR-popcount inference.",
            },
            sort_keys=True,
        ),
        "head_weight": model.head.weight.detach().cpu().numpy().astype(np.float32),
        "head_bias": model.head.bias.detach().cpu().numpy().astype(np.float32),
        "fixture_msb_planes": fixture_planes.astype(np.float32),
        "fixture_logits": fixture_logits.astype(np.float32),
    }

    for idx, block in enumerate(model.blocks):
        conv = block.conv
        bn = block.bn
        if not isinstance(conv, BinaryConv2d):
            raise ValueError("STE model block did not contain BinaryConv2d")
        payload[f"block{idx}_weight"] = binary_weights(conv)
        payload[f"block{idx}_bn_weight"] = bn.weight.detach().cpu().numpy().astype(np.float32)
        payload[f"block{idx}_bn_bias"] = bn.bias.detach().cpu().numpy().astype(np.float32)
        payload[f"block{idx}_bn_mean"] = bn.running_mean.detach().cpu().numpy().astype(np.float32)
        payload[f"block{idx}_bn_var"] = bn.running_var.detach().cpu().numpy().astype(np.float32)
        payload[f"block{idx}_bn_eps"] = np.asarray([bn.eps], dtype=np.float32)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "scouts" / "xnor_heatmap_ste.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **export_payload(args.checkpoint))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("=== export_xnor_heatmap_scout.py ===")
    print(f"PASS  out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
