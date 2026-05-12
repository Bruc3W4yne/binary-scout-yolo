"""
Verify that the native XNOR heatmap scout matches the STE heatmap model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from binary_layer import BinaryConvLayer  # noqa: E402
from heatmap_scout import HeatmapScout, live_heatmap_scores, load_live_checkpoint, save_checkpoint  # noqa: E402
from preprocess import make_tiles  # noqa: E402
from xnor_heatmap_scout import NativeXnorHeatmapScout, live_xnor_heatmap_scores, load_xnor_live_checkpoint  # noqa: E402


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_zero_pad_conv_parity() -> None:
    torch.manual_seed(7)
    model = HeatmapScout("ste").eval()
    conv = model.blocks[0].conv
    x = torch.randn(1, 4, 17, 19)

    with torch.no_grad():
        expected = conv(x).squeeze(0).numpy().astype(np.int32)

    layer = BinaryConvLayer(n_filters=32, n_ch=4, kH=3, kW=3)
    weights = np.where(conv.weight.detach().numpy() >= 0.0, np.int8(1), np.int8(-1))
    layer.set_weights(weights)
    got = layer.forward((x.squeeze(0).numpy() >= 0.0).astype(np.uint8), zero_padding=True)
    expect(np.array_equal(got, expected), f"native zero-pad XNOR conv drifted, max_abs_diff={np.abs(got - expected).max()}")


def verify_full_route_parity() -> None:
    torch.manual_seed(11)
    np.random.seed(11)
    model = HeatmapScout("ste").eval()
    native = NativeXnorHeatmapScout(model)

    rgb = np.random.default_rng(11).integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    from heatmap_scout import rgb_to_msb_planes

    planes = rgb_to_msb_planes(rgb)
    with torch.no_grad():
        expected = model(torch.from_numpy(planes[None])).numpy()
    got, timing = native.logits(planes)

    expect(got.shape == expected.shape == (1, 1, 80, 80), f"bad logits shape: native={got.shape}, torch={expected.shape}")
    expect(float(np.max(np.abs(got - expected))) < 1e-4, "native heatmap logits do not match PyTorch STE")
    expect(timing["scout_xnor_kernel_ms"] >= 0.0 and timing["scout_pack_ms"] >= 0.0, "native timing missing")


def verify_live_scores_route() -> None:
    records = []
    for tile in make_tiles(640):
        records.append(
            {
                "stem": "synthetic",
                "tile_id": tile.tile_id,
                "tile_row": tile.row,
                "tile_col": tile.col,
                "tile": tile.xyxy(),
            }
        )
    rgb = np.random.default_rng(13).integers(0, 256, size=(640, 640, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "heatmap.pt"
        save_checkpoint(HeatmapScout("ste").eval(), path, {"dataset": "synthetic"}, ROOT)
        checkpoint = load_xnor_live_checkpoint(path)
        native_scores, timing, route = live_xnor_heatmap_scores(rgb, records, checkpoint)
        torch_checkpoint = load_live_checkpoint(path)
        torch_scores, _, _ = live_heatmap_scores(rgb, records, torch_checkpoint)

    expect(route == "xnor-heatmap-live", f"bad route: {route}")
    expect(len(native_scores) == len(records), "native route should score every tile")
    expect("scout_xnor_kernel_ms" in timing, "native XNOR timing missing")
    diffs = [abs(native_scores[key] - torch_scores[key]) for key in native_scores]
    expect(max(diffs) < 1e-4, f"native tile scores drifted, max_abs_diff={max(diffs)}")


def main() -> int:
    checks = [
        ("zero_pad_conv_parity", verify_zero_pad_conv_parity),
        ("full_route_parity", verify_full_route_parity),
        ("live_scores_route", verify_live_scores_route),
    ]
    print("=== verify_xnor_heatmap_scout.py ===")
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
