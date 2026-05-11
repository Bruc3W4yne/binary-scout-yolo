"""
verify_packed_kernel.py

Verify the main packed XNOR path used by BinaryConvLayer.

This script compares:

    BinaryConvLayer.forward()

against a slow, explicit NumPy reference implementation on tiny synthetic
inputs. It does not use VisDrone data and does not run YOLO.

Usage:
    python scripts/verify_packed_kernel.py
    python scripts/verify_packed_kernel.py --include-nonbinary

Before running, build the native C library:
    Windows MSYS2/MinGW:  mingw32-make
    Linux/WSL2:           make

The optional --include-nonbinary check verifies that input planes with values
0 and 255 behave like 0 and 1.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from binary_layer import BinaryConvLayer  # noqa: E402
import kernel_wrapper as kw  # noqa: E402
from preprocess import make_tiles  # noqa: E402
from run_yolo_tiles import build_scout_model, live_scout_scores, select_tile_records  # noqa: E402
from scout import SPATIAL_FEATURE_DIM, BinaryXnorExtractor  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    n_ch: int
    n_filters: int
    kH: int
    kW: int
    rows: int
    cols: int
    seed: int


CASES = [
    Case("single-channel-1x1", n_ch=1, n_filters=1, kH=1, kW=1, rows=4, cols=5, seed=10),
    Case("u8-sized-3x3", n_ch=8, n_filters=3, kH=3, kW=3, rows=5, cols=7, seed=20),
    Case("rgb-bitplanes-3x3", n_ch=24, n_filters=5, kH=3, kW=3, rows=6, cols=5, seed=30),
    Case("u64-full-3x3", n_ch=64, n_filters=2, kH=3, kW=3, rows=4, cols=6, seed=40),
]


def make_planes(rng: np.random.Generator, case: Case) -> np.ndarray:
    return rng.integers(0, 2, size=(case.n_ch, case.rows, case.cols), dtype=np.uint8)


def make_weights(rng: np.random.Generator, case: Case) -> np.ndarray:
    bits = rng.integers(
        0,
        2,
        size=(case.n_filters, case.n_ch, case.kH, case.kW),
        dtype=np.int8,
    )
    return np.where(bits == 0, np.int8(-1), np.int8(1))


def numpy_binary_conv_reference(planes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Slow reference for BinaryConvLayer.forward().

    Input planes are interpreted as binary activations:
        0       -> -1
        nonzero -> +1

    Weights are interpreted as:
        <= 0 -> -1
        > 0  -> +1

    Out-of-bounds padding uses input bit 0, which maps to -1. This matches the
    current packed C kernel semantics.
    """
    planes = np.asarray(planes)
    weights = np.asarray(weights)

    n_filters, n_ch, kH, kW = weights.shape
    if planes.ndim != 3:
        raise ValueError(f"planes must be 3D, got shape {planes.shape}")
    if planes.shape[0] != n_ch:
        raise ValueError(f"planes channels {planes.shape[0]} != weights channels {n_ch}")

    _, rows, cols = planes.shape
    padH = kH // 2
    padW = kW // 2
    out = np.empty((n_filters, rows, cols), dtype=np.int32)

    for f in range(n_filters):
        for r in range(rows):
            for c in range(cols):
                score = 0
                for ch in range(n_ch):
                    for kr in range(kH):
                        ir = r + kr - padH
                        for kc in range(kW):
                            ic = c + kc - padW
                            in_pos = False
                            if 0 <= ir < rows and 0 <= ic < cols:
                                in_pos = bool(planes[ch, ir, ic] != 0)
                            w_pos = bool(weights[f, ch, kr, kc] > 0)
                            score += 1 if in_pos == w_pos else -1
                out[f, r, c] = score

    return out


def run_forward_case(case: Case, nonbinary: bool = False) -> None:
    rng = np.random.default_rng(case.seed)
    planes = make_planes(rng, case)
    weights = make_weights(rng, case)

    if nonbinary:
        planes = planes * np.uint8(255)

    layer = BinaryConvLayer(
        n_filters=case.n_filters,
        n_ch=case.n_ch,
        kH=case.kH,
        kW=case.kW,
        seed=case.seed,
    )
    layer.set_weights(weights)

    c_out = layer.forward(planes)
    ref_out = numpy_binary_conv_reference(planes, weights)

    if not np.array_equal(c_out, ref_out):
        diff = c_out.astype(np.int64) - ref_out.astype(np.int64)
        mismatch = np.argwhere(diff != 0)[0]
        idx = tuple(int(v) for v in mismatch)
        raise AssertionError(
            f"{case.name}: mismatch at {idx}; "
            f"c={int(c_out[idx])} ref={int(ref_out[idx])} "
            f"max_abs_diff={int(np.abs(diff).max())}"
        )


def run_threshold_case() -> None:
    rng = np.random.default_rng(123)
    scores = rng.integers(-10, 11, size=(3, 4, 5), dtype=np.int32)
    layer = BinaryConvLayer(n_filters=3, n_ch=1, kH=1, kW=1, seed=123)

    actual = layer.apply_threshold(scores, threshold=0)
    expected = (scores > 0).astype(np.uint8)

    if not np.array_equal(actual, expected):
        raise AssertionError("threshold_i32_to_u8 does not match NumPy thresholding")


def run_live_binary_scout_case() -> None:
    rng = np.random.default_rng(777)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    tiles = make_tiles(img_size=64, tile_size=32, stride=32)
    records = [
        {
            "stem": "synthetic",
            "tile": tile.xyxy(),
            "tile_id": tile.tile_id,
            "tile_row": tile.row,
            "tile_col": tile.col,
        }
        for tile in tiles
    ]

    extractor = BinaryXnorExtractor(n_filters=6, kernel_size=3, threshold=0, seed=7)
    features, timing = extractor.features_with_timing(rgb, tiles)
    if features.shape != (len(tiles), 6):
        raise AssertionError(f"bad live binary feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise AssertionError("live binary features contain NaN or inf")
    for key in ("bitplanes_ms", "pack_ms", "xnor_kernel_ms", "threshold_ms", "tile_summary_ms"):
        if key not in timing or timing[key] < 0:
            raise AssertionError(f"bad timing key {key}: {timing}")

    feature_dim = features.shape[1] + SPATIAL_FEATURE_DIM
    model = build_scout_model(feature_dim, hidden_dim=0)
    with torch.no_grad():
        model.weight.fill_(0.01)
        model.bias.zero_()
    checkpoint = {
        "feature_mode": "binary-xnor-hybrid",
        "feature_dim": feature_dim,
        "hidden_dim": 0,
        "mean": torch.zeros((1, feature_dim), dtype=torch.float32),
        "std": torch.ones((1, feature_dim), dtype=torch.float32),
        "model": model.eval(),
        "feature_metadata": extractor.metadata(),
    }
    scores, route_timing, route = live_scout_scores(rgb, records, checkpoint)
    if route != "binary-xnor-hybrid":
        raise AssertionError(f"bad live scout route: {route}")
    if len(scores) != len(records):
        raise AssertionError("live scout did not score every tile")

    selected = select_tile_records(
        "synthetic",
        records,
        Namespace(selector="binary-xnor-live", top_k=2),
        random.Random(42),
        scores,
    )
    selected_ids = [int(row["tile_id"]) for row in selected]
    if len(selected_ids) != 2 or len(set(selected_ids)) != 2:
        raise AssertionError(f"top-K selection must return unique tiles, got {selected_ids}")
    if route_timing["scout_xnor_kernel_ms"] < 0 or route_timing["scout_mlp_ms"] < 0:
        raise AssertionError(f"bad route timing: {route_timing}")


def expect_raises(name: str, exc_type: type[Exception], fn, *args) -> None:
    try:
        fn(*args)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"{name}: expected {exc_type.__name__}, got {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"{name}: expected {exc_type.__name__}")


def run_shape_guard_case() -> None:
    layer = BinaryConvLayer(n_filters=2, n_ch=3, kH=3, kW=3, seed=99)

    expect_raises(
        "BinaryConvLayer rejects zero channels",
        ValueError,
        BinaryConvLayer,
        2, 0, 3, 3,
    )
    expect_raises(
        "pack_input rejects 2D arrays",
        ValueError,
        layer.pack_input,
        np.zeros((3, 4), dtype=np.uint8),
    )
    expect_raises(
        "forward rejects channel mismatch",
        ValueError,
        layer.forward,
        np.zeros((2, 4, 4), dtype=np.uint8),
    )
    expect_raises(
        "set_filter rejects out-of-range index",
        IndexError,
        layer.set_filter,
        2,
        np.ones((3, 3, 3), dtype=np.int8),
    )
    expect_raises(
        "apply_threshold rejects 2D scores",
        ValueError,
        layer.apply_threshold,
        np.zeros((2, 4), dtype=np.int32),
    )
    expect_raises(
        "wrapper rejects mismatched packed filter shape",
        ValueError,
        kw.xnor_multi_filter_conv,
        np.zeros((4, 4), dtype=np.uint64),
        np.zeros((2, 3, 2), dtype=np.uint64),
        3, 3, 3, 2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-nonbinary",
        action="store_true",
        help=(
            "Also check that 0/255 planes behave like 0/1 planes."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=== verify_packed_kernel.py ===")
    print("Checking BinaryConvLayer.forward() against a NumPy reference.")
    print()

    try:
        for case in CASES:
            run_forward_case(case)
            print(f"PASS  {case.name}")

        run_threshold_case()
        print("PASS  threshold_i32_to_u8")

        run_live_binary_scout_case()
        print("PASS  live_binary_xnor_scout")

        run_shape_guard_case()
        print("PASS  shape_guard_errors")

        if args.include_nonbinary:
            print()
            print("Checking optional nonbinary 0/255 plane handling...")
            run_forward_case(CASES[2], nonbinary=True)
            print("PASS  nonbinary 0/255 planes behave like 0/1 planes")

    except FileNotFoundError as exc:
        sys.stdout.flush()
        print(f"ERROR: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Build the native C library first:", file=sys.stderr)
        print("  Windows MSYS2/MinGW:  mingw32-make", file=sys.stderr)
        print("  Linux/WSL2:           make", file=sys.stderr)
        return 2
    except Exception as exc:
        sys.stdout.flush()
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print()
    print("All requested packed-kernel checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
