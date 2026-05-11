"""Benchmark packed XNOR/popcount kernels against float32 reference kernels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = (len(data) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def summarize(times_ms: list[float]) -> dict[str, float]:
    mean = float(np.mean(times_ms)) if times_ms else 0.0
    return {
        "mean_ms": round(mean, 4),
        "p50_ms": round(percentile(times_ms, 50), 4),
        "p95_ms": round(percentile(times_ms, 95), 4),
        "min_ms": round(float(np.min(times_ms)), 4) if times_ms else 0.0,
        "max_ms": round(float(np.max(times_ms)), 4) if times_ms else 0.0,
        "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


def timed(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return summarize(times)


def packed_weights(rng: np.random.Generator, channels: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    bits = rng.integers(0, 2, size=(channels, 3, 3), dtype=np.uint8)
    shifts = np.arange(channels, dtype=np.uint64).reshape(channels, 1, 1)
    packed = np.bitwise_or.reduce(bits.astype(np.uint64) << shifts, axis=0).astype(dtype)
    float_weights = np.where(bits == 1, 1.0, -1.0).astype(np.float32)
    return packed, float_weights


def make_inputs(rng: np.random.Generator, runs: int, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    u8_inputs = [
        rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        for _ in range(runs)
    ]
    u64_inputs = [
        rng.integers(0, np.iinfo(np.int64).max, size=(size, size), dtype=np.int64).view(np.uint64)
        for _ in range(runs)
    ]
    return u8_inputs, u64_inputs


def run_benchmark(args: argparse.Namespace) -> dict:
    import kernel_wrapper as kw

    rng = np.random.default_rng(args.seed)
    total_inputs = max(args.runs, 1)
    u8_inputs, u64_inputs = make_inputs(rng, total_inputs, args.size)
    w_u8, w_f32_8 = packed_weights(rng, 8, np.uint8)
    w_u64, w_f32_64 = packed_weights(rng, 64, np.uint64)

    idx = {"value": 0}

    def next_item(items: list[np.ndarray]) -> np.ndarray:
        value = items[idx["value"] % len(items)]
        idx["value"] += 1
        return value

    benchmarks = {
        "xnor_u8": lambda: kw.xnor_packed_u8_conv(next_item(u8_inputs), w_u8, 3, 3, 8),
        "float32_u8": lambda: kw.float32_conv_nch_u8(next_item(u8_inputs), w_f32_8, 3, 3, 8),
        "xnor_u64": lambda: kw.xnor_packed_u64_conv(next_item(u64_inputs), w_u64, 3, 3, 64),
        "float32_u64": lambda: kw.float32_conv_nch_u64(next_item(u64_inputs), w_f32_64, 3, 3, 64),
    }

    results = {
        "schema_version": 1,
        "mode": "synthetic",
        "seed": args.seed,
        "size": args.size,
        "runs": args.runs,
        "warmup": args.warmup,
        "benchmarks": {},
    }

    for name, fn in benchmarks.items():
        idx["value"] = 0
        results["benchmarks"][name] = timed(fn, args.warmup, args.runs)

    for channels, xnor_key, f32_key in [
        (8, "xnor_u8", "float32_u8"),
        (64, "xnor_u64", "float32_u64"),
    ]:
        xnor = results["benchmarks"][xnor_key]["mean_ms"]
        f32 = results["benchmarks"][f32_key]["mean_ms"]
        results[f"speedup_{channels}ch"] = round(f32 / xnor, 4) if xnor > 0 else None

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic inputs. This is the supported mode.")
    parser.add_argument("--size", type=int, default=640, help="Synthetic square image size.")
    parser.add_argument("--max-images", type=int, default=None, help="Accepted for command compatibility; maps to --runs when --runs is omitted.")
    parser.add_argument("--runs", type=int, default=None, help="Timed runs per variant.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per variant.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "smoke_xnor_kernel.json")
    args = parser.parse_args()

    if args.runs is None:
        args.runs = args.max_images if args.max_images is not None else 20
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.size < 8:
        parser.error("--size must be >= 8")
    if not args.synthetic:
        args.synthetic = True
    return args


def main() -> int:
    args = parse_args()
    try:
        results = run_benchmark(args)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Build the native library first: Windows MSYS2/MinGW `mingw32-make`; Linux/WSL2 `make`.", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
