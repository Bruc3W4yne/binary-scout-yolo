"""
Append cheap tile-position features to an extracted scout feature cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import make_tiles  # noqa: E402
from scout import SPATIAL_FEATURE_DIM, BinaryXnorExtractor, spatial_tile_features  # noqa: E402


def default_out(path: Path) -> Path:
    name = path.name
    if name.startswith("bitplane_stats_"):
        return path.with_name(name.replace("bitplane_stats_", "bitplane_stats_spatial_", 1))
    if name.startswith("binary_xnor_"):
        return path.with_name(name.replace("binary_xnor_", "binary_xnor_hybrid_", 1))
    return path.with_name(f"{path.stem}_spatial{path.suffix}")


def spatial_mode(feature_mode: str) -> str:
    if feature_mode == "binary-xnor":
        return "binary-xnor-hybrid"
    if feature_mode.endswith("-spatial") or feature_mode.endswith("-hybrid"):
        raise ValueError(f"{feature_mode} already includes spatial features")
    return f"{feature_mode}-spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--keep-first-features", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out or default_out(args.features)

    try:
        data = np.load(args.features)
        metadata = json.loads(str(data["metadata_json"].item()))
        feature_mode = str(metadata.get("feature_mode", ""))
        out_mode = spatial_mode(feature_mode)
        base_features = data["features"].astype(np.float32)
        if args.keep_first_features:
            if not 1 <= args.keep_first_features <= base_features.shape[1]:
                raise ValueError(f"--keep-first-features must be in [1, {base_features.shape[1]}]")
            base_features = base_features[:, : args.keep_first_features]
            metadata["kept_first_features"] = int(args.keep_first_features)
            if feature_mode == "binary-xnor":
                metadata.update(
                    BinaryXnorExtractor(
                        n_filters=args.keep_first_features,
                        kernel_size=int(metadata.get("binary_kernel_size", 3)),
                        threshold=int(metadata.get("binary_threshold", 0)),
                        seed=int(metadata.get("binary_seed", 42)),
                        input_channels=int(metadata.get("binary_input_channels", 24)),
                    ).metadata()
                )

        tiles = make_tiles(args.img_size, args.tile_size, args.stride)
        spatial_by_id = spatial_tile_features(tiles, img_size=args.img_size)
        spatial = spatial_by_id[data["tile_ids"].astype(np.int64)]
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        arrays["features"] = np.hstack([base_features, spatial]).astype(np.float32)
        metadata.update(
            {
                "feature_mode": out_mode,
                "feature_dim": int(base_features.shape[1] + SPATIAL_FEATURE_DIM),
                "spatial_features": [
                    "row_norm",
                    "col_norm",
                    "center_x",
                    "center_y",
                    "abs_center_x",
                    "abs_center_y",
                    "touches_x_border",
                    "touches_y_border",
                ],
            }
        )
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== add_spatial_features.py ===")
    print(f"PASS  out={out}")
    print(f"PASS  features={arrays['features'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
