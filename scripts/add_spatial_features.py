"""
Append cheap tile-position features to a bitplane-stats feature cache.
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
from scout import BINARY_STATS_SPATIAL_DIM, spatial_tile_features  # noqa: E402


def default_out(path: Path) -> Path:
    return path.with_name(path.name.replace("bitplane_stats_", "bitplane_stats_spatial_"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out or default_out(args.features)

    try:
        data = np.load(args.features)
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("feature_mode") != "bitplane-stats":
            raise ValueError(f"expected bitplane-stats features, got {metadata.get('feature_mode')}")

        tiles = make_tiles(args.img_size, args.tile_size, args.stride)
        spatial_by_id = spatial_tile_features(tiles, img_size=args.img_size)
        spatial = spatial_by_id[data["tile_ids"].astype(np.int64)]
        arrays = {key: data[key] for key in data.files if key != "metadata_json"}
        arrays["features"] = np.hstack([data["features"].astype(np.float32), spatial]).astype(np.float32)
        metadata.update(
            {
                "feature_mode": "bitplane-stats-spatial",
                "feature_dim": BINARY_STATS_SPATIAL_DIM,
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
