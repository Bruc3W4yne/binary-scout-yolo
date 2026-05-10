"""
Verify the project tile-grid contract without requiring VisDrone data.

Default contract:
  resized image: 640x640
  tile size:     160x160
  stride:        80
  candidates:    7x7 = 49 full-coverage tiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import make_tiles, tile_grid_metadata  # noqa: E402


def verify_grid(args: argparse.Namespace) -> dict:
    tiles = make_tiles(args.img_size, args.tile_size, args.stride)
    meta = tile_grid_metadata(args.img_size, args.tile_size, args.stride)

    if args.expect_tiles and len(tiles) != args.expect_tiles:
        raise AssertionError(f"expected {args.expect_tiles} tiles, got {len(tiles)}")
    if len({tile.tile_id for tile in tiles}) != len(tiles):
        raise AssertionError("duplicate tile IDs")

    for expected_id, tile in enumerate(tiles):
        if tile.tile_id != expected_id:
            raise AssertionError(f"tile ID order break: expected {expected_id}, got {tile.tile_id}")
        if not (0 <= tile.x1 < tile.x2 <= args.img_size):
            raise AssertionError(f"tile {tile.tile_id} has invalid x bounds: {tile.xyxy()}")
        if not (0 <= tile.y1 < tile.y2 <= args.img_size):
            raise AssertionError(f"tile {tile.tile_id} has invalid y bounds: {tile.xyxy()}")
        if tile.x2 - tile.x1 != args.tile_size or tile.y2 - tile.y1 != args.tile_size:
            raise AssertionError(f"tile {tile.tile_id} does not match tile_size: {tile.xyxy()}")

    if args.require_full_coverage:
        covered = np.zeros((args.img_size, args.img_size), dtype=np.bool_)
        for tile in tiles:
            covered[tile.y1:tile.y2, tile.x1:tile.x2] = True
        if not bool(covered.all()):
            missing = int((~covered).sum())
            raise AssertionError(f"grid does not fully cover image; missing pixels={missing}")

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--expect-tiles", type=int, default=49)
    parser.add_argument("--require-full-coverage", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    try:
        meta = verify_grid(parse_args())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== verify_tile_grid.py ===")
    print(f"PASS  n_tiles={meta['n_tiles']}")
    print(f"PASS  starts_x={meta['starts_x']}")
    print(f"PASS  starts_y={meta['starts_y']}")
    print("PASS  full coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
