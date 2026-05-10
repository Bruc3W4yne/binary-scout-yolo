"""
Render a 7x7 tile-score heatmap from a scout feature cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_scout_recall import (  # noqa: E402
    feature_stems,
    load_feature_file,
    load_oracle_scores,
    load_random_scores,
    load_scout_scores,
)


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def colorize(grid: np.ndarray) -> np.ndarray:
    norm = normalize(grid)
    red = (norm * 255).astype(np.uint8)
    green = ((1.0 - np.abs(norm - 0.5) * 2.0) * 120).astype(np.uint8)
    blue = ((1.0 - norm) * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def draw_topk_borders(rgb: np.ndarray, top_positions: list[tuple[int, int]], cell: int) -> None:
    for row, col in top_positions:
        y1, y2 = row * cell, (row + 1) * cell
        x1, x2 = col * cell, (col + 1) * cell
        rgb[y1:y1 + 3, x1:x2] = 255
        rgb[y2 - 3:y2, x1:x2] = 255
        rgb[y1:y2, x1:x1 + 3] = 255
        rgb[y1:y2, x2 - 3:x2] = 255


def scores_for_mode(args: argparse.Namespace, data: dict[str, np.ndarray]) -> np.ndarray:
    if args.mode == "scout":
        return load_scout_scores(data, args.checkpoint)
    if args.mode == "oracle":
        return load_oracle_scores(data)
    return load_random_scores(len(data["labels"]), args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["scout", "random", "oracle"], default="scout")
    parser.add_argument("--stem", default=None, help="image stem to render; defaults to first image in feature cache")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=7)
    parser.add_argument("--cell", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "heatmaps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "scout" and args.checkpoint is None:
            raise ValueError("--mode scout requires --checkpoint")

        data = load_feature_file(args.features)
        stems = feature_stems(data)
        selected_stem = args.stem or str(stems[0])
        indices = np.flatnonzero(stems == selected_stem)
        if len(indices) == 0:
            raise ValueError(f"stem not found in feature cache: {selected_stem}")

        scores = scores_for_mode(args, data)
        tile_ids = data["tile_ids"][indices].astype(np.int32)
        image_scores = scores[indices].astype(np.float32)
        n_cols = args.grid_cols
        n_rows = int(np.ceil((int(tile_ids.max()) + 1) / n_cols))
        grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        for tile_id, score in zip(tile_ids, image_scores):
            grid[int(tile_id) // n_cols, int(tile_id) % n_cols] = float(score)
        if np.isnan(grid).any():
            raise ValueError("feature cache does not contain a complete tile grid for selected image")

        ranked = sorted(
            [(int(tile_id), float(score)) for tile_id, score in zip(tile_ids, image_scores)],
            key=lambda item: (-item[1], item[0]),
        )
        top = ranked[: min(args.top_k, len(ranked))]
        top_positions = [(tile_id // n_cols, tile_id % n_cols) for tile_id, _ in top]

        rgb = np.kron(colorize(grid), np.ones((args.cell, args.cell, 1), dtype=np.uint8))
        draw_topk_borders(rgb, top_positions, args.cell)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{selected_stem}_{args.mode}_k{args.top_k}"
        png_path = args.out_dir / f"{prefix}.png"
        json_path = args.out_dir / f"{prefix}.json"
        Image.fromarray(rgb).save(png_path)
        json_path.write_text(
            json.dumps(
                {
                    "stem": selected_stem,
                    "mode": args.mode,
                    "features": str(args.features),
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "top_k": args.top_k,
                    "top_tiles": [{"tile_id": tile_id, "score": score} for tile_id, score in top],
                    "score_grid": grid.tolist(),
                },
                indent=2,
            )
            + "\n"
        )

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=== render_scout_heatmap.py ===")
    print(f"PASS  png={png_path}")
    print(f"PASS  json={json_path}")
    print(f"PASS  top_tiles={[tile_id for tile_id, _ in top]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
