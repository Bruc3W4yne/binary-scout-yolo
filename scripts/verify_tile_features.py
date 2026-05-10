"""
Verify an extracted tile-feature NPZ file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--expect-feature-dim", type=int, default=30)
    parser.add_argument("--expect-feature-mode", default="bitplane-stats")
    return parser.parse_args()


def feature_stems(data) -> np.ndarray:
    if "stems" in data.files:
        return data["stems"].astype(str)
    image_ids = data["image_ids"].astype(np.int64)
    image_stems = data["image_stems"].astype(str)
    if image_ids.ndim != 1 or np.any(image_ids < 0) or np.any(image_ids >= len(image_stems)):
        raise AssertionError("image_ids must index image_stems")
    return image_stems[image_ids]


def main() -> int:
    args = parse_args()
    print("=== verify_tile_features.py ===")

    try:
        data = np.load(args.feature_file)
        features = data["features"]
        labels = data["labels"]
        stems = feature_stems(data)
        tile_ids = data["tile_ids"]
        metadata = json.loads(str(data["metadata_json"].item()))

        if features.ndim != 2 or features.shape[1] != args.expect_feature_dim:
            raise AssertionError(f"bad feature shape: {features.shape}")
        if labels.shape != (features.shape[0],):
            raise AssertionError(f"labels shape {labels.shape} does not match features")
        if stems.shape != labels.shape or tile_ids.shape != labels.shape:
            raise AssertionError("stems/tile_ids shapes must match labels")
        if "box_offsets" in data.files:
            offsets = data["box_offsets"]
            box_indices = data["box_indices"]
            if offsets.shape != (features.shape[0] + 1,):
                raise AssertionError("box_offsets must have one more entry than features")
            if offsets[0] != 0 or offsets[-1] != len(box_indices):
                raise AssertionError("box_offsets endpoints do not match box_indices")
            if np.any(offsets[1:] < offsets[:-1]):
                raise AssertionError("box_offsets must be nondecreasing")
        if not np.isfinite(features).all():
            raise AssertionError("features contain NaN or inf")
        if not np.isin(labels, [0, 1]).all():
            raise AssertionError("labels must be 0 or 1")
        if metadata.get("feature_mode") != args.expect_feature_mode:
            raise AssertionError(f"bad feature mode: {metadata}")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS  features={features.shape}")
    print(f"PASS  positives={int(labels.sum())}")
    print(f"PASS  images={len(np.unique(stems))}")
    print(f"PASS  metadata={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
