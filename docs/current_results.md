# Current Results

Generated on the Windows RTX 4090 machine during the first end-to-end implementation pass.

These are not final paper-quality numbers. They are pipeline validation numbers that show the project is runnable and where the next optimization work should focus.

For the exact commands verified in the final handoff pass, see `docs/completion_audit.md`.

## Artifacts Verified

| Artifact | Result |
|---|---:|
| VisDrone train images | 6471 |
| VisDrone val images | 548 |
| Tiles per image | 49 |
| Train tile records | 317079 |
| Val tile records | 26852 |
| Compact train feature cache | 45 MB |
| Compact val feature cache | 3.9 MB |

The compact cache replaced an earlier 1.65 GB cache by removing repeated strings and JSON blobs from every tile record.

Binary-XNOR smoke from clean exported commit `b19ad32`:

```text
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val --max-images 1 --out-dir data\tile_features_b19_smoke --progress-every 0
python scripts\verify_tile_features.py --feature-file data\tile_features_b19_smoke\binary_xnor_val_n1.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
```

The one-image binary cache had 49 rows, 64 features, and recorded binary metadata: 64 filters, 3x3 kernel, threshold 0, seed 42, 24 input channels, and weight hash prefix `deedea25a6e1d153`.

## Tile Coverage

Object-center recall on the full VisDrone val tile labels:

| Method | K=4 | K=8 | K=12 | K=16 | K=20 |
|---|---:|---:|---:|---:|---:|
| Random | 0.249 | 0.447 | 0.598 | 0.721 | 0.809 |
| Bitplane stats scout | 0.199 | 0.362 | 0.514 | 0.642 | 0.746 |
| Spatial bitplane linear scout | 0.320 | 0.492 | 0.601 | 0.695 | 0.773 |
| Spatial bitplane MLP scout | 0.364 | 0.516 | 0.635 | 0.735 | 0.806 |
| Oracle | 0.606 | 0.763 | 0.852 | 0.909 | 0.948 |

Interpretation:

The first bitplane-only linear scout was worse than random. Adding cheap spatial tile features made the scout useful at tight tile budgets, especially K=4 and K=8. A tiny 64-hidden-unit MLP improves coverage further while staying lightweight. There is still large headroom to the oracle upper bound.

### Selector Baseline Pass

Verified on the Windows RTX 4090 machine from clean exported commit `c1418b8` against the full val feature cache:

| Method | K=3 | K=5 | K=8 | Selected Area at K=8 |
|---|---:|---:|---:|---:|
| Random, 5 trials | 0.192 +/- 0.003 | 0.307 +/- 0.005 | 0.452 +/- 0.004 | 0.416 |
| Spatial prior from train split | 0.257 | 0.322 | 0.469 | 0.250 |
| Content heuristic | 0.192 | 0.300 | 0.447 | 0.317 |
| Spatial MLP scout | 0.307 | 0.410 | 0.516 | 0.274 |
| Oracle-count | 0.554 | 0.658 | 0.763 | 0.272 |
| Oracle-greedy set cover | 0.693 | 0.846 | 0.953 | 0.398 |

Interpretation:

The spatial MLP scout is better than random, a simple train-split spatial prior, and the content-only heuristic at K=8. The gap between scout K=8 recall `0.516` and greedy-oracle K=8 recall `0.953` is useful: it says the routing problem is not saturated, and future scout work has real headroom.

## YOLO Routing Smoke Results

Class-agnostic recall against VisDrone boxes with COCO-pretrained `yolov8n.pt`.

| Method | Images | Tiles | Area | Recall | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Full-image YOLO | 25 | 0 | 1.000 | 0.102 | 28.2 ms |
| Random K=8 | 25 | 8 | 0.406 | 0.098 | 39.2 ms |
| Oracle K=8 | 25 | 8 | 0.265 | 0.162 | 41.8 ms |
| Spatial MLP scout K=8, cached scores | 25 | 8 | 0.284 | 0.145 | 41.3 ms |
| Spatial MLP scout K=8, live scores | 25 | 8 | 0.284 | 0.145 | 174.1 ms |
| All tiles | 10 | 49 | 1.000 | 0.228 | 156.2 ms |

Live spatial MLP scout phase timing on the 25-image smoke run:

| Phase | Mean |
|---|---:|
| Image load / resize / labels | 11.0 ms |
| Scout feature + score | 130.8 ms |
| YOLO selected tiles | 31.0 ms |
| Merge | 1.0 ms |

Tiny one-image CUDA smoke from clean exported commit `c1418b8`, used only to prove the new selectors execute end to end:

| Method | Images | Tiles | Area | Recall | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Full-image YOLO | 1 | 0 | 1.000 | 0.024 | 405.1 ms |
| All tiles | 1 | 49 | 1.000 | 0.102 | 494.8 ms |
| Random K=8 | 1 | 8 | 0.391 | 0.063 | 390.2 ms |
| Spatial prior K=8 | 1 | 8 | 0.250 | 0.079 | 388.5 ms |
| Content heuristic K=8 | 1 | 8 | 0.328 | 0.016 | 517.7 ms |
| Oracle-greedy K=8 | 1 | 8 | 0.391 | 0.087 | 376.7 ms |
| Spatial MLP scout K=8, cached scores | 1 | 8 | 0.234 | 0.102 | 398.3 ms |

This one-image YOLO table is a smoke test, not a performance claim. It confirms that `full`, `all`, `random`, `prior`, `heuristic`, `oracle-greedy`, and cached `scout` all run on Windows/CUDA and write JSON metrics.

Timing schema smoke from clean exported commit `baf87f3`, using cached scout K=8, one warmup image, and one measured val image:

| Phase | Mean |
|---|---:|
| Image load | 6.4 ms |
| Resize/preprocess | 10.1 ms |
| Ground-truth parse | 0.6 ms |
| Scout | 0.0 ms |
| YOLO | 17.2 ms |
| Merge/NMS | 1.1 ms |
| Match/eval | 0.8 ms |
| Pipeline latency excluding ground truth | 34.8 ms |
| Wall time | 36.5 ms |

`run_yolo_tiles.py` now reports phase summaries with mean, p50, p95, min, and max. The printed latency is `pipeline_ms_excl_gt`, not wall time with ground-truth parsing mixed in.

## What This Supports

The pipeline is now runnable end to end:

```text
VisDrone image -> tile labels -> scout features -> top-K tiles -> YOLO on selected crops -> merged detections -> JSON metrics
```

The scout can also render a tile-score heatmap with top-K borders via `scripts/render_scout_heatmap.py`.

The best current claim is narrow and defensible:

```text
On full-val tile-label evaluation, a cheap spatial MLP scout selects higher-value tiles than random, a train-split spatial prior, and a content-only heuristic at K=8 while using only 8 of 49 candidate tiles.
```

## What Not To Claim Yet

Do not claim:

```text
The system beats YOLO overall.
The binary scout is already fast enough for edge deployment.
The current YOLO numbers are final VisDrone accuracy.
The bitplane-only scout works well.
```

The live scout timing shows that feature extraction is currently the main optimization target. Cached routing is useful for isolating detector behavior, but live routing is the honest end-to-end benchmark.

## Next Best Improvements

1. Optimize live scout feature extraction, especially avoiding repeated full-image Python/PIL work.
2. Train a small CNN or MLP scout on richer tile features if the spatial linear scout is not strong enough.
3. Add K sweeps for routed YOLO beyond K=8.
4. Run all-tile YOLO on a larger subset for a better SAHI-like cost/recall reference.
5. Decide whether YOLO must be fine-tuned on VisDrone for class-aware mAP, or whether class-agnostic smoke recall is enough for the school scope.
