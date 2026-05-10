# Current Results

Generated on the Windows RTX 4090 machine during the first end-to-end implementation pass.

These are not final paper-quality numbers. They are pipeline validation numbers that show the project is runnable and where the next optimization work should focus.

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

## Tile Coverage

Object-center recall on the full VisDrone val tile labels:

| Method | K=4 | K=8 | K=12 | K=16 | K=20 |
|---|---:|---:|---:|---:|---:|
| Random | 0.249 | 0.447 | 0.598 | 0.721 | 0.809 |
| Bitplane stats scout | 0.199 | 0.362 | 0.514 | 0.642 | 0.746 |
| Spatial bitplane scout | 0.320 | 0.492 | 0.601 | 0.695 | 0.773 |
| Oracle | 0.606 | 0.763 | 0.852 | 0.909 | 0.948 |

Interpretation:

The first bitplane-only linear scout was worse than random. Adding cheap spatial tile features made the scout useful at tight tile budgets, especially K=4 and K=8. There is still large headroom to the oracle upper bound.

## YOLO Routing Smoke Results

Class-agnostic recall against VisDrone boxes with COCO-pretrained `yolov8n.pt`.

| Method | Images | Tiles | Area | Recall | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Full-image YOLO | 25 | 0 | 1.000 | 0.102 | 28.2 ms |
| Random K=8 | 25 | 8 | 0.406 | 0.098 | 39.2 ms |
| Oracle K=8 | 25 | 8 | 0.265 | 0.162 | 41.8 ms |
| Spatial scout K=8, cached scores | 25 | 8 | 0.284 | 0.144 | 39.7 ms |
| Spatial scout K=8, live scores | 25 | 8 | 0.284 | 0.144 | 185.2 ms |
| All tiles | 10 | 49 | 1.000 | 0.228 | 156.2 ms |

Live spatial scout phase timing on the 25-image smoke run:

| Phase | Mean |
|---|---:|
| Image load / resize / labels | 21.7 ms |
| Scout feature + score | 142.4 ms |
| YOLO selected tiles | 31.6 ms |
| Merge | 0.4 ms |

## What This Supports

The pipeline is now runnable end to end:

```text
VisDrone image -> tile labels -> scout features -> top-K tiles -> YOLO on selected crops -> merged detections -> JSON metrics
```

The best current claim is narrow and defensible:

```text
On an initial smoke benchmark, a cheap spatial scout selected higher-value tiles than random at K=8 and approached the oracle direction while using only 8 of 49 tiles.
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
