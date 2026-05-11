# Final Binary-XNOR Pass Results

This pass made the binary path live, measurable, and more aligned with the original scout-guided tiling proposal. Results were run on the Windows RTX 4090 machine with `yolov8n.pt`, the first 100 VisDrone val images, `--crop-source original`, and one warmup image.

The detector is still COCO-pretrained YOLO. These are class-agnostic routing/pipeline results, not official VisDrone mAP.

## What Changed

- `binary-xnor-live` now computes RGB bitplanes, uint64 packing, C XNOR-popcount, threshold maps, tile summaries, MLP scores, top-K routing, YOLO crops, and merge/NMS at inference time.
- Live scout timing now records bitplane, pack, XNOR kernel, threshold, tile summary, spatial, MLP, top-K, YOLO, merge, total, and p95 timing.
- `binary-xnor-hybrid` combines XNOR tile summaries with cheap tile-geometry/spatial features. It is the main binary route, but must be reported as hybrid.
- Scout training supports a count-aware objective: weighted BCE plus optional normalized object-count regression.
- Binary filter budgets can be tested by keeping the first N binary features before adding spatial features.
- Detector summaries now include precision, F1, and false positives.

## Verification

Windows checks passed:

```powershell
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_detector_utils.py
python scripts\verify_routing.py
python scripts\verify_tile_contracts.py
python -m pytest -q
```

`pytest` result: `4 passed`.

## Tile Recall

Object-center recall on full VisDrone val tile labels:

| Scout | K=4 | K=8 | K=12 | K=16 | K=20 |
|---|---:|---:|---:|---:|---:|
| Old binary-XNOR cached | 0.289 | 0.491 | 0.643 | 0.760 | 0.837 |
| Pure binary-XNOR + count loss | 0.297 | 0.511 | 0.653 | 0.766 | 0.836 |
| 8-filter binary-XNOR hybrid | 0.373 | 0.531 | 0.641 | 0.726 | 0.791 |
| 16-filter binary-XNOR hybrid | 0.389 | 0.550 | 0.661 | 0.741 | 0.804 |
| 64-filter binary-XNOR hybrid | 0.380 | 0.553 | 0.666 | 0.752 | 0.817 |
| Spatial scout reference | 0.364 | 0.516 | 0.635 | 0.735 | 0.806 |

The hybrid binary route now beats the old binary scout and the spatial scout at tile recall K=8/K=12. The 8-filter version is the best speed/quality binary candidate.

## Detector Results

Class-agnostic detector benchmark on the first 100 VisDrone val images:

| Method | K | Calls | Recall | Small recall | Precision | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-image YOLO | - | 1 | 0.129 | 0.084 | 0.855 | 12.8 ms | 20.0 ms |
| All tiles | 49 | 49 | 0.399 | 0.355 | 0.531 | 102.8 ms | 115.5 ms |
| Random tiles | 8 | 8 | 0.155 | 0.142 | 0.637 | 22.7 ms | 28.9 ms |
| Spatial scout live | 8 | 8 | 0.228 | 0.210 | 0.646 | 30.0 ms | 37.2 ms |
| Spatial scout live | 12 | 12 | 0.267 | 0.245 | 0.615 | 38.4 ms | 45.8 ms |
| Old binary-XNOR cached | 8 | 8 | 0.191 | 0.168 | 0.630 | 23.8 ms | 30.1 ms |
| Old binary-XNOR cached | 12 | 12 | 0.245 | 0.217 | 0.613 | 30.7 ms | 37.4 ms |
| Pure binary-XNOR live | 8 | 8 | 0.210 | 0.190 | 0.626 | 201.8 ms | 239.5 ms |
| Pure binary-XNOR live | 12 | 12 | 0.264 | 0.239 | 0.614 | 189.6 ms | 200.0 ms |
| 8-filter binary-XNOR hybrid live | 8 | 8 | 0.213 | 0.206 | 0.636 | 56.6 ms | 65.7 ms |
| 8-filter binary-XNOR hybrid live | 12 | 12 | 0.258 | 0.244 | 0.612 | 69.3 ms | 77.0 ms |
| 16-filter binary-XNOR hybrid live | 8 | 8 | 0.218 | 0.208 | 0.647 | 80.7 ms | 88.8 ms |
| 16-filter binary-XNOR hybrid live | 12 | 12 | 0.262 | 0.247 | 0.619 | 78.6 ms | 88.3 ms |
| 64-filter binary-XNOR hybrid live | 8 | 8 | 0.235 | 0.223 | 0.651 | 200.0 ms | 239.4 ms |
| 64-filter binary-XNOR hybrid live | 12 | 12 | 0.281 | 0.262 | 0.622 | 187.7 ms | 197.6 ms |

## Latency Breakdown

Mean live scout timing:

| Route | Scout total | XNOR kernel | Threshold | Tile summary |
|---|---:|---:|---:|---:|
| 8-filter hybrid | 28.8 ms | 15.5 ms | 2.3 ms | 1.0 ms |
| 16-filter hybrid | 50.8 ms | 34.1 ms | 4.7 ms | 2.0 ms |
| Pure 64-filter binary | 138.1 ms | 102.4 ms | 18.4 ms | 7.0 ms |
| 64-filter hybrid | 137.1 ms | 101.3 ms | 18.6 ms | 7.0 ms |

The default-grid tile-summary optimization reduced the 64-filter binary routes from roughly `485 ms` to about `190-200 ms` end-to-end. The remaining binary cost is dominated by the CPU XNOR kernel and thresholding.

## Interpretation

The binary path now genuinely helps compared with the old binary scout:

- K=8 detector recall: `0.191 -> 0.213` for the practical 8-filter live hybrid.
- K=8 small recall: `0.168 -> 0.206`.
- K=12 detector recall: `0.245 -> 0.258`.
- K=12 small recall: `0.217 -> 0.244`.

The pure 64-filter live binary route closes the required ablation:

- K=8: pure binary is `0.210/0.190` total/small recall, while the 8-filter hybrid is `0.213/0.206` and much faster.
- K=12: pure binary is `0.264/0.239`, while the 8-filter hybrid is `0.258/0.244` and much faster.

This means the pure binary features are real and useful, but the practical route is the smaller hybrid. The hybrid keeps the XNOR-popcount visual path, improves K=8 small-object recall, and cuts most of the 64-filter binary latency by using an 8-filter budget plus cheap tile priors.

The 64-filter hybrid even beats spatial scout recall at K=8 and K=12, but it is too slow to be the practical final route.

The best final binary claim is:

```text
A live 8-filter binary-XNOR hybrid scout improves over the previous binary route and recovers much of the tiled-detector small-object gain while using 8-12 detector calls instead of 49. It is faster than exhaustive tiling and much faster than the pure 64-filter binary route, but slower than the current spatial scout because the CPU XNOR kernel remains the bottleneck.
```

Do not claim the binary route is faster than full-image YOLO or faster than the spatial scout. Do claim it is a real live binary scout/router with measured recall and latency, plus a clear filter-budget tradeoff.

## Custom Detector Loss

The PowerPoint detector loss remains future work. The current project deliberately isolates tile routing while keeping YOLO fixed. Adding a custom YOLO loss without VisDrone fine-tuning and official mAP would create a larger experimental burden than this final pass can justify.
