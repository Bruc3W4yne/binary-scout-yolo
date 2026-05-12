# Binary/XNOR Heatmap Final Decision

Date: 2026-05-12

Final decision: `BINARY_XNOR_HELPS=no` under the strict benchmark contract.

This is not a failure of the project pipeline. It is the honest result of the final binary/XNOR pass: selective tiling works, the heatmap scout idea works, and the native XNOR scout is implemented and measurable, but the current CPU XNOR route does not simultaneously reach the required recall recovery and latency target.

## What Was Tested

- Same detector route for all tiled runs: original-resolution crops into `yolov8n.pt`.
- Same 100-image VisDrone validation slice for the final comparison.
- No deployable selector used validation labels before selecting tiles.
- Routing policies tested: top-K, tile-NMS, MMR, heatmap-coverage, and adaptive heatmap coverage.
- Final XNOR route tested with both the original 640-trained STE checkpoint and a 320-native STE fine-tune.

## Decision Criteria

A positive XNOR result required:

- no-GT XNOR route,
- mean detector calls <= 24,
- latency <= 60% of exhaustive tiled YOLO latency,
- overall and small-object gain recovery >= 85%,
- and matched-budget improvement over current XNOR-320 top-K by at least 0.015 recall or small recall, or within 0.02 of oracle.

## Result

Exhaustive tiled YOLO reached `0.399` recall and `0.355` small-object recall at `107.2 ms` in the final oracle-rank benchmark artifact.

The best strict-budget XNOR result was:

| Route | Calls | Recall | Small Recall | Latency | Gain Recovery | Small Gain Recovery | Cost vs All |
|---|---:|---:|---:|---:|---:|---:|---:|
| XNOR-320 tile-NMS K18, oracle-rank checkpoint | 18 | 0.344 | 0.309 | 63.8 ms | 0.80 | 0.83 | 0.59 |

The best recall XNOR result was:

| Route | Calls | Recall | Small Recall | Latency | Gain Recovery | Small Gain Recovery | Cost vs All |
|---|---:|---:|---:|---:|---:|---:|---:|
| XNOR-320 tile-NMS K24, oracle-rank checkpoint | 24 | 0.361 | 0.320 | 75.8 ms | 0.86 | 0.87 | 0.71 |

K18 is close but misses both the strict recall and latency targets. K24 recovers more recall, but it spends too much latency and too many detector crops to be the low-cost headline.

## Blocker

Primary blocker: CPU XNOR scout latency plus remaining scout-ranking gap.

Oracle-greedy proves the grid and detector can support the desired tradeoff: K12 reaches `0.363` recall and `0.329` small recall at `32.9 ms`, K18 reaches `0.384` recall and `0.345` small recall at `44.0 ms`, and K24 reaches `0.395` recall and `0.352` small recall at `56.8 ms`. The learned GPU heatmap scout remains a useful desktop reference, but the final gap is not the selective-tiling concept; it is the current native CPU XNOR scout implementation and fixed-K ranking quality.

## Report Framing

Defensible claim:

> A scout-guided selective tiling pipeline can recover much of exhaustive tiling's small-object recall at lower detector cost. Oracle selection demonstrates the opportunity; the native CPU XNOR scout is implemented and benchmarked, but its final detector-level result is partial validation rather than a full success against the strict K18 latency-recall target.

Avoid claiming:

- Binary/XNOR beats YOLO.
- Native XNOR is already the fastest final route.
- The custom binary kernel alone improves detection accuracy.

## Next Work If We Had More Time

- Move more of the XNOR scout postprocess into C to reduce the current CPU overhead.
- Train the scout with a ranking/coverage objective closer to the final tile selection metric.
- Benchmark on actual edge hardware where CPU/GPU power and availability differ from a 4090 desktop.
- Fine-tune the detector on VisDrone; current detector numbers use COCO-pretrained `yolov8n.pt`.
