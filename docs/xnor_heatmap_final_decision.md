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

Exhaustive tiled YOLO reached `0.399` recall and `0.355` small-object recall at `100.2 ms`.

The best strict-budget XNOR result was:

| Route | Calls | Recall | Small Recall | Latency | Gain Recovery | Small Gain Recovery | Cost vs All |
|---|---:|---:|---:|---:|---:|---:|---:|
| XNOR-320 tile-NMS K18, 320-trained checkpoint | 18 | 0.347 | 0.308 | 56.9 ms | 0.80 | 0.83 | 0.57 |

The best recall XNOR result was:

| Route | Calls | Recall | Small Recall | Latency | Gain Recovery | Small Gain Recovery | Cost vs All |
|---|---:|---:|---:|---:|---:|---:|---:|
| XNOR-320 tile-NMS K24, 320-trained checkpoint | 24 | 0.361 | 0.323 | 68.4 ms | 0.86 | 0.88 | 0.68 |

K18 is cheap enough but not accurate enough. K24 is accurate enough but not cheap enough.

## Blocker

Primary blocker: CPU XNOR scout latency plus remaining scout-ranking gap.

Oracle-greedy proves the grid and detector can support the desired tradeoff: K12 reaches `0.363` recall and `0.329` small recall at `31.3 ms`. The learned GPU heatmap scout also reaches the target recovery at K20 with `51.4 ms`. The gap is therefore not the selective-tiling concept; it is the current native CPU XNOR scout implementation and ranking quality.

## Report Framing

Defensible claim:

> A scout-guided selective tiling pipeline can recover most of exhaustive tiling's small-object recall at substantially lower detector cost. In our implementation, the learned heatmap scout met the target tradeoff, while the native CPU XNOR scout was implemented and benchmarked but did not beat the learned scout under the final latency-recall criterion.

Avoid claiming:

- Binary/XNOR beats YOLO.
- Native XNOR is already the fastest final route.
- The custom binary kernel alone improves detection accuracy.

## Next Work If We Had More Time

- Move more of the XNOR scout postprocess into C to reduce the current CPU overhead.
- Train the scout with a ranking/coverage objective closer to the final tile selection metric.
- Benchmark on actual edge hardware where CPU/GPU power and availability differ from a 4090 desktop.
- Fine-tune the detector on VisDrone; current detector numbers use COCO-pretrained `yolov8n.pt`.
