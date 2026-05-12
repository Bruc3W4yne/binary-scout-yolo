# XNOR Final Router Benchmark Summary

Date: 2026-05-12

Machine: Windows PC, NVIDIA RTX 4090, CUDA PyTorch, `yolov8n.pt`.

Dataset slice: first 100 VisDrone validation images.

Detector setup: `--crop-source original`, `--detector-imgsz 640`, `--conf 0.25`, `--yolo-iou 0.7`, `--merge-iou 0.5`, `--match-iou 0.5`.

Success contract for `BINARY_XNOR_HELPS=yes`: no-GT XNOR route, mean detector calls <= 24, latency <= 0.60x exhaustive tiling, overall and small gain recovery >= 0.85, and matched-budget improvement over current XNOR-320 top-K by at least 0.015 recall or small recall, or within 0.02 of oracle.

## Main Results

Gain recovery is measured between full-image YOLO and exhaustive original-crop tiling:

`gain = (route_recall - full_recall) / (all_tiles_recall - full_recall)`.

| Route | Calls | Recall | Small Recall | Mean Latency | P95 Latency | Gain Recovery | Small Gain Recovery | Cost vs All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-image YOLO | 1 | 0.129 | 0.084 | 13.3 ms | 20.1 ms | 0.00 | 0.00 | 0.13 |
| Exhaustive tiled YOLO | 49 | 0.399 | 0.355 | 100.2 ms | 111.5 ms | 1.00 | 1.00 | 1.00 |
| Heuristic K8 | 8 | 0.157 | 0.139 | 25.2 ms | 32.4 ms | 0.10 | 0.20 | 0.25 |
| Oracle-greedy K12 | 12 | 0.363 | 0.329 | 31.3 ms | 36.9 ms | 0.86 | 0.90 | 0.31 |
| Oracle-greedy K20 | 20 | 0.389 | 0.349 | 46.2 ms | 51.2 ms | 0.96 | 0.98 | 0.46 |
| Learned heatmap GPU tile-NMS K20 | 20 | 0.358 | 0.320 | 51.4 ms | 57.8 ms | 0.85 | 0.87 | 0.51 |
| XNOR-320 tile-NMS K20, 640-trained checkpoint | 20 | 0.348 | 0.309 | 59.8 ms | 66.3 ms | 0.81 | 0.83 | 0.60 |
| XNOR-320 tile-NMS K18, 320-trained checkpoint | 18 | 0.347 | 0.308 | 56.9 ms | 62.9 ms | 0.80 | 0.83 | 0.57 |
| XNOR-320 tile-NMS K20, 320-trained checkpoint | 20 | 0.353 | 0.315 | 61.7 ms | 70.2 ms | 0.83 | 0.85 | 0.62 |
| XNOR-320 tile-NMS K24, 320-trained checkpoint | 24 | 0.361 | 0.323 | 68.4 ms | 75.9 ms | 0.86 | 0.88 | 0.68 |

## Interpretation

- The selective-tiling pipeline is viable: the learned heatmap scout with tile-NMS reaches the target gain recovery at about half the latency of exhaustive tiling.
- The native CPU XNOR route improved after 320-native training and tile-NMS, but it does not satisfy the strict positive criterion.
- The best XNOR route inside the latency budget is K18: it costs 57% of exhaustive tiling but reaches only 80% overall gain recovery and 83% small-object gain recovery.
- The best XNOR route by recall is K24: it reaches 86% overall and 88% small-object gain recovery, but costs 68% of exhaustive tiling.
- Oracle-greedy shows this is not only a tiling-grid limitation: with perfect tile choice, K12 already reaches the target. The remaining blockers are XNOR scout ranking quality and native CPU scout overhead.

Decision: `BINARY_XNOR_HELPS=no` under the strict goal contract. The report should frame this honestly: binary/XNOR is implemented and useful as an investigated edge scout, but the current CPU XNOR path does not beat the simpler learned GPU scout under the measured Windows 4090 benchmark.
