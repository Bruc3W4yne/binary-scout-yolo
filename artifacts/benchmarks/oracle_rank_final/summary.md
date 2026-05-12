# Oracle-Rank XNOR Final Benchmark

Date: 2026-05-12  
Machine: Windows desktop, RTX 4090 FE, CUDA PyTorch  
Repo: clean clone at `C:\Users\Bruc3W4yne\binary-scout-yolo-final`  
Split: first 100 official VisDrone val images  
Detector: COCO-pretrained `yolov8n.pt`  
Crop mode: original-resolution tiles for every tiled run  
Tile grid: 49 tiles, 160x160 on the 640 evaluation canvas, stride 80  

## What Changed

This pass added oracle-rank supervision for the existing 320-input STE heatmap scout. The deployed route is still the PowerPoint-aligned path:

`image -> CPU native XNOR heatmap scout -> tile scores -> tile-NMS top-K -> GPU YOLO on selected original-resolution crops -> merge/evaluate`

The detector, YOLO weights, tile grid, crop geometry, NMS, timing rules, and selected-tile batching were not changed.

## Training Runs

| Checkpoint | Target | Epochs | Batch | Input | Notes |
|---|---:|---:|---:|---:|---|
| `heatmap_ste_320_objectness_e10.pt` | objectness | 10 | 32 | 320 | Same-code baseline |
| `heatmap_ste_320_oracle_rank_e10.pt` | oracle-rank | 10 | 32 | 320 | Rank <=18 positive, weights 3/2/1 |

Scout-only gate with native XNOR + tile-NMS:

| Route | K12 | K16 | K18 | K20 | K24 |
|---|---:|---:|---:|---:|---:|
| Objectness XNOR object recall | 0.767 | 0.899 | 0.942 | 0.964 | 0.973 |
| Oracle-rank XNOR object recall | 0.833 | 0.922 | 0.946 | 0.960 | 0.969 |

The gate passed for the intended low-K direction: oracle-rank improved K12, K16, and K18, but not K20/K24.

## Detector Benchmarks

| Run | K | Recall | Small Recall | Tiles | Mean Latency | p95 Latency | Scout | YOLO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full YOLO | - | 0.129 | 0.084 | 0 | 13.7 ms | 22.5 ms | 0.0 ms | 5.3 ms |
| Exhaustive tiled YOLO | 49 | 0.399 | 0.355 | 49 | 107.2 ms | 120.6 ms | 0.0 ms | 93.7 ms |
| Oracle greedy | 12 | 0.363 | 0.329 | 12 | 32.9 ms | 41.1 ms | 0.0 ms | 22.0 ms |
| Oracle greedy | 16 | 0.378 | 0.342 | 16 | 39.8 ms | 46.7 ms | 0.0 ms | 28.4 ms |
| Oracle greedy | 18 | 0.384 | 0.345 | 18 | 44.0 ms | 50.8 ms | 0.0 ms | 32.3 ms |
| Oracle greedy | 24 | 0.395 | 0.352 | 24 | 56.8 ms | 66.1 ms | 0.0 ms | 44.3 ms |
| Learned GPU objectness | 20 | 0.357 | 0.317 | 20 | 54.6 ms | 61.3 ms | 8.6 ms | 35.3 ms |
| Learned GPU oracle-rank | 20 | 0.354 | 0.317 | 20 | 54.6 ms | 59.7 ms | 8.6 ms | 35.2 ms |
| Objectness XNOR | 12 | 0.299 | 0.259 | 12 | 51.7 ms | 58.5 ms | 19.7 ms | 21.9 ms |
| Objectness XNOR | 16 | 0.336 | 0.299 | 16 | 59.1 ms | 64.5 ms | 19.9 ms | 28.7 ms |
| Objectness XNOR | 18 | 0.343 | 0.306 | 18 | 64.0 ms | 71.7 ms | 20.3 ms | 32.8 ms |
| Objectness XNOR | 20 | 0.349 | 0.311 | 20 | 67.1 ms | 74.6 ms | 20.3 ms | 35.9 ms |
| Objectness XNOR | 24 | 0.363 | 0.321 | 24 | 75.9 ms | 84.4 ms | 20.1 ms | 44.5 ms |
| Oracle-rank XNOR | 12 | 0.316 | 0.285 | 12 | 52.1 ms | 58.8 ms | 19.8 ms | 22.2 ms |
| Oracle-rank XNOR | 16 | 0.338 | 0.303 | 16 | 59.2 ms | 65.1 ms | 19.9 ms | 28.7 ms |
| Oracle-rank XNOR | 18 | 0.344 | 0.309 | 18 | 63.8 ms | 69.7 ms | 20.0 ms | 33.0 ms |
| Oracle-rank XNOR | 20 | 0.348 | 0.310 | 20 | 66.7 ms | 74.0 ms | 20.1 ms | 35.8 ms |
| Oracle-rank XNOR | 24 | 0.361 | 0.320 | 24 | 75.8 ms | 85.7 ms | 19.9 ms | 44.6 ms |

## Decision

The oracle-rank supervision improved the binary/XNOR scout at low K, especially K12:

- K12: +0.017 overall recall, +0.026 small recall, +0.4 ms latency.
- K16: +0.002 overall recall, +0.004 small recall, equal latency.
- K18: +0.002 overall recall, +0.003 small recall, slightly lower latency.

It did not hit the primary success target:

- Target: K18 recall >= 0.355, small recall >= 0.318, latency <= 60.1 ms.
- Actual: K18 recall 0.344, small recall 0.309, latency 63.8 ms.

K24 reaches stronger recall recovery, but it spends too much latency and too many detector crops to be the headline selective-routing win.

## Report Framing

Use this as an honest partial success:

- The full pipeline is implemented and benchmarked end to end.
- Tiling materially improves small-object recall over full-image YOLO.
- Oracle tile selection proves that few high-value tiles can recover most exhaustive-tiling recall.
- The learned scout can approach the oracle trend, and oracle-rank supervision moves useful detections earlier for low-K routing.
- The native CPU XNOR path is real, measured, and PowerPoint-aligned, but its current implementation has about 20 ms scout overhead on this machine, so it does not beat the GPU learned-scout latency on the RTX 4090.
- The final limitation is not the detector or tiling setup; it is the ranking quality and CPU overhead of the current XNOR scout.

