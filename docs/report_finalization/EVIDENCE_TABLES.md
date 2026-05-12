# Final Evidence Tables

Source of truth: `artifacts/benchmarks/oracle_rank_final/summary.md`.

All detector rows use the first 100 official VisDrone validation images, COCO-pretrained `yolov8n.pt`, original-resolution crops for tiled methods, the 49-tile grid, and Windows RTX 4090 timing. The metrics are class-agnostic pipeline recall and latency, not official VisDrone mAP.

## Main Detector Benchmark

| Method | K | Recall | Small recall | Tiles | Mean latency | p95 latency | Scout | YOLO |
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

## XNOR K-Sweep

| Route | K12 recall | K12 small | K16 recall | K16 small | K18 recall | K18 small | K20 recall | K20 small | K24 recall | K24 small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Objectness XNOR | 0.299 | 0.259 | 0.336 | 0.299 | 0.343 | 0.306 | 0.349 | 0.311 | 0.363 | 0.321 |
| Oracle-rank XNOR | 0.316 | 0.285 | 0.338 | 0.303 | 0.344 | 0.309 | 0.348 | 0.310 | 0.361 | 0.320 |

## Oracle Upper Bound

| Oracle route | K | Recall | Small recall | Mean latency | Percent exhaustive small recall |
|---|---:|---:|---:|---:|---:|
| Oracle greedy | 12 | 0.363 | 0.329 | 32.9 ms | 92.7% |
| Oracle greedy | 16 | 0.378 | 0.342 | 39.8 ms | 96.3% |
| Oracle greedy | 18 | 0.384 | 0.345 | 44.0 ms | 97.2% |
| Oracle greedy | 24 | 0.395 | 0.352 | 56.8 ms | 99.2% |

## Scout-Only Gate

| Scout route | K12 | K16 | K18 | K20 | K24 |
|---|---:|---:|---:|---:|---:|
| Objectness XNOR object recall | 0.767 | 0.899 | 0.942 | 0.964 | 0.973 |
| Oracle-rank XNOR object recall | 0.833 | 0.922 | 0.946 | 0.960 | 0.969 |

## LaTeX Snippets

```latex
\begin{table}[t]
\centering
\caption{Final detector benchmark on 100 VisDrone validation images. Tiled methods use original-resolution crops.}
\label{tab:final_detector}
\begin{tabular}{lrrrrr}
\toprule
Method & $K$ & Recall & Small recall & Tiles & Mean ms \\
\midrule
Full YOLO & -- & 0.129 & 0.084 & 0 & 13.7 \\
Exhaustive tiled YOLO & 49 & 0.399 & 0.355 & 49 & 107.2 \\
Oracle greedy & 12 & 0.363 & 0.329 & 12 & 32.9 \\
Oracle greedy & 18 & 0.384 & 0.345 & 18 & 44.0 \\
Learned GPU objectness & 20 & 0.357 & 0.317 & 20 & 54.6 \\
Objectness XNOR & 18 & 0.343 & 0.306 & 18 & 64.0 \\
Oracle-rank XNOR & 18 & 0.344 & 0.309 & 18 & 63.8 \\
Oracle-rank XNOR & 24 & 0.361 & 0.320 & 24 & 75.8 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Scout-only object-center recall for native XNOR scouts on the full validation split.}
\label{tab:scout_gate}
\begin{tabular}{lrrrrr}
\toprule
Scout & $K=12$ & $K=16$ & $K=18$ & $K=20$ & $K=24$ \\
\midrule
Objectness XNOR & 0.767 & 0.899 & 0.942 & 0.964 & 0.973 \\
Oracle-rank XNOR & 0.833 & 0.922 & 0.946 & 0.960 & 0.969 \\
\bottomrule
\end{tabular}
\end{table}
```
