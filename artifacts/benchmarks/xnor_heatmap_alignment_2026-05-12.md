# XNOR Heatmap Alignment Benchmark

Date: 2026-05-12
Host: DESKTOP-JV3PK9G, RTX 4090 FE
Commit: 7e03568
Checkpoint: runs/heatmap_scout/heatmap_ste_e10.pt

This bounded smoke benchmark checks whether the PowerPoint-aligned route exists end to end: input image -> learned STE heatmap scout with native C XNOR-popcount binary convs -> top-K tiles -> YOLO -> merged detections.

| selector | K | route | recall | small recall | latency ms | scout ms | YOLO ms | detector calls |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| full |  |  | 0.105 | 0.058 | 14.2 | 0.0 | 4.7 | 1.0 |
| all |  |  | 0.253 | 0.194 | 116.3 | 0.0 | 103.7 | 49.0 |
| heuristic | 8 |  | 0.129 | 0.099 | 29.6 | 2.5 | 16.8 | 8.0 |
| learned-heatmap-live | 3 | learned-heatmap-ste | 0.097 | 0.069 | 26.7 | 8.1 | 8.6 | 3.0 |
| learned-heatmap-live | 5 | learned-heatmap-ste | 0.117 | 0.086 | 30.3 | 8.1 | 11.9 | 5.0 |
| learned-heatmap-live | 8 | learned-heatmap-ste | 0.148 | 0.110 | 37.0 | 8.4 | 17.9 | 8.0 |
| learned-heatmap-live | 12 | learned-heatmap-ste | 0.187 | 0.139 | 42.6 | 7.7 | 24.2 | 12.0 |
| learned-heatmap-live | 16 | learned-heatmap-ste | 0.209 | 0.154 | 53.1 | 7.9 | 33.9 | 16.0 |
| learned-heatmap-live | 20 | learned-heatmap-ste | 0.219 | 0.159 | 58.7 | 7.8 | 39.7 | 20.0 |
| learned-heatmap-live | 24 | learned-heatmap-ste | 0.235 | 0.176 | 68.4 | 7.8 | 48.9 | 24.0 |
| xnor-heatmap-live | 3 | xnor-heatmap-live | 0.097 | 0.069 | 75.1 | 56.6 | 8.7 | 3.0 |
| xnor-heatmap-live | 5 | xnor-heatmap-live | 0.117 | 0.086 | 78.0 | 56.3 | 11.6 | 5.0 |
| xnor-heatmap-live | 8 | xnor-heatmap-live | 0.148 | 0.110 | 90.7 | 62.9 | 17.3 | 8.0 |
| xnor-heatmap-live | 12 | xnor-heatmap-live | 0.187 | 0.139 | 91.7 | 56.4 | 24.7 | 12.0 |
| xnor-heatmap-live | 16 | xnor-heatmap-live | 0.209 | 0.154 | 99.6 | 56.2 | 32.3 | 16.0 |
| xnor-heatmap-live | 20 | xnor-heatmap-live | 0.219 | 0.159 | 107.8 | 56.4 | 40.0 | 20.0 |
| xnor-heatmap-live | 24 | xnor-heatmap-live | 0.235 | 0.176 | 116.4 | 56.6 | 48.2 | 24.0 |

## Readout

- Alignment: `xnor-heatmap-live` now runs the learned heatmap scout binary convolution body through native packed XNOR-popcount and selects live tiles for YOLO.
- Correctness: local and Windows parity tests passed; XNOR heatmap routing matches the PyTorch STE heatmap route on the tested checkpoint.
- Performance: on this RTX 4090 workstation, CUDA PyTorch remains faster for the scout. The native C route is CPU-side; its best bounded point here is K=20 with recall 0.219 at 107.8 ms versus exhaustive tiling recall 0.253 at 116.3 ms.
- Claim boundary: this supports the architecture/operation-layer implementation claim, but not a broad claim that CPU XNOR beats CUDA PyTorch on a 4090. Edge-device speed claims still need Jetson-class or CPU-float baseline measurements.
