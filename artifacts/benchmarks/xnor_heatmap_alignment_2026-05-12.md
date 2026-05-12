# XNOR Heatmap Alignment Benchmark

Date: 2026-05-12
Host: DESKTOP-JV3PK9G, RTX 4090 FE
Commit: 8347b91
Checkpoint: runs/heatmap_scout/heatmap_ste_e10.pt

This bounded smoke benchmark checks the intended split: CPU native XNOR scout selects tiles, then GPU YOLO detects objects in selected original-resolution crops.

| selector | K | route | recall | small recall | latency ms | scout ms | YOLO ms | detector calls |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| full |  |  | 0.105 | 0.058 | 14.2 | 0.0 | 4.7 | 1.0 |
| all |  |  | 0.253 | 0.194 | 116.3 | 0.0 | 103.7 | 49.0 |
| heuristic | 8 |  | 0.129 | 0.099 | 29.6 | 2.5 | 16.8 | 8.0 |
| learned-heatmap-live | 8 | learned-heatmap-ste | 0.148 | 0.110 | 37.0 | 8.4 | 17.9 | 8.0 |
| learned-heatmap-live | 12 | learned-heatmap-ste | 0.187 | 0.139 | 42.6 | 7.7 | 24.2 | 12.0 |
| learned-heatmap-live | 20 | learned-heatmap-ste | 0.219 | 0.159 | 58.7 | 7.8 | 39.7 | 20.0 |
| learned-heatmap-live | 24 | learned-heatmap-ste | 0.235 | 0.176 | 68.4 | 7.8 | 48.9 | 24.0 |
| xnor-heatmap-live | 8 | xnor-heatmap-live | 0.148 | 0.110 | 90.7 | 62.9 | 17.3 | 8.0 |
| xnor-heatmap-live | 12 | xnor-heatmap-live | 0.187 | 0.139 | 91.7 | 56.4 | 24.7 | 12.0 |
| xnor-heatmap-live | 20 | xnor-heatmap-live | 0.219 | 0.159 | 107.8 | 56.4 | 40.0 | 20.0 |
| xnor-heatmap-live | 24 | xnor-heatmap-live | 0.235 | 0.176 | 116.4 | 56.6 | 48.2 | 24.0 |
| xnor-heatmap-320-live | 8 | xnor-heatmap-320-live | 0.128 | 0.080 | 46.9 | 17.3 | 18.7 | 8.0 |
| xnor-heatmap-320-live | 12 | xnor-heatmap-320-live | 0.167 | 0.113 | 54.3 | 16.8 | 26.6 | 12.0 |
| xnor-heatmap-320-live | 16 | xnor-heatmap-320-live | 0.178 | 0.124 | 62.4 | 16.7 | 34.6 | 16.0 |
| xnor-heatmap-320-live | 20 | xnor-heatmap-320-live | 0.203 | 0.147 | 70.7 | 16.5 | 42.8 | 20.0 |
| xnor-heatmap-320-live | 24 | xnor-heatmap-320-live | 0.215 | 0.156 | 80.1 | 16.6 | 52.0 | 24.0 |
| xnor-heatmap-320-live | 28 | xnor-heatmap-320-live | 0.231 | 0.171 | 91.3 | 17.4 | 61.9 | 28.0 |
| xnor-heatmap-320-live | 32 | xnor-heatmap-320-live | 0.234 | 0.175 | 97.3 | 16.8 | 68.7 | 32.0 |
| xnor-heatmap-320-live | 36 | xnor-heatmap-320-live | 0.243 | 0.183 | 105.8 | 16.9 | 76.6 | 36.0 |
| xnor-heatmap-320-live | 40 | xnor-heatmap-320-live | 0.250 | 0.190 | 114.9 | 16.7 | 85.8 | 40.0 |

## Readout

- Architecture alignment: `xnor-heatmap-live` and `xnor-heatmap-320-live` both run the learned STE heatmap scout binary convolution body through native packed XNOR-popcount on CPU, then route selected tiles to GPU YOLO.
- Performance alignment: the 640 route is correct but too expensive on this workstation. The 320 route creates the defensible Pareto point the original project wanted.
- Best efficiency point: `xnor-heatmap-320-live` at K=36 reaches recall `0.243` and small recall `0.183` at `105.8 ms`, using 36 detector calls.
- Closest all-tile recovery point: `xnor-heatmap-320-live` at K=40 reaches recall `0.250` and small recall `0.190` at `114.9 ms`, compared with exhaustive tiling recall `0.253` / small recall `0.194` at `116.3 ms` and 49 detector calls.
- Claim boundary: these are bounded COCO-pretrained YOLO smoke benchmarks on 30 validation images, not final VisDrone mAP/AP_small. The result supports the selective-tiling pipeline claim; a purpose-trained 320 scout or edge hardware test remains the best next research step.
