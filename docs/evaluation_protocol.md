# Evaluation Protocol

This repo is benchmark-ready when it can compare tile routing choices without code changes. Smoke runs prove wiring; longer runs produce report numbers.

## Detector Crop Source

`run_yolo_tiles.py` has two tiled-detector crop modes:

- `--crop-source resized`: crop detector tiles from the already resized 640x640 image. This preserves the older behavior and is useful as a control.
- `--crop-source original`: select tiles on the same 640x640 scout grid, map those tiles back to the original image, run YOLO on the original-resolution crop, then project detections back to the 640x640 evaluation canvas.

Use `--crop-source original` for the final small-object/high-resolution tiling claim. Without it, the benchmark is only a resized-canvas routing demo.

## Required Comparisons

Tile-label recall:

```powershell
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode random --random-trials 5 --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode prior --prior-features data\tile_features\bitplane_stats_spatial_train.npz --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode heuristic --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-count --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-greedy --top-k-values 4 8 12 16 20
```

YOLO routing smoke:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector all --crop-source original --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector random --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector prior --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector heuristic --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector oracle-greedy --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector scout --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector scout-live --crop-source original --top-k 8 --split val --max-images 25 --device cuda --weights yolov8n.pt --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
```

Longer same-subset K sweep, written as explicit commands instead of a new experiment framework.
Run the K-independent references once:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 100 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector all --crop-source original --split val --max-images 100 --device cuda --weights yolov8n.pt
```

Then sweep the routed selectors:

```powershell
foreach ($k in 4,8,12,16,20) {
  python scripts\run_yolo_tiles.py --selector random --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector prior --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector heuristic --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector oracle-greedy --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt
  python scripts\run_yolo_tiles.py --selector scout --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
  python scripts\run_yolo_tiles.py --selector scout-live --crop-source original --top-k $k --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
}
```

Binary-XNOR scout comparison:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_train.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_val.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
python scripts\train_scout.py --train-features data\tile_features\binary_xnor_train.npz --val-features data\tile_features\binary_xnor_val.npz --out-dir runs\scout_binary_xnor_mlp --hidden-dim 64 --epochs 30 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor_val.npz --checkpoint runs\scout_binary_xnor_mlp\scout_binary_xnor.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\run_yolo_tiles.py --selector scout --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --features data\tile_features\binary_xnor_val.npz --checkpoint runs\scout_binary_xnor_mlp\scout_binary_xnor.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor_mlp\scout_binary_xnor.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 12 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor_mlp\scout_binary_xnor.pt
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split val
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_hybrid_train.npz --expect-feature-dim 72 --expect-feature-mode binary-xnor-hybrid
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_hybrid_val.npz --expect-feature-dim 72 --expect-feature-mode binary-xnor-hybrid
python scripts\train_scout.py --train-features data\tile_features\binary_xnor_hybrid_train.npz --val-features data\tile_features\binary_xnor_hybrid_val.npz --out-dir runs\scout_binary_xnor_hybrid_mlp --hidden-dim 64 --epochs 30 --count-alpha 0.2 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor_hybrid_val.npz --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 12 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt
python scripts\add_spatial_features.py --features data\tile_features\binary_xnor_train.npz --out data\tile_features\binary_xnor8_hybrid_train.npz --keep-first-features 8
python scripts\add_spatial_features.py --features data\tile_features\binary_xnor_val.npz --out data\tile_features\binary_xnor8_hybrid_val.npz --keep-first-features 8
python scripts\train_scout.py --train-features data\tile_features\binary_xnor8_hybrid_train.npz --val-features data\tile_features\binary_xnor8_hybrid_val.npz --out-dir runs\scout_binary_xnor8_hybrid_mlp --hidden-dim 64 --epochs 30 --count-alpha 0.2 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor8_hybrid_val.npz --checkpoint runs\scout_binary_xnor8_hybrid_mlp\scout_binary_xnor_hybrid.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor8_hybrid_mlp\scout_binary_xnor_hybrid.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 12 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor8_hybrid_mlp\scout_binary_xnor_hybrid.pt
```

Full binary-XNOR feature extraction may take a while because the current native feature extractor is CPU-oriented. Use `--max-images 25` first if you only need a smoke result. `binary-xnor-hybrid` includes XNOR summaries plus cheap tile geometry/spatial-prior features; report it honestly as hybrid, not as a pure binary scout. The `--keep-first-features` option is for a binary filter-budget sweep; it reduces live XNOR compute while keeping the same seed/order and updating checkpoint metadata.

## Metrics To Report

Report class-agnostic detector recall, class-agnostic precision/F1, false positives, small/medium/large object recall, selected tile count, selected area fraction, detector calls, mean latency, p95 latency, crop source, and the timing phase breakdown written by `run_yolo_tiles.py`.

The JSON rows include `detector_calls`; the summary includes `mean_detector_calls`. This counts detector inputs/crops evaluated: 1 for full-image YOLO, 49 for all tiles, and the selected crop count for routed modes. It does not count Python `model.predict()` invocations.

The JSON summary includes `small_object_recall`, `medium_object_recall`, and `large_object_recall` using COCO-style area thresholds on the 640x640 evaluation canvas by default: small `< 32^2`, medium `< 96^2`, large `>= 96^2`.

The phase `pipeline_ms_excl_gt` is the main latency number. Ground-truth parsing and match/eval timings are reported for measurement transparency, not deployment latency claims.

For live binary-XNOR routes, use the nested `scout_timing_ms` row field or the flattened `scout_bitplanes_ms`, `scout_pack_ms`, `scout_xnor_kernel_ms`, `scout_threshold_ms`, `scout_tile_summary_ms`, `scout_spatial_ms`, `scout_mlp_ms`, and `scout_topk_ms` phase summaries. Cached binary results are routing-quality baselines, not fair end-to-end latency numbers.

## Report Table

| Method | Crop source | Detector calls | Tile budget | Recall | Small recall | Precision | Mean latency | p95 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full image YOLO | resized full image | 1 | full image | measured | measured | measured | measured | measured |
| All tiles | original | 49 | all tiles | measured | measured | measured | measured | measured |
| Random top-K | original | K | selected | measured | measured | measured | measured | measured |
| Train-split spatial prior | original | K | selected | measured | measured | measured | measured | measured |
| Content heuristic | original | K | selected | measured | measured | measured | measured | measured |
| Oracle-greedy | original | K | selected | upper bound | upper bound | measured | measured | measured |
| Cached scout | original | K | selected | measured | measured | measured | measured | measured |
| Live spatial scout | original | K | selected | measured | measured | measured | measured | measured |
| Live binary-XNOR pure | original | K | selected | measured | measured | measured | measured | measured |
| Live binary-XNOR hybrid | original | K | selected | measured | measured | measured | measured | measured |

## Interpretation Rules

Cached scout results isolate detector routing quality. Live scout results are the honest end-to-end latency path.

Pure binary-XNOR and binary-XNOR hybrid should be reported separately. The hybrid is the intended final binary route if it improves recall because it keeps the C XNOR-popcount visual features while adding low-cost tile priors that a practical router would know.

The high-resolution tiling claim only applies to tiled detector runs with `--crop-source original`.

COCO-pretrained `yolov8n.pt` smoke runs are wiring and tradeoff evidence. Final VisDrone mAP requires a VisDrone-compatible detector or fine-tuning and is outside this cleanup pass.
