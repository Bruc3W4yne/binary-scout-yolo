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
```

Full binary-XNOR feature extraction may take hours because the current native feature extractor is CPU-oriented. Use `--max-images 25` first if you only need a smoke result.

## Metrics To Report

Report class-agnostic detector recall, small/medium/large object recall, selected tile count, selected area fraction, detector calls, mean latency, p95 latency, crop source, and the timing phase breakdown written by `run_yolo_tiles.py`.

The JSON rows include `detector_calls`; the summary includes `mean_detector_calls`. This counts detector inputs/crops evaluated: 1 for full-image YOLO, 49 for all tiles, and the selected crop count for routed modes. It does not count Python `model.predict()` invocations.

The JSON summary includes `small_object_recall`, `medium_object_recall`, and `large_object_recall` using COCO-style area thresholds on the 640x640 evaluation canvas by default: small `< 32^2`, medium `< 96^2`, large `>= 96^2`.

The phase `pipeline_ms_excl_gt` is the main latency number. Ground-truth parsing and match/eval timings are reported for measurement transparency, not deployment latency claims.

## Report Table

| Method | Crop source | Detector calls | Tile budget | Recall | Small recall | Mean latency | p95 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Full image YOLO | resized full image | 1 | full image | measured | measured | measured | measured |
| All tiles | original | 49 | all tiles | measured | measured | measured | measured |
| Random top-K | original | K | selected | measured | measured | measured | measured |
| Train-split spatial prior | original | K | selected | measured | measured | measured | measured |
| Content heuristic | original | K | selected | measured | measured | measured | measured |
| Oracle-greedy | original | K | selected | upper bound | upper bound | measured | measured |
| Cached scout | original | K | selected | measured | measured | measured | measured |
| Live scout | original | K | selected | measured | measured | measured | measured |

## Interpretation Rules

Cached scout results isolate detector routing quality. Live scout results are the honest end-to-end latency path.

The high-resolution tiling claim only applies to tiled detector runs with `--crop-source original`.

COCO-pretrained `yolov8n.pt` smoke runs are wiring and tradeoff evidence. Final VisDrone mAP requires a VisDrone-compatible detector or fine-tuning and is outside this cleanup pass.
