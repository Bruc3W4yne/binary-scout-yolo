# Binary Scout YOLO

Binary scout-guided selective tiling for UAV small-object detection.

The pipeline goal is:

```text
UAV image -> binary/XNOR scout tile scores -> top-K tiles -> YOLO on selected crops -> merged detections
```

The current repo implements the verified binary core, VisDrone tile dataset generation, C-free and binary-XNOR scout features, scout recall evaluation, and YOLO selected-tile routing. Tiled detector runs can now use `--crop-source original` so the scout selects on the 640x640 grid while YOLO runs on original-resolution crops.

See `docs/final_binary_pass_results.md` for the latest Windows RTX 4090 binary-live results. See `docs/current_results.md` for historical smoke results and caveats, `docs/final_project_claims.md` for the defensible novelty/claim framing, `docs/evaluation_protocol.md` for longer benchmark commands, `docs/binary_xnor_core.md` for the native binary core, and `docs/completion_audit.md` for the current phase-gate handoff.

## Windows Quickstart

From PowerShell in the repo:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
python scripts\verify_detector_utils.py
python scripts\verify_routing.py
python -m pytest -q
```

Download VisDrone DET train/val:

```powershell
python scripts\download_visdrone.py --splits train val
python scripts\verify_visdrone_dataset.py --split-mode official
```

Build the tile dataset:

```powershell
python scripts\make_tile_dataset.py --split-mode official
python scripts\verify_tile_dataset.py
```

Extract C-free scout features and train the baseline scout:

```powershell
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split train
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split val
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_train.npz
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_val.npz
python scripts\add_spatial_features.py --features data\tile_features\bitplane_stats_train.npz
python scripts\add_spatial_features.py --features data\tile_features\bitplane_stats_val.npz
python scripts\train_scout.py --train-features data\tile_features\bitplane_stats_spatial_train.npz --val-features data\tile_features\bitplane_stats_spatial_val.npz --out-dir runs\scout_spatial_mlp --hidden-dim 64 --epochs 30 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode random --random-trials 5 --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode prior --prior-features data\tile_features\bitplane_stats_spatial_train.npz --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode heuristic --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-count --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-greedy --top-k-values 4 8 12 16 20
python scripts\render_scout_heatmap.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --mode scout --top-k 8
```

Small feature/training smoke:

```powershell
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split train --max-images 25
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split val --max-images 25
python scripts\add_spatial_features.py --features data\tile_features\bitplane_stats_train_n25.npz
python scripts\add_spatial_features.py --features data\tile_features\bitplane_stats_val_n25.npz
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_spatial_train_n25.npz --expect-feature-dim 38 --expect-feature-mode bitplane-stats-spatial
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_spatial_val_n25.npz --expect-feature-dim 38 --expect-feature-mode bitplane-stats-spatial
python scripts\train_scout.py --train-features data\tile_features\bitplane_stats_spatial_train_n25.npz --val-features data\tile_features\bitplane_stats_spatial_val_n25.npz --out-dir runs\scout_spatial_mlp_n25 --hidden-dim 64 --epochs 3 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val_n25.npz --checkpoint runs\scout_spatial_mlp_n25\scout_bitplane_stats_spatial.pt --mode scout --top-k-values 4 8 --out data\results\smoke_recall_scout_n25.json
```

Only the commands that accept `--max-images` use it. Limited feature outputs are written with an `_n25` suffix so they cannot be mistaken for full-split caches.

Extract binary-XNOR scout features:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val --max-images 25
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_val_n25.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split val --max-images 25
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_hybrid_val_n25.npz --expect-feature-dim 72 --expect-feature-mode binary-xnor-hybrid
```

Remove `--max-images` when ready for the full split.

Train and evaluate cached pure and hybrid binary-XNOR scout comparisons:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val
python scripts\train_scout.py --train-features data\tile_features\binary_xnor_train.npz --val-features data\tile_features\binary_xnor_val.npz --out-dir runs\scout_binary_xnor_mlp --hidden-dim 64 --epochs 30 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor_val.npz --checkpoint runs\scout_binary_xnor_mlp\scout_binary_xnor.pt --mode scout --top-k-values 4 8 12 16 20
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split val
python scripts\train_scout.py --train-features data\tile_features\binary_xnor_hybrid_train.npz --val-features data\tile_features\binary_xnor_hybrid_val.npz --out-dir runs\scout_binary_xnor_hybrid_mlp --hidden-dim 64 --epochs 30 --count-alpha 0.2 --device cuda
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor_hybrid_val.npz --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt --mode scout --top-k-values 4 8 12 16 20
```

For the practical live binary route, reuse the full 64-filter binary cache and keep a smaller filter budget before adding spatial features:

```powershell
python scripts\add_spatial_features.py --features data\tile_features\binary_xnor_train.npz --out data\tile_features\binary_xnor8_hybrid_train.npz --keep-first-features 8
python scripts\add_spatial_features.py --features data\tile_features\binary_xnor_val.npz --out data\tile_features\binary_xnor8_hybrid_val.npz --keep-first-features 8
python scripts\train_scout.py --train-features data\tile_features\binary_xnor8_hybrid_train.npz --val-features data\tile_features\binary_xnor8_hybrid_val.npz --out-dir runs\scout_binary_xnor8_hybrid_mlp --hidden-dim 64 --epochs 30 --count-alpha 0.2 --device cuda
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor8_hybrid_mlp\scout_binary_xnor_hybrid.pt
```

Run detector/router smoke checks:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector all --crop-source original --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector random --crop-source original --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector prior --crop-source original --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector heuristic --crop-source original --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector oracle-greedy --crop-source original --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector scout --crop-source original --top-k 8 --split val --max-images 2 --device cuda --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector scout-live --crop-source original --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt
```

`scout` uses cached tile scores to isolate detector routing. `scout-live` computes the checkpoint's feature mode inside the timed path; `binary-xnor-live` is a stricter alias that requires a binary-XNOR checkpoint and writes bitplane/pack/XNOR/threshold/tile-summary/MLP/top-K timing fields. `prior`, `heuristic`, and `oracle-greedy` are baselines that keep the scout claim honest. Use `--crop-source original` for the high-resolution tiling claim; the default `resized` mode is retained as a control. These detector numbers are a pipeline smoke signal, not final VisDrone accuracy: `yolov8n.pt` is COCO-pretrained unless you later fine-tune or replace the weights.

`run_yolo_tiles.py` uses one warmup image by default and prints pipeline latency excluding ground-truth parsing. The JSON output also includes detector input counts, precision/false-positive sanity metrics, small/medium/large object recall, image-load, resize, ground-truth, scout, YOLO, merge/NMS, match/eval, pipeline, and wall-clock timing summaries. `detector_calls` means detector inputs/crops evaluated, not Python `model.predict()` calls.

## Data Contract

Default image contract:

```text
resize: stretch to 640x640
tile_size: 160
stride: 80
tile starts: 0, 80, 160, 240, 320, 400, 480
tiles per image: 49
positive tile: center of at least one valid VisDrone box lies inside tile
```

The detector can either crop selected tiles from the resized 640x640 canvas or map those selected tile boxes back to original image coordinates with `--crop-source original`.

VisDrone valid object categories are original `category_id` 1 through 10, stored as zero-based `class_id` 0 through 9. Ignored regions and category 11 are skipped.

## Source Notes

The official VisDrone dataset repo lists DET train and val splits for Task 1. The downloader uses Ultralytics' GitHub release mirror for script-friendly zip downloads while preserving the same split names and layout.
