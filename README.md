# Binary Scout YOLO

Binary scout-guided selective tiling for UAV small-object detection.

The pipeline goal is:

```text
UAV image -> binary/XNOR scout tile scores -> top-K tiles -> YOLO on selected crops -> merged detections
```

The current repo implements the verified binary core, VisDrone tile dataset generation, C-free and binary-XNOR scout features, scout recall evaluation, and YOLO selected-tile routing.

See `docs/current_results.md` for the latest Windows RTX 4090 smoke results and caveats. See `docs/research_framing.md` for the defensible novelty/claim framing, and `docs/completion_audit.md` for the current phase-gate handoff.

## Windows Quickstart

From PowerShell in the repo:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
python scripts\verify_detector_utils.py
python scripts\verify_routing.py
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

For a quick smoke run, add `--max-images 25`; limited outputs are written with an `_n25` suffix so they cannot be mistaken for full-split caches.

Extract binary-XNOR scout features:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val --max-images 25
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_val_n25.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
```

Remove `--max-images` when ready for the full split.

Run detector/router smoke checks:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector all --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector random --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector prior --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector heuristic --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector oracle-greedy --top-k 8 --split val --max-images 2 --device cuda
python scripts\run_yolo_tiles.py --selector scout --top-k 8 --split val --max-images 2 --device cuda --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
```

`scout` uses cached tile scores to isolate detector routing. `scout-live` computes bitplane scout features inside the timed path and reports phase timings. `prior`, `heuristic`, and `oracle-greedy` are baselines that keep the scout claim honest. These detector numbers are a pipeline smoke signal, not final VisDrone accuracy: `yolov8n.pt` is COCO-pretrained unless you later fine-tune or replace the weights.

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

VisDrone valid object categories are original `category_id` 1 through 10, stored as zero-based `class_id` 0 through 9. Ignored regions and category 11 are skipped.

## Source Notes

The official VisDrone dataset repo lists DET train and val splits for Task 1. The downloader uses Ultralytics' GitHub release mirror for script-friendly zip downloads while preserving the same split names and layout.
