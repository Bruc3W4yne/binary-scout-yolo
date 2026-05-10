# Binary Scout YOLO

Binary scout-guided selective tiling for UAV small-object detection.

The pipeline goal is:

```text
UAV image -> binary/XNOR scout tile scores -> top-K tiles -> YOLO on selected crops -> merged detections
```

The current repo implements the verified binary core, VisDrone tile dataset generation, C-free scout features, binary-XNOR scout features, and linear scout recall evaluation. YOLO selected-tile inference is the next major layer.

## Windows Quickstart

From PowerShell in the repo:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
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
python scripts\train_scout.py --epochs 10 --device cuda
python scripts\evaluate_scout_recall.py --mode scout --top-k-values 4 8 12 16 20
python scripts\evaluate_scout_recall.py --mode random --top-k-values 4 8 12 16 20
```

Extract binary-XNOR scout features:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val --max-images 25
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_val.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
```

Remove `--max-images` when ready for the full split.

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
