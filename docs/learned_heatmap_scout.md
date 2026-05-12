# Learned Heatmap Scout

This pass adds the missing trainable scout route that is closest to the original project idea:

```text
full image -> top-4 grayscale MSB bitplanes -> learned heatmap scout -> top-K tiles -> YOLO crops -> merged detections
```

The scout is still a router, not a replacement detector. YOLO remains responsible for final object detection. The new part is that tile selection can be learned from image pixels instead of using fixed XNOR summaries or hand-written spatial priors.

## What Is Implemented

- `src/heatmap_scout.py`
  - `HeatmapScout(variant="float"|"ste")`
  - input shape `[B,4,640,640]` with top-4 grayscale MSB bitplanes in `{-1,+1}`
  - three Conv/BN/activation/MaxPool blocks: `4->32`, `32->64`, `64->64`
  - output logits `[B,1,80,80]`
  - STE variant signs hidden conv weights and activations during forward passes
  - target rasterization, tile target generation, tile score pooling, checkpoint load/save, and binary export metadata
- `scripts/train_heatmap_scout.py`
  - trains float or STE heatmap scouts on VisDrone train/val roots
  - supports synthetic smoke training when VisDrone is unavailable
  - uses dense BCEWithLogits, soft Dice, and auxiliary tile BCE
- `scripts/evaluate_heatmap_scout_recall.py`
  - evaluates top-K object-center coverage from live image scoring
  - supports `learned-heatmap`, `random`, `prior`, `heuristic`, `oracle-count`, and `oracle-greedy`
- `scripts/run_yolo_tiles.py`
  - adds `learned-heatmap` and `learned-heatmap-live`
  - routes selected original-resolution tile crops through the existing YOLO merge path
- `scripts/verify_heatmap_scout.py`
  - verifies target rasterization, float/STE shapes and backprop, deterministic top-K, checkpoint load, export, and live scoring

## Commands

Windows verification:

```powershell
python scripts\verify_heatmap_scout.py
python scripts\train_heatmap_scout.py --synthetic-images 2 --epochs 1 --batch 1 --device cpu --out runs\heatmap_scout\heatmap_float_smoke.pt
python scripts\train_heatmap_scout.py --synthetic-images 1 --epochs 1 --batch 1 --variant ste --device cpu --out runs\heatmap_scout\heatmap_ste_smoke.pt
```

Full VisDrone training:

```powershell
python scripts\train_heatmap_scout.py --variant float --train-root data\VisDrone2019-DET-train --val-root data\VisDrone2019-DET-val --epochs 20 --batch 8 --workers 4 --device cuda --out runs\heatmap_scout\heatmap_float.pt
python scripts\train_heatmap_scout.py --variant ste --init runs\heatmap_scout\heatmap_float.pt --train-root data\VisDrone2019-DET-train --val-root data\VisDrone2019-DET-val --epochs 20 --batch 8 --workers 4 --device cuda --out runs\heatmap_scout\heatmap_ste.pt
```

Selector-only recall:

```powershell
python scripts\evaluate_heatmap_scout_recall.py --checkpoint runs\heatmap_scout\heatmap_float.pt --mode learned-heatmap --top-k-values 8 12 --device cuda
python scripts\evaluate_heatmap_scout_recall.py --checkpoint runs\heatmap_scout\heatmap_ste.pt --mode learned-heatmap --top-k-values 8 12 --device cuda --out data\results_heatmap_scout_recall_learned_heatmap_ste.json
python scripts\evaluate_heatmap_scout_recall.py --mode random --random-trials 5 --top-k-values 8 12
python scripts\evaluate_heatmap_scout_recall.py --mode heuristic --top-k-values 8 12
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --mode scout --top-k-values 8 12
python scripts\evaluate_scout_recall.py --features data\tile_features\binary_xnor_hybrid_val.npz --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt --mode scout --top-k-values 8 12
```

Timed YOLO routing:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 100 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector all --crop-source original --split val --max-images 100 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector random --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt
python scripts\run_yolo_tiles.py --selector scout-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\scout_binary_xnor8_hybrid_mlp\scout_binary_xnor_hybrid.pt
python scripts\run_yolo_tiles.py --selector learned-heatmap-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\heatmap_scout\heatmap_ste.pt
python scripts\run_yolo_tiles.py --selector learned-heatmap-live --crop-source original --top-k 12 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\heatmap_scout\heatmap_ste.pt
```

## Local Smoke Results

These checks were run on the Mac without VisDrone:

| Check | Result |
|---|---|
| `verify_heatmap_scout.py` | passed |
| float synthetic train, 2 images, 1 epoch | passed |
| STE synthetic train, 1 image, 1 epoch | passed |
| selector-only learned heatmap on synthetic JSONL | passed |
| selector-only random on synthetic JSONL | passed |

Full VisDrone selector recall and YOLO latency still need to be run on the Windows RTX 4090 machine.

## Interpretation

The learned heatmap scout is the most faithful version of the original scout idea in this repo. The current binary-XNOR route proves the native C XNOR-popcount core and live tile routing path; the learned heatmap route tests whether a trained binary-style scout can select better tiles from image content before YOLO runs.

The float model should be treated as an upper-bound sanity check for the learned signal. The STE model is the project-facing binary route. If the STE model improves tile recall or YOLO recall at the same K, it is the best candidate to port deeper into native C/XNOR.

## Native C/XNOR Next Step

This pass does not claim full native C inference for the heatmap scout. The checkpoint exports signed binary-layer metadata, and the repo already has a C XNOR-popcount kernel for packed binary convolution. A full C heatmap route should only be implemented after the learned STE scout shows useful routing quality, because the measured bottleneck in the previous binary pass was native CPU XNOR/threshold cost.

Defensible claim after this pass:

```text
The project now includes both a measured native XNOR-popcount route and a trainable binary heatmap scout route. The final choice between them should be made from selector recall and timed YOLO routing results, not from architecture preference alone.
```
