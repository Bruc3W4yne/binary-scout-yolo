# Learned Heatmap Scout

This pass adds the missing trainable scout route that is closest to the original project idea:

```text
full image -> top-4 grayscale MSB bitplanes -> learned heatmap scout -> top-K tiles -> YOLO crops -> merged detections
```

The scout is still a router, not a replacement detector. YOLO remains responsible for final object detection. The new part is that tile selection can be learned from image pixels instead of using fixed XNOR summaries or hand-written spatial priors.

For PowerPoint alignment, use `xnor-heatmap-live`: it loads the STE heatmap
checkpoint, runs the learned binary convolution body through the native packed
XNOR-popcount kernels on CPU, ranks top-K tiles, then sends those crops to YOLO
on the GPU. `learned-heatmap-live` is the CUDA/PyTorch reference route for the
same trained scout. `binary-xnor-live` is an older feature/MLP ablation, not the
final learned heatmap scout.

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
- `src/xnor_heatmap_scout.py`
  - exports the STE scout into a native CPU route
  - uses packed XNOR-popcount for the learned binary conv layers
  - keeps YOLO on the GPU for selected tile detection
- `scripts/verify_heatmap_scout.py`
  - verifies target rasterization, float/STE shapes and backprop, deterministic top-K, checkpoint load, export, and live scoring
- `scripts/verify_xnor_heatmap_scout.py`
  - verifies native XNOR accumulator parity and full-route parity against the PyTorch STE scout

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
python scripts\run_yolo_tiles.py --selector xnor-heatmap-live --crop-source original --top-k 8 --split val --max-images 100 --device cuda --weights yolov8n.pt --checkpoint runs\heatmap_scout\heatmap_ste.pt
```

`xnor-heatmap-live` keeps selection on the CPU/native XNOR path and detector
work on the GPU. On an RTX 4090 this is slower than the CUDA/PyTorch scout, but
it is the route that validates the architecture/operation-layer split from the
presentation.

## Windows RTX 4090 Results

These were run on the first 100 VisDrone val images with `yolov8n.pt`, `--crop-source original`, and one warmup image on an NVIDIA RTX 4090. The detector is still COCO-pretrained YOLO, so this is a routing/pipeline benchmark rather than final VisDrone mAP. The benchmark code was `f716cba`; the latest code also records `git_commit` and `device_name` in new result JSON files.

Selector-only object-center recall on full VisDrone val:

| Selector | K=8 | K=12 |
|---|---:|---:|
| Random | 0.461 | 0.612 |
| Heuristic | 0.447 | 0.600 |
| Spatial scout | 0.516 | 0.635 |
| Binary XNOR8 hybrid | 0.531 | 0.641 |
| Heatmap float e5 | 0.600 | 0.725 |
| Heatmap STE e10 | 0.523 | 0.666 |

Timed YOLO routing on 100 val images:

| Method | K | Calls | Recall | Small recall | Precision | Mean ms | p95 ms | Scout mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full YOLO | - | 1.0 | 0.129 | 0.084 | 0.855 | 13.5 | 20.6 | 0.0 |
| All tiles | - | 49.0 | 0.399 | 0.355 | 0.531 | 102.7 | 113.0 | 0.0 |
| Random | 8 | 8.0 | 0.155 | 0.142 | 0.637 | 22.6 | 29.1 | 0.0 |
| Random | 12 | 12.0 | 0.215 | 0.196 | 0.622 | 29.9 | 35.6 | 0.0 |
| Spatial live | 8 | 8.0 | 0.228 | 0.210 | 0.646 | 30.7 | 41.7 | 3.0 |
| Spatial live | 12 | 12.0 | 0.267 | 0.245 | 0.615 | 38.8 | 46.4 | 3.0 |
| Binary XNOR8 live | 8 | 8.0 | 0.213 | 0.206 | 0.636 | 57.5 | 66.2 | 29.3 |
| Binary XNOR8 live | 12 | 12.0 | 0.258 | 0.244 | 0.612 | 65.5 | 74.0 | 28.9 |
| Heatmap float e5 | 8 | 8.0 | 0.250 | 0.227 | 0.650 | 32.7 | 39.2 | 7.8 |
| Heatmap float e5 | 12 | 12.0 | 0.295 | 0.263 | 0.616 | 39.7 | 44.7 | 7.8 |
| Heatmap STE e10 | 8 | 8.0 | 0.237 | 0.208 | 0.649 | 32.4 | 38.4 | 8.0 |
| Heatmap STE e10 | 12 | 12.0 | 0.289 | 0.251 | 0.621 | 39.7 | 45.4 | 8.0 |

Phase timing from the same JSON outputs:

| Method | K | YOLO mean ms | Merge mean ms | Total p50 ms | Total p95 ms |
|---|---:|---:|---:|---:|---:|
| Full YOLO | - | 5.5 | 0.0 | 12.7 | 20.6 |
| All tiles | - | 89.4 | 4.6 | 102.6 | 113.0 |
| Random | 8 | 13.6 | 0.7 | 21.2 | 29.1 |
| Random | 12 | 20.3 | 1.1 | 29.4 | 35.6 |
| Spatial live | 8 | 16.4 | 1.4 | 29.4 | 41.7 |
| Spatial live | 12 | 23.8 | 2.0 | 37.9 | 46.4 |
| Binary XNOR8 live | 8 | 16.8 | 1.4 | 56.8 | 66.2 |
| Binary XNOR8 live | 12 | 24.8 | 2.0 | 65.2 | 74.0 |
| Heatmap float e5 | 8 | 15.0 | 1.4 | 32.2 | 39.2 |
| Heatmap float e5 | 12 | 21.5 | 2.0 | 39.0 | 44.7 |
| Heatmap STE e10 | 8 | 14.6 | 1.3 | 32.0 | 38.4 |
| Heatmap STE e10 | 12 | 21.4 | 1.9 | 39.2 | 45.4 |

The strongest practical result is the learned heatmap route:

- Heatmap float K12 reaches `0.295` recall at `39.7 ms`, versus spatial K12 `0.267` at `38.8 ms`.
- Heatmap STE K12 reaches `0.289` recall at `39.7 ms`, beating both spatial K12 and binary-XNOR8 K12 recall while using far less scout time than the native CPU XNOR path.
- Heatmap STE K8 beats spatial detector recall (`0.237` vs `0.228`) but is behind heatmap float and only slightly ahead in small-object recall compared with spatial.

## Local Smoke Results

These checks were run on the Mac without VisDrone:

| Check | Result |
|---|---|
| `verify_heatmap_scout.py` | passed |
| float synthetic train, 2 images, 1 epoch | passed |
| STE synthetic train, 1 image, 1 epoch | passed |
| selector-only learned heatmap on synthetic JSONL | passed |
| selector-only random on synthetic JSONL | passed |

The same verifier suite, pytest, synthetic training, full-split heatmap training, selector-only evaluation, and timed YOLO routing were also run on the Windows RTX 4090 machine.

## Interpretation

The learned heatmap scout is the most faithful version of the original scout idea in this repo. The current binary-XNOR route proves the native C XNOR-popcount core and live tile routing path; the learned heatmap route tests whether a trained binary-style scout can select better tiles from image content before YOLO runs.

The float model is the upper-bound sanity check for the learned signal. The STE model is the project-facing binary route. The STE route now has a real result: it improves K12 detector recall over both the spatial scout and the measured 8-filter binary-XNOR live route while keeping latency close to spatial. That is the strongest evidence that the original learned binary-scout idea is worth presenting.

## Native C/XNOR Next Step

This pass does not claim full native C inference for the heatmap scout. The checkpoint exports signed binary-layer metadata, and the repo already has a C XNOR-popcount kernel for packed binary convolution. A full C heatmap route should only be implemented after the learned STE scout shows useful routing quality, because the measured bottleneck in the previous binary pass was native CPU XNOR/threshold cost.

Defensible claim after this pass:

```text
The project now includes both a measured native XNOR-popcount route and a trainable binary heatmap scout route. The final choice between them should be made from selector recall and timed YOLO routing results, not from architecture preference alone.
```
