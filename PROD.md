# PROD.md

## 0. Source of truth

This document tracks the current clean project repository:

```text
/Users/bruc3w4yne/binary-scout-yolo
```

Do not use omitted planning documents, local PDFs, slide decks, untracked files, previous drafts, or generated artifacts as project truth.

The current tracked repository contains:

```text
Makefile
requirements.txt
setup.ps1
setup.sh

src/kernel.c
src/kernel_wrapper.py
src/binary_layer.py
src/preprocess.py
src/scout.py
src/detector.py
experiments/bnn_detector/bnn_model.py

scripts/verify_kernel.py
scripts/verify_packed_kernel.py
scripts/verify_tile_contracts.py
scripts/make_tile_dataset.py
scripts/extract_tile_features.py
scripts/train_scout.py
scripts/evaluate_scout_recall.py
scripts/run_yolo_tiles.py
scripts/benchmark_yolo.py
scripts/benchmark_packed.py
experiments/bnn_detector/train_bnn.py
scripts/run_first_layer.py
scripts/legacy/*
```

The project owner has clarified the intended target machine:

```text
Windows PC
NVIDIA RTX 4090 Founders Edition
Local Windows/CUDA development first
Docker later for handoff/reproducibility
```

A previous Codex macOS build failure is a portability observation only. It is not the target runtime and should not block the project. The immediate target is the owner/student’s Windows + RTX 4090 machine.

---

## 1. Final Product Contract

### Required final product

The final product is a **full scout + YOLO detection pipeline** for UAV/drone imagery:

```text
full UAV image
    -> cheap binary/XNOR scout heatmap over candidate tiles
    -> tile scores
    -> select top-K or thresholded interesting tiles
    -> run YOLO on selected high-resolution tile crops
    -> map tile detections back to full-image coordinates
    -> NMS / merge detections
    -> evaluate accuracy, latency, selected area, tile count, and compute tradeoff
```

The scout scores/selects tiles. YOLO is the expensive detector run after selection. Any coarse heatmap is scout output or an intermediate feature map, not the final detector.

### Main project contribution

The main contribution is:

> A low-level packed binary/XNOR-popcount scout that cheaply selects candidate regions before running YOLO, with measured end-to-end detection tradeoffs.

The binary/XNOR component must be a real implementation path using the tracked C kernel, ctypes wrapper, and `BinaryConvLayer` where possible. A Python-only bit-plane/statistics scout may be used as an early baseline or smoke test, but it cannot support the final binary/XNOR claim.

### Required final comparison

The final report/demo should compare at least:

```text
1. Full-image YOLO baseline
2. Exhaustive tiled YOLO baseline
3. Random tile selection + YOLO baseline
4. Python bit-plane/statistics scout + YOLO baseline
5. Binary/XNOR scout + YOLO final pipeline
```

If time is short, the minimum final comparison is:

```text
1. Full-image YOLO
2. Random top-K tiles + YOLO
3. Binary/XNOR scout top-K tiles + YOLO
```

### Not the focus

Do not make these the project focus unless explicitly approved later:

```text
Full BNN detector training
Custom loss function research
Sobel edge detector as object detector
State-of-the-art YOLO research/training
New detector architecture research
```

YOLO fine-tuning on VisDrone is allowed, and is preferred if time permits and class-aware metrics are required. It should be treated as evaluation support for the final pipeline, not as the project novelty.

`experiments/bnn_detector/bnn_model.py` and `experiments/bnn_detector/train_bnn.py` are optional experimental extensions. They are not the primary success path.

---

## 2. Project thesis

Recommended thesis:

> We build a drone-image detection pipeline that uses a cheap packed binary/XNOR-popcount scout to identify promising image tiles, then runs YOLO only on selected high-resolution tiles. We measure whether this reduces detector workload while preserving object recall and useful detection quality.

This is drone/edge relevant because the pipeline is designed to reduce how much image area reaches the expensive detector. However, final measurements will be on a Windows PC with RTX 4090 FE. Drone feasibility should be argued honestly using measured latency, FPS, tile count, selected area, model/kernel size, and a power/compute discussion. Do not claim direct drone-board deployment unless it is actually tested on drone-class hardware.

---

## 3. Hardware and compute responsibilities

### Target machine

Immediate runnable demo target:

```text
OS: Windows
GPU: NVIDIA RTX 4090 Founders Edition
Development shell: PowerShell
Python environment: venv from setup.ps1
Dataset: local VisDrone path under data/
```

### CPU vs GPU roles

The tracked C XNOR kernels are CPU kernels:

```text
src/kernel.c
src/kernel_wrapper.py
src/binary_layer.py
```

They use C, ctypes, and optionally OpenMP. They do not run on the RTX 4090.

The RTX 4090 is for:

```text
PyTorch training of scout classifiers
optional CUDA YOLO inference experiments
possible full-image/tiled YOLO acceleration
```

Do not mix CPU and GPU latency claims without labeling them.

Example wording:

```text
Binary/XNOR scout latency was measured on CPU.
YOLO latency was measured on GPU using PyTorch CUDA.
Total pipeline latency includes CPU scout + GPU detector transfer/inference.
```

or:

```text
Both scout and YOLO were measured on CPU.
```

Pick one measurement setup and report it clearly.

---

## 4. Recommended default decisions

These defaults are chosen to keep implementation simple and testable. They can be changed later, but do not silently change them without updating config outputs and report claims.

### Locked owner decisions

These decisions came from the project owner and should be treated as locked unless explicitly changed later:

```text
Final product:
  Full binary/XNOR scout + YOLO detection pipeline.

Novelty framing:
  System-level pipeline novelty, not a new detector/loss.

Deployment framing:
  Drone/edge motivated, but measured locally on Windows RTX 4090.
  Do not claim real drone deployment without drone-class hardware tests.

Dataset:
  Use local VisDrone2019-DET-train with deterministic train/val split by default.
  If the official VisDrone2019-DET-val split is downloaded later, support it as an upgrade.

Metrics:
  Prefer class-aware VisDrone metrics if YOLO is fine-tuned and time permits.
  Use simplified class-agnostic recall/precision first if time is constrained.

YOLO weights:
  Prefer fine-tuning YOLO on VisDrone if feasible.
  Use generic/pretrained YOLO only for smoke tests or simplified metrics.

Classes:
  Store all valid VisDrone classes.
  Class-agnostic "any object" scoring is acceptable for early scout and simplified detector metrics.

Tile/top-K:
  Start with 160x160 tiles, stride 80, top_k=12 for smoke tests.
  Sweep top_k=[4,8,12,16,20] for final evaluation.

C kernel:
  Try native Windows MSYS2/MinGW first.
  Fall back to WSL2 only if native Windows becomes a time sink.

YOLO backend:
  Prefer Ultralytics/PyTorch CUDA on the RTX 4090 for final demo.

Dependencies:
  Common helper dependencies are allowed if they simplify the project and are documented.

Workflow:
  Build the smallest correct modular pieces first.
  CLI scripts are required.
  A notebook is optional later for teacher/examiner presentation.
```

### Image normalization policy

Default:

```text
Resize/stretch every image to 640x640.
```

Reason:

```text
The tracked scripts already use direct 640x640 resize.
This avoids letterbox coordinate complexity for the first full pipeline.
```

Limitation:

```text
Aspect ratio is distorted.
Report this as a limitation.
```

### Tile policy

Default smoke-test tile grid:

```text
img_size = 640
tile_size = 160
stride = 80
```

For `640x640`, tile starts are:

```text
0, 80, 160, 240, 320, 400, 480
```

This creates:

```text
7 x 7 = 49 candidate tiles per image
```

Default smoke-test tile budget:

```text
top_k = 12
```

Final evaluation should sweep:

```text
top_k = [4, 8, 12, 16, 20]
```

### Positive tile rule

Default:

```text
A tile is positive if the center of at least one valid ground-truth box lies inside the tile.
```

Reason:

```text
Small objects often have low IoU with large tiles.
Center coverage is simple and stable for scout training/recall.
```

### YOLO backend default

Early baseline:

```text
Use the tracked ONNX CPU path only if it is the fastest way to get a smoke test.
```

Preferred final demo on the RTX 4090:

```text
Use YOLO through Ultralytics/PyTorch CUDA on the RTX 4090.
Use ONNX CPU only as a fallback or early smoke-test path, and label it clearly.
```

The chosen YOLO backend must be recorded in every results JSON:

```text
"yolo_backend": "onnx_cpu" | "ultralytics_cuda" | "other"
```

The YOLO weight source must also be recorded:

```text
"yolo_weights": "yolov8n.pt" | "custom_visdrone.pt" | "other"
```

If using generic pretrained COCO YOLO weights, class-aware VisDrone metrics are not a safe final claim because class vocabularies may not match. In that case, report simplified class-agnostic recall/precision and clearly state that it is not COCO mAP and not a fully trained VisDrone detector evaluation. Class-aware mAP requires a detector trained or fine-tuned for the evaluated classes. If time permits, fine-tune YOLO on VisDrone so the final report can include stronger class-aware results.

### C-kernel route default

The final binary/XNOR claim requires a working compiled kernel:

```text
src/kernel.dll on native Windows
or
src/kernel.so under WSL2/Linux
```

Recommended local path:

```text
Native Windows with MSYS2/MinGW is strongly preferred.
WSL2 is the fallback if native Windows C/OpenMP setup becomes a time sink.
```

Python-only scout baselines can proceed before the C kernel works. The final binary/XNOR pipeline cannot.

---

## 5. Repository component roles

### `src/kernel.c`

Low-level CPU compute core.

Important functions:

```text
xnor_popcount_conv       simple single-channel XNOR-popcount convolution
xnor_packed_u8_conv      packed XNOR convolution for <=8 channels
xnor_packed_u64_conv     packed XNOR convolution for <=64 channels
xnor_multi_filter_conv   multi-filter packed XNOR convolution; key for scout
pack_channels_to_u64     packs [channels,H,W] binary planes into [H,W] uint64
threshold_i32_to_u8      thresholds int32 feature maps to uint8 binary maps
sobel_conv               edge-demo kernel; not an object detector
```

Use `xnor_multi_filter_conv` through `BinaryConvLayer` for the final binary scout.

### `src/kernel_wrapper.py`

ctypes bridge to the compiled C shared library:

```text
src/kernel.dll on Windows
src/kernel.so on Linux/WSL2
```

The wrapper is required for `BinaryConvLayer`.

### `src/binary_layer.py`

Main tracked abstraction for binary/XNOR feature extraction.

Role:

```text
Input:  [24,H,W] uint8 bit planes
Output: [n_filters,H,W] int32 score maps
Then threshold to [n_filters,H,W] uint8 binary feature maps
```

Use this in the final binary scout.

### `experiments/bnn_detector/bnn_model.py`

Experimental full BNN detector.

Do not prioritize this until the scout + YOLO pipeline is complete.

### `experiments/bnn_detector/train_bnn.py`

Experimental full BNN training script.

Useful for VisDrone parsing examples, but not the main project path.

### `scripts/benchmark_packed.py`

Important low-level benchmark.

Use it to support claims about packed XNOR-popcount vs float32 reference on equivalent convolution work.

### `scripts/benchmark_yolo.py`

Current YOLO ONNX CPU latency baseline.

It does not provide final detection quality and currently uses CPU ONNX Runtime.

### `scripts/legacy/run_pipeline.py`

Legacy tracked version runs YOLO and Sobel side-by-side. It is not the final pipeline.

The real scout + YOLO routing script is `scripts/run_yolo_tiles.py`.

---

## 6. Data contracts

## 6.1 Dataset paths

Expected dataset layout:

```text
data/VisDrone2019-DET-train/images
data/VisDrone2019-DET-train/annotations
```

Default implementation uses a deterministic train/validation split from `VisDrone2019-DET-train`, because that is the local dataset path known to exist from the tracked code. The official VisDrone2019 detection validation split exists online and can be added later if downloaded:

```text
data/VisDrone2019-DET-val/images
data/VisDrone2019-DET-val/annotations
```

If the official validation split is present, final metrics should prefer it over an internal split. Smoke tests may still use `--max-images` on the train split.

Image files:

```text
*.jpg
```

Annotation files:

```text
*.txt
```

Image and annotation stems should match:

```text
images/<stem>.jpg
annotations/<stem>.txt
```

All scripts that touch the dataset should accept:

```text
--data-root data/VisDrone2019-DET-train
```

If validation data is added, evaluation scripts should also accept:

```text
--val-root data/VisDrone2019-DET-val
```

Default:

```text
data/VisDrone2019-DET-train
```

## 6.2 VisDrone annotation parsing

Use the same interpretation as `src/preprocess.py`.

Each line:

```text
x1, y1, w, h, score, category_id, truncation, occlusion
```

Rules:

```text
score == 0:
  ignored region; skip

category_id == 0:
  ignore; skip

category_id in 1..10:
  valid object; map to class index category_id - 1

category_id == 11:
  others; skip

w <= 0 or h <= 0:
  invalid; skip
```

Internal parsed box before resizing:

```text
xyxy in original image pixels:
[x1, y1, x2, y2]
where:
x2 = x1 + w
y2 = y1 + h
```

Class index:

```text
0..9
```

## 6.3 Resize contract

Default resize:

```text
stretch original image to 640x640
```

Scale boxes:

```text
sx = 640 / original_width
sy = 640 / original_height

x1_resized = x1 * sx
y1_resized = y1 * sy
x2_resized = x2 * sx
y2_resized = y2 * sy
```

Clamp boxes to:

```text
0 <= x1 < x2 <= 640
0 <= y1 < y2 <= 640
```

Every output JSON must record:

```json
{
  "img_size": 640,
  "resize": "stretch_640"
}
```

## 6.4 Box coordinate contract

Internal boxes:

```text
format: [x1, y1, x2, y2]
space: resized 640x640 image
type: float for model/evaluation, int only for drawing/cropping
convention: x1,y1 inclusive start; x2,y2 exclusive end
```

## 6.5 Tile coordinate contract

Tile format:

```text
[x1, y1, x2, y2]
```

Tile space:

```text
resized 640x640 image
```

Tile convention:

```text
x1,y1 inclusive start
x2,y2 exclusive end
```

Default tile generation:

```text
tile_size = 160
stride = 80
include only full tiles inside 640x640
```

For default grid:

```text
starts = [0, 80, 160, 240, 320, 400, 480]
n_tiles = 49
```

## 6.6 Tile-positive contract

Default positive rule:

```text
center_in_tile
```

For each ground-truth box:

```text
cx = (box_x1 + box_x2) / 2
cy = (box_y1 + box_y2) / 2
```

A tile is positive if:

```text
tile_x1 <= cx < tile_x2
tile_y1 <= cy < tile_y2
```

A tile record should store which boxes made it positive:

```text
box_indices
classes
n_objects
```

## 6.7 Small-object rule

Project-local small-object definition:

```text
small if resized box area < 32 * 32 pixels
```

Report this as a project-local rule, not a universal VisDrone rule.

## 6.8 Selected area fraction

For selected tiles, compute union area, not simple `top_k * tile_area`.

Implementation:

```text
mask = zeros [640,640] bool
for selected tile:
    mask[y1:y2, x1:x2] = True
selected_area_fraction = mask.mean()
```

This handles overlapping tiles correctly.

---

## 7. Feature and model contracts

## 7.1 RGB bit-plane contract

Function:

```python
rgb_to_bitplanes(rgb: np.ndarray) -> np.ndarray
```

Input:

```text
rgb: [H,W,3] uint8
```

Output:

```text
planes: [24,H,W] uint8, values 0 or 1
```

Channel order:

```text
planes[0..7]    = R bits 0..7, LSB to MSB
planes[8..15]   = G bits 0..7, LSB to MSB
planes[16..23]  = B bits 0..7, LSB to MSB
```

## 7.2 Baseline feature mode: `bitplane-stats`

Purpose:

```text
C-free baseline/smoke test.
Not the final binary/XNOR claim.
```

Input:

```text
RGB image resized to 640x640
tiles
```

Features per tile:

```text
24 bit-plane means over tile
3 RGB channel means over tile
3 RGB channel std values over tile
```

Feature dimension:

```text
30
```

Model:

```text
torch.nn.Linear(30, 1)
```

Checkpoint name:

```text
runs/scout/scout_bitplane_stats.pt
```

## 7.3 Main feature mode: `binary-xnor`

Purpose:

```text
Final binary/XNOR scout feature path.
Requires compiled C kernel and BinaryConvLayer.
```

Input:

```text
RGB image resized to 640x640
bit planes [24,640,640]
tiles
```

Feature extractor:

```python
BinaryConvLayer(n_filters=64, n_ch=24, kH=3, kW=3, seed=42)
```

Feature extraction:

```text
1. Run BinaryConvLayer.forward(planes) -> scores [64,640,640] int32
2. Apply threshold -> binary maps [64,640,640] uint8
3. For each tile, compute mean activation per filter
4. Tile feature vector shape: [64]
```

Model:

```text
torch.nn.Linear(64, 1)
```

Checkpoint name:

```text
runs/scout/scout_binary_xnor_linear.pt
```

Important limitation:

```text
If BinaryConvLayer weights are fixed/random/deterministic, then only the tile classifier is learned.
Do not claim learned binary filters unless the binary filters themselves are trained.
```

Acceptable claim:

```text
A learned tile classifier using binary/XNOR-derived features.
```

Not acceptable unless implemented:

```text
A fully learned binary convolutional scout.
```

## 7.4 Optional model: `binary-xnor-mlp`

Only add after linear scout works.

Model:

```text
Linear(64, 32)
ReLU
Dropout(0.1)
Linear(32, 1)
```

Checkpoint name:

```text
runs/scout/scout_binary_xnor_mlp.pt
```

## 7.5 Random tile baseline

Purpose:

```text
Tests whether the scout is better than selecting random tiles.
```

For each image and each K:

```text
select K tiles uniformly at random from candidate tiles
use same K as scout
use fixed seed
```

Report mean and standard deviation over at least:

```text
5 random trials
```

if runtime permits.

## 7.6 Exhaustive tiled YOLO baseline

Purpose:

```text
Measures upper-bound tiled detection behavior when YOLO runs on every candidate tile.
```

This is expensive but useful for final comparison.

Default:

```text
Run YOLO on all 49 tiles for a limited validation subset if full run is too slow.
```

---

## 8. Caching policy

Feature extraction can be slower than training. Use explicit caching.

## 8.1 Tile dataset cache

Output:

```text
data/tile_dataset/train_tiles.jsonl
data/tile_dataset/val_tiles.jsonl
data/tile_dataset/summary.json
```

## 8.2 Feature cache

Add script:

```text
scripts/extract_tile_features.py
```

Outputs:

```text
data/tile_features/bitplane_stats_train.npz
data/tile_features/bitplane_stats_val.npz
data/tile_features/binary_xnor_train.npz
data/tile_features/binary_xnor_val.npz
```

Each `.npz` must contain:

```text
X          float32 [N,D]
y          float32 [N]
stems      string/object [N]
tiles      int32 [N,4]
classes    optional object/list or int summary
```

Also write metadata JSON:

```text
data/tile_features/<feature_mode>_<split>_meta.json
```

Metadata schema:

```json
{
  "feature_mode": "binary-xnor",
  "split": "val",
  "img_size": 640,
  "resize": "stretch_640",
  "tile_size": 160,
  "stride": 80,
  "feature_dim": 64,
  "n_samples": 720,
  "source_tiles": "data/tile_dataset/val_tiles.jsonl",
  "requires_c_kernel": true
}
```

Training should load `.npz` feature files instead of re-running feature extraction every epoch.

## 8.3 Detection result cache

Final YOLO evaluation outputs should be saved as JSON so metrics can be inspected without rerunning everything.

Recommended outputs:

```text
data/results_full_yolo.json
data/results_exhaustive_tiled_yolo.json
data/results_random_tiles_yolo.json
data/results_bitplane_stats_yolo.json
data/results_binary_xnor_yolo.json
```

---

## 9. Output JSON schemas

## 9.1 Tile dataset summary

Path:

```text
data/tile_dataset/summary.json
```

Schema:

```json
{
  "img_size": 640,
  "resize": "stretch_640",
  "tile_size": 160,
  "stride": 80,
  "positive_rule": "center_in_tile",
  "seed": 42,
  "val_fraction": 0.2,
  "n_images_total": 100,
  "n_images_train": 80,
  "n_images_val": 20,
  "n_tiles_train": 2880,
  "n_tiles_val": 720,
  "positive_tiles_train": 0,
  "positive_tiles_val": 0,
  "negative_tiles_train": 0,
  "negative_tiles_val": 0
}
```

## 9.2 Tile JSONL record

Paths:

```text
data/tile_dataset/train_tiles.jsonl
data/tile_dataset/val_tiles.jsonl
```

Each line:

```json
{
  "stem": "0000001_02999_d_0000005",
  "image_path": "data/VisDrone2019-DET-train/images/0000001_02999_d_0000005.jpg",
  "annotation_path": "data/VisDrone2019-DET-train/annotations/0000001_02999_d_0000005.txt",
  "split": "train",
  "resize": "stretch_640",
  "tile": [0, 0, 160, 160],
  "label": 1,
  "box_indices": [0, 3],
  "classes": [1, 4],
  "n_objects": 2
}
```

## 9.3 Scout training log

Path:

```text
runs/scout/train_log.json
```

Schema:

```json
{
  "model_type": "binary-xnor-linear",
  "feature_mode": "binary-xnor",
  "checkpoint": "runs/scout/scout_binary_xnor_linear.pt",
  "seed": 42,
  "epochs": 20,
  "batch": 1024,
  "lr": 0.001,
  "pos_weight": 3.5,
  "history": [
    {
      "epoch": 1,
      "train_bce": 0.0,
      "val_bce": 0.0
    }
  ]
}
```

## 9.4 Scout recall result

Path:

```text
data/results_scout_recall_<mode>.json
```

Schema:

```json
{
  "config": {
    "mode": "binary-xnor",
    "checkpoint": "runs/scout/scout_binary_xnor_linear.pt",
    "feature_mode": "binary-xnor",
    "img_size": 640,
    "resize": "stretch_640",
    "tile_size": 160,
    "stride": 80,
    "top_k_values": [4, 8, 12, 16, 20],
    "positive_rule": "center_in_tile"
  },
  "metrics": {
    "n_images": 20,
    "n_ground_truth_boxes": 0,
    "latency_ms": {
      "mean": 0.0,
      "p50": 0.0,
      "p95": 0.0
    },
    "by_top_k": [
      {
        "top_k": 12,
        "n_selected_tiles_mean": 12.0,
        "selected_area_fraction_mean": 0.0,
        "center_recall_all": 0.0,
        "center_recall_small": 0.0,
        "center_recall_medium_large": 0.0
      }
    ]
  },
  "images": [
    {
      "stem": "0000001_02999_d_0000005",
      "n_boxes": 5,
      "top_k": 12,
      "n_boxes_covered": 4,
      "selected_area_fraction": 0.42,
      "selected_tiles": [
        {
          "tile": [0, 0, 160, 160],
          "score": 0.91
        }
      ]
    }
  ]
}
```

## 9.5 Detection output schema

Path examples:

```text
data/results_full_yolo.json
data/results_binary_xnor_yolo.json
```

Schema:

```json
{
  "config": {
    "method": "binary_xnor_scout_yolo",
    "img_size": 640,
    "resize": "stretch_640",
    "tile_size": 160,
    "stride": 80,
    "top_k": 12,
    "scout_checkpoint": "runs/scout/scout_binary_xnor_linear.pt",
    "yolo_backend": "ultralytics_cuda",
    "conf_threshold": 0.25,
    "nms_iou_threshold": 0.5,
    "match_iou_threshold": 0.5,
    "class_agnostic_matching": true
  },
  "metrics": {
    "n_images": 20,
    "n_ground_truth_boxes": 0,
    "n_detections": 0,
    "recall_all": 0.0,
    "recall_small": 0.0,
    "precision": 0.0,
    "selected_tiles_mean": 12.0,
    "selected_area_fraction_mean": 0.0,
    "latency_ms": {
      "total_mean": 0.0,
      "total_p50": 0.0,
      "total_p95": 0.0,
      "scout_mean": 0.0,
      "yolo_mean": 0.0,
      "postprocess_mean": 0.0
    }
  },
  "images": [
    {
      "stem": "0000001_02999_d_0000005",
      "n_gt": 5,
      "n_det": 6,
      "n_gt_matched": 4,
      "selected_tiles": [
        [0, 0, 160, 160]
      ],
      "detections": [
        {
          "xyxy": [10.0, 20.0, 80.0, 100.0],
          "score": 0.72,
          "class_id": 3
        }
      ]
    }
  ]
}
```

---

## 10. YOLO tile detection contract

## 10.1 Full-image YOLO baseline

Input:

```text
resized 640x640 RGB image
```

Output:

```text
detections in full 640x640 coordinate space
```

This is the main accuracy baseline.

## 10.2 Selected-tile YOLO

For each selected tile:

```text
1. Crop tile from resized 640x640 RGB image.
2. Resize tile crop to YOLO input size, default 640x640.
3. Run YOLO.
4. Map predicted boxes back to tile coordinates.
5. Offset boxes into full-image 640x640 coordinates.
```

If tile crop is `[tx1, ty1, tx2, ty2]` and YOLO input is `640x640`, map from YOLO crop prediction `[x1, y1, x2, y2]` back to full image:

```text
scale_x = (tx2 - tx1) / 640
scale_y = (ty2 - ty1) / 640

full_x1 = tx1 + x1 * scale_x
full_y1 = ty1 + y1 * scale_y
full_x2 = tx1 + x2 * scale_x
full_y2 = ty1 + y2 * scale_y
```

Clip mapped boxes to:

```text
0..640
```

## 10.3 NMS / merge

After collecting detections from all selected tiles:

```text
Apply NMS in full-image coordinate space.
```

Defaults:

```text
conf_threshold = 0.25
nms_iou_threshold = 0.5
```

NMS can be class-aware or class-agnostic, but record the choice in JSON.

Recommended default for first implementation:

```text
class-aware NMS for final detections
class-agnostic matching for simplified recall
```

## 10.4 Ground-truth matching

For simplified detection evaluation:

```text
A ground-truth box is matched if at least one detection has IoU >= 0.5.
```

Recommended first implementation:

```text
class_agnostic_matching = true
```

Reason:

```text
The project is primarily about region selection and compute reduction.
Class-aware mAP is a later improvement.
```

Report clearly:

```text
simplified class-agnostic recall/precision, not COCO mAP
```

Optional later:

```text
class-aware matching
mAP@0.5
mAP@0.5:0.95
```

---

## 11. Coding conventions and quality constraints

### 11.1 File organization

Reusable code belongs in `src/`.

Scripts belong in `scripts/` and should be thin CLIs:

```text
parse args
call src functions
save outputs
print concise summary
```

Do not put large reusable logic only inside scripts.

### 11.2 Required new modules

Keep new files minimal.

Required:

```text
src/preprocess.py
src/scout.py
src/yolo_utils.py

scripts/check_env.py
scripts/make_tile_dataset.py
scripts/extract_tile_features.py
scripts/train_scout.py
scripts/evaluate_scout_recall.py
scripts/evaluate_scout_yolo.py
scripts/verify_binary_layer.py
```

Optional:

```text
scripts/run_scout.py
scripts/run_scout_yolo.py
```

If `evaluate_scout_yolo.py` can produce demo images, separate demo scripts are not required.

### 11.3 Path handling

Use:

```python
from pathlib import Path
```

Do not manually concatenate path strings.

Every dataset script should accept:

```text
--data-root
--img-size
--seed
--max-images
```

where relevant.

### 11.4 Determinism

Every split, random baseline, and model init must use explicit seed.

Default:

```text
seed = 42
```

Record seed in every output JSON.

### 11.5 Dependencies

Common helper dependencies are allowed if they reduce code complexity and are recorded in `requirements.txt`. Do not add heavyweight dependencies casually.

Allowed from current requirements:

```text
torch
torchvision
ultralytics
Pillow
numpy
scipy
onnxruntime
```

Allowed additions if useful:

```text
tqdm              progress bars
matplotlib        plots/demo images if PIL alone is awkward
opencv-python     optional image drawing/NMS helpers only if needed
pandas            optional final summary tables only if needed
```

Prefer:

```text
PIL for image I/O and drawing
NumPy for metrics
PyTorch for scout training
```

CLI scripts are required for every phase. A notebook may be added later as a presentation artifact for the teacher/examiner, but it must call the same `src/` logic and must not become the only runnable workflow.

Do not require `sklearn` for AUC unless it is added deliberately and documented.

### 11.6 Dataclasses

Use small dataclasses where helpful:

```python
@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int

@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class ScoutConfig:
    img_size: int = 640
    tile_size: int = 160
    stride: int = 80
    top_k: int = 12
```

Avoid clever abstractions. Keep objects simple and serializable.

### 11.7 Error handling

Scripts should fail early with clear messages if:

```text
dataset path missing
no images found
annotations missing
C kernel required but kernel.dll/kernel.so missing
checkpoint missing
YOLO weights missing
CUDA requested but unavailable
```

### 11.8 Testing and verification

Every implementation phase must have one command that verifies it.

Do not proceed to final claims until the relevant verification command passes.

### 11.9 Performance reporting

Always report:

```text
hardware
OS
CPU/GPU backend
compiler route for C kernel
number of images
warmup count if used
mean latency
p50 latency
p95 latency
FPS
```

### 11.10 No overclaiming in code comments or outputs

Avoid print statements like:

```text
"fast detector complete"
"accuracy preserved"
"drone-ready"
```

Use measured language:

```text
"selected_area_fraction=..."
"recall_all=..."
"mean_latency_ms=..."
```

---

## 12. Implementation roadmap

Implementation strategy:

```text
Build as much of the full pipeline as possible, but only by completing modular pieces in order.
Each phase must produce a usable artifact and a verification command before the next layer depends on it.
Prefer a small correct vertical slice over a large unverified script.
```

## Phase 0 — Windows + RTX 4090 environment verification

Goal:

```text
Confirm local Windows/CUDA environment and dataset paths.
```

PowerShell commands from repo root:

```powershell
.\setup.ps1 cu121
.\venv\Scripts\Activate.ps1
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
Test-Path data\VisDrone2019-DET-train\images
Test-Path data\VisDrone2019-DET-train\annotations
```

Add optional helper:

```text
scripts/check_env.py
```

Verification command:

```powershell
python scripts\check_env.py
```

Done when:

```text
venv activates
PyTorch imports
torch.cuda.is_available() is true
RTX 4090 is printed
VisDrone paths exist
```

---

## Phase 1 — Shared preprocessing and tile dataset

Goal:

```text
Create reproducible tile labels from VisDrone annotations.
```

Add:

```text
src/preprocess.py
scripts/make_tile_dataset.py
```

Required functions in `src/preprocess.py`:

```text
load_rgb_resized(path, img_size=640, resize="stretch")
rgb_to_bitplanes(rgb)
parse_visdrone_annotations(path)
scale_boxes_to_resized(boxes, orig_size, img_size=640, resize="stretch")
make_tiles(img_size=640, tile_size=160, stride=80)
label_tiles(tiles, boxes, rule="center_in_tile")
tile_union_area_fraction(tiles, img_size=640)
box_iou_xyxy(a, b)
```

Verification command:

```powershell
python scripts\make_tile_dataset.py --max-images 50 --tile-size 160 --stride 80 --val-frac 0.2 --seed 42
```

Expected outputs:

```text
data/tile_dataset/train_tiles.jsonl
data/tile_dataset/val_tiles.jsonl
data/tile_dataset/summary.json
```

Done when:

```text
summary.json reports image counts, tile counts, positive tiles, and negative tiles
```

---

## Phase 2 — C-free baseline features and scout training

Goal:

```text
Prove the tile ML pipeline works before depending on C kernel setup.
```

Add:

```text
scripts/extract_tile_features.py
scripts/train_scout.py
scripts/evaluate_scout_recall.py
```

Feature extraction command:

```powershell
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split train
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split val
```

Training command:

```powershell
python scripts\train_scout.py --model bitplane-stats --epochs 2 --batch 1024 --lr 0.001 --seed 42
```

Scout recall command:

```powershell
python scripts\evaluate_scout_recall.py --mode bitplane-stats --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k-values 4 8 12 16 20
```

Done when:

```text
bitplane-stats features are cached
a scout checkpoint is saved
scout recall JSON is produced
```

This baseline is not the final binary/XNOR claim.

---

## Phase 3 — C kernel route and binary layer verification

Goal:

```text
Enable real low-level binary/XNOR feature extraction.
```

Choose one route:

```text
A. Native Windows MSYS2/MinGW
B. WSL2 Ubuntu/GCC/OpenMP
```

Native Windows setup follows the tracked `setup.ps1` comments:

```text
Install MSYS2.
From MSYS2 MinGW64 shell:
  pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openmp make

Add:
  C:\msys64\mingw64\bin

to Windows PATH.
```

Native Windows verification:

```powershell
make
python scripts\verify_kernel.py
python scripts\verify_binary_layer.py
```

WSL2 verification:

```bash
make
python scripts/verify_kernel.py
python scripts/verify_binary_layer.py
```

Add:

```text
scripts/verify_binary_layer.py
```

This script must compare `BinaryConvLayer.forward()` against a slow NumPy reference and compare thresholding against NumPy.

Done when:

```text
src/kernel.dll or src/kernel.so exists
verify_kernel.py passes
verify_binary_layer.py passes
```

---

## Phase 4 — Binary/XNOR feature extraction and learned scout

Goal:

```text
Train the main scout using binary/XNOR-derived features.
```

Feature extraction:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val
```

Training:

```powershell
python scripts\train_scout.py --model binary-xnor-linear --epochs 20 --batch 1024 --lr 0.001 --seed 42
```

Evaluation:

```powershell
python scripts\evaluate_scout_recall.py --mode binary-xnor --checkpoint runs\scout\scout_binary_xnor_linear.pt --split val --top-k-values 4 8 12 16 20
```

Done when:

```text
binary_xnor_train.npz and binary_xnor_val.npz exist
binary-XNOR scout checkpoint exists
binary-XNOR scout recall is measured
binary-XNOR scout beats random top-K on recall/area tradeoff
```

---

## Phase 5 — YOLO utility and full-image baseline

Goal:

```text
Run YOLO on full images and produce detections in 640x640 coordinates.
```

Add:

```text
src/yolo_utils.py
```

Functions:

```text
load_yolo_model(...)
run_yolo_on_image(...)
nms_xyxy(...)
match_detections_to_ground_truth(...)
```

Implement or wrap YOLO post-processing carefully.

Full-image baseline command:

```powershell
python scripts\evaluate_scout_yolo.py --method full-yolo --split val --max-images 20
```

Expected output:

```text
data/results_full_yolo.json
```

Done when:

```text
full-image YOLO produces detections
detections are matched to ground truth
latency and recall/precision are reported
```

---

## Phase 6 — Selected-tile YOLO pipeline

Goal:

```text
Run YOLO only on selected tiles from each scout/baseline and merge detections.
```

Required methods:

```text
random-tiles-yolo
bitplane-stats-yolo
binary-xnor-yolo
```

Optional but useful:

```text
exhaustive-tiled-yolo
```

Commands:

```powershell
python scripts\evaluate_scout_yolo.py --method random-tiles-yolo --split val --top-k 12 --max-images 20

python scripts\evaluate_scout_yolo.py --method bitplane-stats-yolo --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k 12 --max-images 20

python scripts\evaluate_scout_yolo.py --method binary-xnor-yolo --checkpoint runs\scout\scout_binary_xnor_linear.pt --split val --top-k 12 --max-images 20
```

Optional:

```powershell
python scripts\evaluate_scout_yolo.py --method exhaustive-tiled-yolo --split val --max-images 20
```

Expected outputs:

```text
data/results_random_tiles_yolo.json
data/results_bitplane_stats_yolo.json
data/results_binary_xnor_yolo.json
data/results_exhaustive_tiled_yolo.json
```

Done when:

```text
selected tile crops run through YOLO
detections are mapped back to full-image coordinates
NMS/merge is applied
simplified detection metrics are reported
latency breakdown includes scout, YOLO, and postprocess time
```

---

## Phase 7 — Final benchmark table and demo artifacts

Goal:

```text
Produce final evidence for presentation/report.
```

Required table columns:

```text
method
top_k
selected_tiles_mean
selected_area_fraction_mean
recall_all
recall_small
precision
total_latency_ms_mean
total_latency_ms_p50
total_latency_ms_p95
fps
scout_latency_ms_mean
yolo_latency_ms_mean
notes
```

Required visual artifacts:

```text
data/demo/scout_heatmap.png
data/demo/selected_tiles.png
data/demo/final_detections.png
```

Required result files:

```text
data/results_full_yolo.json
data/results_random_tiles_yolo.json
data/results_bitplane_stats_yolo.json
data/results_binary_xnor_yolo.json
```

Done when:

```text
The full binary/XNOR scout + YOLO pipeline runs on the Windows RTX 4090 machine
and produces merged detections with metrics.
```

---

## 13. Concrete Codex task list

### Task 1 — Windows environment check

Files touched:

```text
scripts/check_env.py
```

Expected output:

```text
Print Python version, torch version, CUDA availability, CUDA device name,
dataset path existence, and whether src/kernel.dll/src/kernel.so exists.
```

Verification:

```powershell
.\setup.ps1 cu121
.\venv\Scripts\Activate.ps1
python scripts\check_env.py
```

Risk:

```text
Low
```

---

### Task 2 — Preprocessing module

Files touched:

```text
src/preprocess.py
```

Expected output:

```text
Shared image loading, resizing, bit-plane conversion, annotation parsing,
box scaling, tile generation, tile labeling, IoU, and selected-area helpers.
```

Verification:

```powershell
python -m compileall src
```

Risk:

```text
Medium
```

Reason:

```text
Coordinate bugs are common.
```

---

### Task 3 — Tile dataset generator

Files touched:

```text
scripts/make_tile_dataset.py
src/preprocess.py
```

Expected output:

```text
data/tile_dataset/train_tiles.jsonl
data/tile_dataset/val_tiles.jsonl
data/tile_dataset/summary.json
```

Verification:

```powershell
python scripts\make_tile_dataset.py --max-images 50 --tile-size 160 --stride 80 --val-frac 0.2 --seed 42
```

Risk:

```text
Medium
```

---

### Task 4 — Binary layer verification

Files touched:

```text
scripts/verify_binary_layer.py
```

Expected output:

```text
BinaryConvLayer.forward() matches slow NumPy reference.
threshold_i32_to_u8 wrapper matches NumPy thresholding.
```

Verification:

```powershell
python scripts\verify_binary_layer.py
```

Risk:

```text
Low after C kernel builds
High before C kernel route is configured
```

---

### Task 5 — Feature extraction script

Files touched:

```text
scripts/extract_tile_features.py
src/preprocess.py
src/scout.py
```

Expected output:

```text
data/tile_features/bitplane_stats_train.npz
data/tile_features/bitplane_stats_val.npz
data/tile_features/binary_xnor_train.npz
data/tile_features/binary_xnor_val.npz
```

Verification:

```powershell
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split train
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split val
```

Binary verification after C kernel works:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split train
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val
```

Risk:

```text
Medium
```

---

### Task 6 — Scout module and training

Files touched:

```text
src/scout.py
scripts/train_scout.py
```

Expected output:

```text
runs/scout/scout_bitplane_stats.pt
runs/scout/scout_binary_xnor_linear.pt
runs/scout/train_log.json
```

Verification:

```powershell
python scripts\train_scout.py --model bitplane-stats --epochs 2 --batch 1024 --lr 0.001 --seed 42
```

Binary verification:

```powershell
python scripts\train_scout.py --model binary-xnor-linear --epochs 2 --batch 1024 --lr 0.001 --seed 42
```

Risk:

```text
Medium
```

Reason:

```text
Class imbalance may require pos_weight.
```

Required loss:

```text
BCEWithLogitsLoss(pos_weight=n_negative / max(n_positive, 1))
```

---

### Task 7 — Scout recall evaluator

Files touched:

```text
scripts/evaluate_scout_recall.py
src/scout.py
src/preprocess.py
```

Expected output:

```text
data/results_scout_recall_bitplane_stats.json
data/results_scout_recall_binary_xnor.json
data/results_scout_recall_random.json
```

Verification:

```powershell
python scripts\evaluate_scout_recall.py --mode random --split val --top-k-values 4 8 12 16 20

python scripts\evaluate_scout_recall.py --mode bitplane-stats --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k-values 4 8 12 16 20

python scripts\evaluate_scout_recall.py --mode binary-xnor --checkpoint runs\scout\scout_binary_xnor_linear.pt --split val --top-k-values 4 8 12 16 20
```

Risk:

```text
Medium
```

---

### Task 8 — YOLO utilities

Files touched:

```text
src/yolo_utils.py
```

Expected output:

```text
Reusable YOLO loading/inference helpers, NMS, box mapping, and GT matching.
```

Verification:

```powershell
python -m compileall src
```

Risk:

```text
High
```

Reason:

```text
YOLO output format, NMS, coordinate mapping, and matching are bug-prone.
```

---

### Task 9 — Full scout + YOLO evaluator

Files touched:

```text
scripts/evaluate_scout_yolo.py
src/yolo_utils.py
src/scout.py
src/preprocess.py
```

Expected output:

```text
data/results_full_yolo.json
data/results_random_tiles_yolo.json
data/results_bitplane_stats_yolo.json
data/results_binary_xnor_yolo.json
```

Verification:

```powershell
python scripts\evaluate_scout_yolo.py --method full-yolo --split val --max-images 5

python scripts\evaluate_scout_yolo.py --method random-tiles-yolo --split val --top-k 12 --max-images 5

python scripts\evaluate_scout_yolo.py --method bitplane-stats-yolo --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k 12 --max-images 5

python scripts\evaluate_scout_yolo.py --method binary-xnor-yolo --checkpoint runs\scout\scout_binary_xnor_linear.pt --split val --top-k 12 --max-images 5
```

Risk:

```text
High
```

---

### Task 10 — Final demo images and summary table

Files touched:

```text
scripts/evaluate_scout_yolo.py
or
scripts/run_scout_yolo.py
```

Expected output:

```text
data/demo/scout_heatmap.png
data/demo/selected_tiles.png
data/demo/final_detections.png
data/final_summary.json
```

Verification:

```powershell
python scripts\evaluate_scout_yolo.py --method binary-xnor-yolo --checkpoint runs\scout\scout_binary_xnor_linear.pt --split val --top-k 12 --max-images 5 --save-demo
```

Risk:

```text
Medium
```

---

## 14. Claim ladder and overclaiming guardrails

### Milestone 0 — Environment verified

Can claim:

```text
The project runs locally on the Windows RTX 4090 development machine.
PyTorch sees CUDA.
Dataset paths are valid.
```

Cannot claim:

```text
Binary kernel works.
Scout works.
Detection pipeline works.
```

### Milestone 1 — Tile dataset generated

Can claim:

```text
VisDrone annotations were converted into reproducible tile-occupancy labels.
```

Cannot claim:

```text
Scout learned useful objectness.
```

### Milestone 2 — C-free scout baseline trained

Can claim:

```text
A simple learned tile-occupancy baseline was trained and evaluated.
```

Cannot claim:

```text
Binary/XNOR efficiency.
```

### Milestone 3 — C kernel and BinaryConvLayer verified

Can claim:

```text
The tracked C XNOR-popcount implementation builds on the chosen route.
BinaryConvLayer matches a reference implementation on tested inputs.
```

Cannot claim:

```text
End-to-end speedup.
```

### Milestone 4 — Binary/XNOR scout recall measured

Can claim:

```text
A learned tile classifier using binary/XNOR-derived features selects tiles
with measured recall and selected-area fraction.
```

Cannot claim:

```text
Final detection quality is preserved.
```

### Milestone 5 — Scout + YOLO detections merged

Can claim:

```text
The full pipeline runs:
binary/XNOR scout -> selected tiles -> YOLO -> mapped boxes -> NMS.
```

Cannot claim:

```text
It improves speed/accuracy until measured against baselines.
```

### Milestone 6 — Final benchmark complete

Can claim, if supported by results:

```text
Compared with full-image YOLO and tile baselines, the binary/XNOR scout + YOLO
pipeline achieves measured tradeoffs in recall, precision, latency, tile count,
and selected image area on the tested Windows RTX 4090 setup.
```

Still do not claim:

```text
State-of-the-art.
Drone-board deployment.
General hardware speedup.
Full BNN detector.
Learned binary convolution filters unless implemented.
```

---

## 15. Definition of done for next group meeting

The next group meeting should show progress toward the full pipeline, but should not require every final benchmark to be complete.

### Required environment proof

Run in PowerShell:

```powershell
.\setup.ps1 cu121
.\venv\Scripts\Activate.ps1
python scripts\check_env.py
```

Must show:

```text
torch imports
cuda_available = true
RTX 4090 device name
VisDrone image path exists
VisDrone annotation path exists
```

### Required data milestone

```powershell
python scripts\make_tile_dataset.py --max-images 50 --tile-size 160 --stride 80 --val-frac 0.2 --seed 42
```

Required artifact:

```text
data/tile_dataset/summary.json
```

### Required C-free scout milestone

```powershell
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split train
python scripts\extract_tile_features.py --feature-mode bitplane-stats --split val
python scripts\train_scout.py --model bitplane-stats --epochs 2 --batch 1024 --lr 0.001 --seed 42
python scripts\evaluate_scout_recall.py --mode bitplane-stats --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k-values 4 8 12 16 20
```

Required artifacts:

```text
runs/scout/scout_bitplane_stats.pt
runs/scout/train_log.json
data/results_scout_recall_bitplane_stats.json
```

### Required C-kernel decision

By the meeting, document one:

```text
A. Native Windows MSYS2/MinGW kernel build is working.
B. WSL2 kernel build is working.
C. C kernel is not working yet, but exact next setup step is known.
```

If A or B is working, also show:

```powershell
python scripts\verify_kernel.py
python scripts\verify_binary_layer.py
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val
```

### Required YOLO smoke test

At minimum:

```powershell
python scripts\evaluate_scout_yolo.py --method full-yolo --split val --max-images 5
```

If selected-tile code is ready:

```powershell
python scripts\evaluate_scout_yolo.py --method bitplane-stats-yolo --checkpoint runs\scout\scout_bitplane_stats.pt --split val --top-k 12 --max-images 5
```

### Questions the group should be able to answer

```text
1. Does PyTorch see the RTX 4090?
2. Is VisDrone available at the expected path?
3. How many positive/negative tiles were generated?
4. Does a learned tile scout train and produce recall metrics?
5. Which C-kernel route is being used: native Windows or WSL2?
6. If the C kernel works, does BinaryConvLayer match a reference?
7. Does full-image YOLO produce detections and metrics?
8. Does selected-tile YOLO map boxes back correctly?
9. Is the binary/XNOR scout recall good enough to use in the final YOLO pipeline?
```

---

## 16. Final report structure

Recommended final report structure:

```text
1. Problem
   UAV imagery has small objects and high-resolution scenes.
   Running a heavy detector everywhere can waste compute.

2. Product idea
   Use a cheap binary/XNOR scout to select candidate tiles.
   Run YOLO only on selected tiles.

3. Implementation
   Preprocessing and tile labels.
   Binary/XNOR C kernel and Python wrapper.
   BinaryConvLayer feature extraction.
   Learned tile classifier.
   YOLO tile detector and box merge.

4. Baselines
   Full-image YOLO.
   Exhaustive tiled YOLO.
   Random tile selection.
   Bit-plane statistics scout.
   Binary/XNOR scout.

5. Metrics
   Recall all.
   Recall small.
   Precision.
   Selected area fraction.
   Tile count.
   Latency/FPS.
   Model/kernel size.
   Hardware/backend description.

6. Results
   Tables with measured values.
   Demo images.
   Failure cases.

7. Drone/edge discussion
   Honest discussion using PC measurements.
   Estimate implications for edge devices.
   Do not claim deployment unless tested.

8. Limitations
   Stretch resize.
   Simplified matching if not mAP.
   CPU XNOR kernel.
   RTX 4090 not drone hardware.
   Fixed/random binary filters unless trained.

9. Conclusion
   State what worked, what failed, and next steps.

10. Optional notebook appendix
   A clean walkthrough notebook for teacher/examiner review.
   It should call the CLI/src pipeline outputs, not replace them.
```

---

## 17. Final one-liner

Use this in README/slides:

> We built a full UAV image detection pipeline where a CPU packed binary/XNOR-popcount scout selects promising tiles, YOLO detects objects only on those selected crops, and the merged detections are evaluated against full-image YOLO using recall, selected area, latency, FPS, and compute tradeoff metrics.
