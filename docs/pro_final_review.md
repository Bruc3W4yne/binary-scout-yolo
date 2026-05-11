# GPT-5.5 Pro Final Review

Source conversation:

```text
https://chatgpt.com/c/6a01716f-1a28-8332-965b-c79cc32c70cb
```

The current git-tracked repo was attached as `binary-scout-yolo-final-pass-context.zip`.

This document records the actionable feedback consumed from GPT-5.5 Pro. It is a distilled review, not a verbatim transcript.

## A. Executive Verdict

Proceed, but freeze the architecture.

Pro's verdict:

```text
Green-light the project direction.
Red-light the current final claims.
```

The repo is already a credible runnable prototype, not a blank slate. The next move should not be inventing a smarter model. It should be making the existing selective-tiling pipeline honest, measured, binary-integrated, and defensible.

The current codebase is decent to good for a school ML/systems project. It has:

- real C/XNOR-popcount core
- Python wrappers
- tile contracts
- VisDrone parsing
- scout training
- routed YOLO inference
- merge utilities
- smoke tests
- result documentation

But it is not yet ready to claim "binary scout-guided efficient UAV detection" as a finished result.

The current strongest reported scout is the spatial bitplane MLP, not the binary-XNOR path. The bitplane-only scout is worse than random at low K in the current results, and spatial features improve it substantially. That means the current best scout may be learning dataset layout priors more than image content. This is acceptable only if exposed with a spatial-prior baseline.

The current YOLO smoke numbers also do not support an efficiency claim yet:

- full-image YOLO: about 28 ms mean latency, recall 0.102
- selected-tile live scout K=8: recall 0.145, but about 174 ms mean latency
- all tiles: higher recall, higher cost

This is useful prototype evidence, but not evidence for "faster than full YOLO" or "edge real-time."

## B. Highest-Priority Changes

### 1. Fix Evidence Before Adding Model Complexity

The next Codex goal should be an evaluation-and-integration hardening pass, not a modeling exploration pass.

Do not build these yet:

- MobileNet scout
- CUDA scout kernels
- experimental full BNN detector
- broad new detector architecture

Measure the existing pipeline correctly first.

### 2. Add Missing Baselines

The result table needs these baselines before it is credible:

| Baseline | Why it matters |
|---|---|
| Random, multi-seed | Current random appears single-seed; report mean/std. |
| Spatial-prior only | Critical because the current best scout may exploit tile location. |
| Bitplane stats, no spatial | Content-only baseline. |
| Bitplane stats + spatial | Current best family. |
| Binary-XNOR, no spatial | Required to test the actual binary claim. |
| Binary-XNOR + spatial | Best chance for the binary path to be competitive. |
| Greedy oracle set-cover | Current oracle is object-count based, not a true coverage oracle. |
| All tiles | Exhaustive tiled upper-cost reference. |
| Full image YOLO | Non-tiled detector reference. |

Without spatial-prior and true greedy-oracle baselines, the project may accidentally present a weak or misleading scout result.

### 3. Integrate Binary-XNOR Into Live Routing

The binary implementation is meaningful at the systems level but under-integrated at the ML level.

Current issues:

- full train/val binary-XNOR feature extraction has not been shown as the main result
- filters appear fixed/random rather than learned
- binary-XNOR is not integrated into `scout-live`
- `binary_xnor_features` recreates the binary layer per image
- binary feature cache metadata should record filter count, seed, kernel size, threshold, and feature dimension

Conclusion:

```text
The binary path is currently a verified feature extractor, not yet a convincing binary scout.
```

Recommendation:

Keep CPU C/OpenMP for now. Reuse the binary layer object, add binary live inference, train the same simple linear/MLP scout on binary features, and measure it. Consider CUDA only after the CPU scout is correctly measured and shown to be useful.

### 4. Fix Latency Measurement

`run_yolo_tiles.py` should separate:

```text
image_load_ms
resize_preprocess_ms
gt_parse_ms
scout_ms
yolo_ms
merge_nms_ms
match_eval_ms
pipeline_ms_excl_gt
wall_ms
```

For CUDA timing, add `torch.cuda.synchronize()` before and after timed detector sections. Add warmup runs. Report mean, p50, and p95.

The headline latency should be pipeline time excluding ground-truth parsing and metric matching.

### 5. Clean Stale Documentation

Useful current docs:

- `README.md`
- `docs/current_results.md`
- `docs/research_framing.md`
- `docs/completion_audit.md`

Problem docs:

- `PROD.md` is too large and partly stale.
- `docs/binary_core_explained.md` is stale in places.

Examples of stale or dangerous references:

- `evaluate_scout_yolo.py`
- `--mode binary-xnor`
- outdated script contracts that do not match current CLIs

The repo needs one current truth source, not several partially conflicting roadmaps.

### 6. Fix Or Quarantine Broken/Stale Utilities

`scripts/benchmark_packed.py` appears to have a shape bug: helper functions return flattened float weights while the wrapper expects 3D channel/kernel weights.

Fix it or move it to legacy. Do not keep broken performance scripts in the main path.

### 7. Add Small Routing/Metrics Modules

Add small modules:

```text
src/routing.py
src/metrics.py
```

Keep them tiny. The purpose is to avoid random/oracle/prior/scout behavior diverging between tile-recall evaluation and YOLO evaluation.

### 8. Do Not Force Binary To Win

Acceptable final outcomes:

- binary-XNOR + spatial scout is best or close enough
- bitplane + spatial MLP is best
- spatial prior alone is surprisingly strong
- live scout is too slow to beat full-frame YOLO but still reduces exhaustive tiling cost

The project only fails if it pretends the binary route won when it did not.

## C. Research-Backed Claim Framing

Correct framing:

> Small-object UAV detection often benefits from tiled or sliced inference because small far-away objects occupy few pixels in a full image. SAHI-style slicing is already a known approach: split the image into slices, run a detector on each slice, and merge detections. This project does not invent slicing. It studies whether a cheap scout can select only the most promising slices and recover much of exhaustive tiling's benefit at lower tile count and measured latency.

For VisDrone, use official detection language:

- train/val/test-dev split sizes: 6,471 / 548 / 1,580 images
- COCO-style AP metrics
- AP over IoU thresholds 0.50 to 0.95
- AP50, AP75, and AR

Credible target detector table:

```text
mAP50-95
mAP50
mAP75
AP_small
AP_medium
AP_large
recall
precision
F1
```

If the detector remains COCO-pretrained instead of VisDrone-trained, do not present those numbers as final VisDrone detection performance. Present them as a class-agnostic routing smoke test.

For the binary part, cite XNOR/BNN literature only as motivation, not as proof that this implementation is efficient on a drone. XNOR-Net supports the idea of binary-weight/binary-activation convolution with XNOR/popcount efficiency. FINN and related FPGA work support the edge-hardware motivation. The actual project claim must come from measured CPU/GPU pipeline results.

Sharpened claim:

> We implement and measure a scout-guided selective tiling pipeline for UAV small-object detection. The scout uses simple bitplane or binary-XNOR-derived features to rank candidate tiles before YOLO inference. We compare against full-frame YOLO, exhaustive tiling, random tile selection, spatial-prior selection, and oracle tile selection. We report detection quality, tile coverage, selected area, tile count, and real latency.

Avoid these claims unless final measurements prove them:

```text
State of the art UAV detection
Edge deployed
Real-time drone inference
Binary neural network detector
XNOR makes YOLO faster
Beats SAHI
Binary scout is best
```

## D. First 10 Implementation Tasks

### 1. Clean Stale Current-Truth Docs

Update:

- `README.md`
- `docs/current_results.md`
- `docs/research_framing.md`
- `docs/completion_audit.md`
- `docs/binary_core_explained.md`
- `PROD.md`

Verification:

```bash
rg "evaluate_scout_yolo|--mode binary-xnor|--model bitplane-stats|scout_binary_xnor_linear|36" README.md PROD.md docs
```

Acceptance: no stale current-path command references remain unless clearly marked historical.

### 2. Fix Or Quarantine `scripts/benchmark_packed.py`

Fix the flattened-weight bug or move the script to legacy until it works.

Verification:

```bash
python scripts/benchmark_packed.py --help
python -m compileall -q scripts/benchmark_packed.py
```

If adding a tiny smoke mode:

```bash
python scripts/benchmark_packed.py --max-images 2
```

Acceptance: benchmark script either works or is no longer presented as a current performance tool.

### 3. Add `src/routing.py`

Centralize selection logic for:

```text
all
random
prior
oracle-count
oracle-greedy
scout
```

Include deterministic tie-breaking and selected-area reporting.

Verification:

```bash
python -m compileall -q src scripts
python scripts/verify_routing.py
```

Acceptance: `oracle-greedy`, `random`, and `prior` are testable without YOLO.

### 4. Upgrade `evaluate_scout_recall.py`

Add:

```text
--mode prior
--mode oracle-greedy
--random-trials N
--random-seed
selected_area_fraction
mean_selected_tiles
mean/std reporting for random
```

Verification:

```bash
python scripts/evaluate_scout_recall.py --features data/tile_features/bitplane_stats_spatial_val.npz --mode random --random-trials 5 --top-k-values 4 8 12
python scripts/evaluate_scout_recall.py --features data/tile_features/bitplane_stats_spatial_val.npz --mode prior --top-k-values 4 8 12
python scripts/evaluate_scout_recall.py --features data/tile_features/bitplane_stats_spatial_val.npz --mode oracle-greedy --top-k-values 4 8 12
```

Acceptance: routing recall output includes credible random and oracle baselines.

### 5. Add Spatial-Prior And Spatial-Only Baselines

Implement either spatial-only feature extraction or a direct prior mode from train tile statistics. Generalize `add_spatial_features.py` into `append_spatial_features.py` while keeping backward compatibility.

Acceptance: the project can prove whether learned scouts beat simple location priors.

### 6. Refactor Binary-XNOR Feature Extraction For Reuse

Create a reusable binary feature extractor object or pass a persistent `BinaryConvLayer` into the feature function. Add metadata for binary params.

Acceptance: binary feature extraction no longer reconstructs the layer unnecessarily per image, and metadata is complete.

### 7. Add Cached-Vs-Live Feature Verification

Add `scripts/verify_live_features.py` to compare feature vectors from a saved cache against live recomputation for the same images.

Verification:

```bash
python scripts/verify_live_features.py --feature-file data/tile_features_smoke/val_binary-xnor_features.npz --feature-mode binary-xnor --max-images 3
```

Acceptance: cached and live binary features match for deterministic settings.

### 8. Fix `run_yolo_tiles.py` Timing

Separate timing fields and add CUDA synchronization/warmup.

Verification:

```bash
python scripts/run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs/scout_spatial_mlp/scout_bitplane_stats_spatial.pt
python scripts/run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 2 --device cpu --checkpoint runs/scout_spatial_mlp/scout_bitplane_stats_spatial.pt
```

Acceptance: output contains per-phase timing plus aggregate mean/p50/p95, and headline latency excludes ground-truth parsing.

### 9. Add Binary-XNOR Live Scout Support

Extend `scout-live` to support:

```text
binary-xnor
binary-xnor-spatial
bitplane-stats
bitplane-stats-spatial
```

Verification:

```bash
python scripts/run_yolo_tiles.py --selector scout-live --live-feature-mode binary-xnor --scout-checkpoint data/scouts/binary_xnor_mlp/best.pt --top-k 8 --split val --max-images 2 --device cuda --out data/results/smoke_binary_live.json
```

For cached/live consistency:

```bash
python scripts/run_yolo_tiles.py --selector scout --feature-file data/tile_features_smoke/val_binary-xnor_features.npz --scout-checkpoint data/scouts/binary_xnor_mlp/best.pt --top-k 8 --split val --max-images 2 --device cuda --out data/results/smoke_binary_cached.json
```

Acceptance: cached and live selected tile IDs match on the same deterministic sample.

### 10. Run The First Credible Routing And Detector Sweep

Add `scripts/run_yolo_sweep.py` or equivalent aggregation.

Smoke verification:

```bash
python scripts/run_yolo_sweep.py --selectors full all random prior oracle-greedy scout scout-live --top-k-values 4 8 12 --split val --max-images 5 --device cuda --out-dir data/results/sweep_smoke
```

Larger verification:

```bash
python scripts/run_yolo_sweep.py --selectors full all random prior oracle-greedy scout scout-live --top-k-values 4 8 12 16 20 --split val --max-images 100 --device cuda --out-dir data/results/sweep_val100
```

Acceptance: `data/results/sweep_val100/summary.md` and `summary.json` contain detector recall, precision, F1, TP/FP/FN, selected area, tile count, and latency mean/p50/p95 for every method.
