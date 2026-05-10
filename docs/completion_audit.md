# Completion Audit

Generated after the first full implementation/hardening pass.

This is the handoff truth for what is currently runnable. The project is not a finished research paper yet, but the repo now contains a working CLI pipeline that can be built, tested, and benchmarked in pieces.

## Status

The current implementation is a runnable end-to-end prototype:

```text
VisDrone image
-> deterministic 49-tile grid
-> tile occupancy labels
-> bitplane/spatial or binary-XNOR scout features
-> trained scout scores
-> top-K tile routing
-> YOLO on selected crops
-> coordinate merge/NMS
-> JSON metrics and heatmap output
```

Best current claim:

```text
A scout-guided selected-tile pipeline is implemented and measurable. On the current full-val tile-label evaluation, the tiny spatial MLP scout covers 0.516 of object centers at K=8, compared with 0.447 for random K=8 and 0.763 for oracle K=8.
```

Do not claim that the system beats YOLO overall, that it is already drone-ready, or that the binary-XNOR path is already the fastest live scout.

## Phase Gate Audit

| Gate | Status | Evidence | Caveat |
|---|---|---|---|
| Environment bootstrap | Pass | Windows SSH works as `desktop-jv3pk9g\bruc3w4yne`; Python 3.10.14; PyTorch `2.11.0+cu126`; CUDA available on `NVIDIA GeForce RTX 4090`; `mingw32-make clean && mingw32-make` produced `src\kernel.dll`. | Windows checkout is a synced working tree, not a clean git checkout, because noninteractive git auth prompted. Mac/GitHub remain source of truth. |
| Binary core | Pass | `python scripts\verify_packed_kernel.py --include-nonbinary` passed `single-channel-1x1`, `u8-sized-3x3`, `rgb-bitplanes-3x3`, `u64-full-3x3`, `threshold_i32_to_u8`, `shape_guard_errors`, and nonbinary 0/255 plane handling. | Current kernel is CPU/OpenMP C through ctypes, not CUDA. That is fine for the binary-scout claim, but not a 4090 acceleration claim. |
| Data contracts | Pass | `python scripts\verify_tile_contracts.py` passed; `python scripts\verify_tile_dataset.py` verified 6471 train images, 548 val images, 317079 train tile records, 26852 val tile records, and disjoint train/val splits. | Uses stretch-to-640 resize. Letterbox support is not implemented. |
| Scout pipeline | Pass | Feature caches exist for `bitplane-stats`, spatial bitplane features, and a 25-image binary-XNOR smoke cache. `python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_spatial_val.npz --expect-feature-dim 38 --expect-feature-mode bitplane-stats-spatial` passed. `evaluate_scout_recall.py` with `runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt` reached K=8 object recall `0.516`. | Full train/val binary-XNOR scout training has not been run yet. Current best scout is bitplane/spatial MLP, not binary-XNOR. |
| Detector/router | Pass | `scripts\run_yolo_tiles.py` supports `full`, `all`, `random`, `oracle`, `scout`, and `scout-live`. `python scripts\run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --weights yolov8n.pt` passed and wrote JSON metrics. | The YOLO smoke benchmark uses COCO-pretrained `yolov8n.pt`, so class-aware VisDrone mAP is not final. |
| Benchmarks/smoke tests | Pass for smoke, partial for final | `docs/current_results.md` records K sweeps, cached and live scout routing, full-image YOLO, random/oracle/scout K=8, and all-tile YOLO on a 10-image subset. | Full all-tile validation and proper mAP are intentionally not run yet because they are heavier. Commands are available, but final paper-quality results remain future work. |
| Code-quality pass | Pass for first handoff | Legacy demos moved under `scripts/legacy/`; full BNN detector moved under `experiments/bnn_detector/`; feature cache schema compacted; binary wrapper guards added; README updated. | Script imports still use local `sys.path` insertion instead of editable package installation. That is acceptable for a school CLI repo but should be cleaned before packaging. |
| Handoff docs | Pass with warning | `README.md`, `docs/current_results.md`, `docs/research_framing.md`, and this audit describe the runnable path, current claims, commands, and caveats. | `PROD.md` is still a long design document with historical roadmap sections. Use `README.md` and this audit for exact current commands. |
| GPT-5.5 Pro orchestration | Blocked in final pass | Earlier architecture work used the ChatGPT web workflow, but the final audit could not create a fresh GPT-5.5 Pro pass: Browser currently reports no available in-app backend, and Computer Use cannot capture the Zen window (`cgWindowNotFound`). | Retry when Browser/Computer Use can see ChatGPT again. Do not block the code handoff on this. |

## Current Verification Commands

On Windows PowerShell from the repo root:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
mingw32-make clean
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
python scripts\verify_detector_utils.py
python scripts\verify_tile_dataset.py
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_spatial_val.npz --expect-feature-dim 38 --expect-feature-mode bitplane-stats-spatial
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --top-k-values 8
python scripts\run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --weights yolov8n.pt
```

## Best Next Work

1. Optimize live scout feature extraction. It is currently the largest latency cost in the honest `scout-live` path.
2. Run selected YOLO K sweeps for K=4, 8, 12, 16, and 20, not only tile-label recall sweeps.
3. Run a larger all-tile YOLO subset to strengthen the SAHI-style reference point.
4. Decide whether the report needs class-aware VisDrone mAP. If yes, fine-tune or use VisDrone-compatible YOLO weights.
5. Train and evaluate a full binary-XNOR scout if the final presentation needs the binary implementation to be more than a verified feature-smoke path.

## Presentation Framing

The novelty is the complete measured pipeline, not a new detector:

```text
SAHI-style tiling is known.
YOLO is known.
BNN/XNOR-popcount is known.
The project contribution is combining them into a simple scout-routing system and measuring whether cheap tile selection can recover useful small-object coverage while reducing detector calls.
```

That framing is honest, feasible, and interesting enough for a strong school project if the final report is careful about what is measured versus what is proposed.
