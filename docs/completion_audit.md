# Completion Audit

Generated May 11, 2026 after the Windows-first final verification pass.

This is the handoff truth for the current repo. The project is not a finished research paper, but it is a runnable CLI implementation of the selective-tiling pipeline with documented Windows/CUDA smoke evidence and clear limits.

## Final Status

The implemented path is:

```text
VisDrone image
-> deterministic 49-tile grid
-> tile occupancy labels
-> scout features or live scout scoring
-> top-K tile routing
-> YOLO on selected crops
-> coordinate merge/NMS
-> JSON recall, selection, and timing metrics
```

The strongest current claim is:

```text
On full-val tile-label evaluation, the spatial bitplane MLP scout covers 0.516 of object centers at K=8. That beats 5-trial random K=8 at 0.452 +/- 0.004, train-split spatial prior at 0.469, and content heuristic at 0.447, while greedy-oracle K=8 shows headroom at 0.953.
```

Do not claim that this beats YOLO overall, is state of the art, is already drone-real-time, or that the binary-XNOR path is the fastest live scout. The binary path is currently a verified C/OpenMP XNOR-popcount scout feature implementation.

## Acceptance Evidence

| Requirement | Status | Evidence |
|---|---|---|
| Windows RTX 4090 target is authoritative | Pass | Final verification ran by SSH on the Windows RTX 4090 machine, not on the Mac. Mac checks were editing/smoke only. |
| Clean exported repo verification | Pass | A clean Windows tree was exported for the final fast verifier bundle. The heavier data/feature/YOLO evidence came from `3e633d3`; the final cleanup changed docs/setup only. |
| Native binary core builds on Windows | Pass | `mingw32-make clean`, `mingw32-make`, and `mingw32-make test` passed. The C test suite reported `55 / 55 passed`. |
| Packed C/Python wrapper is validated | Pass | `python scripts\verify_packed_kernel.py --include-nonbinary` passed all parity, threshold, shape-guard, and nonbinary 0/255 plane checks. |
| Python contracts are testable | Pass | `python -m compileall -q src scripts tests` passed. `python -m pytest -q` passed with `4 passed in 2.29s`. |
| Dataset contracts use official VisDrone train/val | Pass | `verify_visdrone_dataset.py --split-mode official` checked train and val samples. `verify_tile_dataset.py` verified 6471 train images, 548 val images, 317079 train tile records, 26852 val tile records, and disjoint splits. |
| Feature cache contracts are checked | Pass | `verify_tile_features.py` passed for `data\tile_features\bitplane_stats_spatial_val.npz`, expecting 38 features and `bitplane-stats-spatial`. |
| Tile-label recall baselines are credible | Pass | Full-val K=8 recall: scout `0.516`, random `0.452 +/- 0.004`, prior `0.469`, oracle-greedy `0.953`. |
| YOLO router selectors execute | Pass | One-image CUDA smoke from commit `3e633d3` ran `full`, `all`, `random`, `prior`, `heuristic`, `oracle-greedy`, cached `scout`, and `scout-live`, each writing JSON. |
| Timing separates pipeline phases | Pass | `run_yolo_tiles.py` reports image load, resize/preprocess, ground-truth parse, scout, YOLO, merge/NMS, match/eval, pipeline excluding GT, and wall time summaries. |
| Binary-XNOR benchmark is available | Pass | `benchmark_xnor_kernel.py --synthetic --max-images 5 --runs 10 --warmup 2` wrote JSON and measured C XNOR paths against float references. |
| Claim audit exists | Pass | `docs/final_project_claims.md` separates thesis, defensible claim, binary-XNOR claim, novelty framing, and claims to avoid. |
| Long benchmark commands are ready | Pass | `docs/evaluation_protocol.md` lists tile-label recall, YOLO smoke, and longer K-sweep commands without adding a new experiment framework. |
| Stale artifacts are removed | Pass | Legacy planning docs, stale `PROD.md`, stale legacy scripts, out-of-scope full-BNN detector experiment, the consumed `docs/next_goal.md`, and the internal review transcript were removed. |
| Final external review gate | Pass after fixes | The final review found no core source-code blocker. It rejected command-truth cleanup issues in `README.md`, `setup.ps1`, `docs/current_results.md`, and this audit; those issues were patched before handoff. |

## Windows Commands Run

Core verification on the clean exported final Windows tree:

```powershell
python -m compileall -q src scripts tests
python -m pytest -q
mingw32-make clean
mingw32-make
mingw32-make test
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
python scripts\verify_detector_utils.py
python scripts\verify_tile_grid.py
python scripts\verify_routing.py
```

Dataset, feature, recall, and kernel evidence from clean exported Windows commit `3e633d3`:

```powershell
python scripts\verify_visdrone_dataset.py --split-mode official
python scripts\verify_tile_dataset.py
python scripts\verify_tile_features.py --feature-file data\tile_features\bitplane_stats_spatial_val.npz --expect-feature-dim 38 --expect-feature-mode bitplane-stats-spatial
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --top-k-values 8 --out data\results\smoke_recall_scout_3e633d3.json
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode random --random-trials 5 --top-k-values 8 --out data\results\smoke_recall_random_3e633d3.json
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode prior --prior-features data\tile_features\bitplane_stats_spatial_train.npz --top-k-values 8 --out data\results\smoke_recall_prior_3e633d3.json
python scripts\evaluate_scout_recall.py --features data\tile_features\bitplane_stats_spatial_val.npz --mode oracle-greedy --top-k-values 8 --out data\results\smoke_recall_oracle_greedy_3e633d3.json
python scripts\benchmark_xnor_kernel.py --synthetic --max-images 5 --runs 10 --warmup 2 --out data\results\smoke_xnor_kernel_3e633d3.json
```

One-image CUDA YOLO selector smoke:

```powershell
python scripts\run_yolo_tiles.py --selector full --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_full_3e633d3.json
python scripts\run_yolo_tiles.py --selector all --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_all_3e633d3.json
python scripts\run_yolo_tiles.py --selector random --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_random_3e633d3.json
python scripts\run_yolo_tiles.py --selector prior --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_prior_3e633d3.json
python scripts\run_yolo_tiles.py --selector heuristic --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_heuristic_3e633d3.json
python scripts\run_yolo_tiles.py --selector oracle-greedy --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --out data\results\smoke_yolo_oracle_greedy_3e633d3.json
python scripts\run_yolo_tiles.py --selector scout --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --features data\tile_features\bitplane_stats_spatial_val.npz --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --out data\results\smoke_yolo_scout_3e633d3.json
python scripts\run_yolo_tiles.py --selector scout-live --top-k 8 --split val --max-images 1 --warmup-images 1 --device cuda --weights yolov8n.pt --checkpoint runs\scout_spatial_mlp\scout_bitplane_stats_spatial.pt --out data\results\smoke_yolo_scout_live_3e633d3.json
```

## File Retention Ledger

| File | Decision | Reason |
|---|---|---|
| `.gitignore` | Keep | Excludes local data, runs, weights, builds, and Python artifacts. |
| `Makefile` | Keep | Small cross-platform native build/test entrypoint. |
| `README.md` | Keep | Main command truth for Windows setup, data, features, training, and routing. |
| `requirements.txt` | Keep | Minimal Python runtime/test dependencies. PyTorch CUDA install remains explicit in setup/docs. |
| `setup.ps1` | Keep | Windows helper for PyTorch CUDA install, non-torch requirements, and native build. |
| `setup.sh` | Keep | Small Unix helper for development and smoke checks. |
| `docs/binary_xnor_core.md` | Keep | Explains the native binary path and its limits. |
| `docs/current_results.md` | Keep | Records measured smoke results and caveats. |
| `docs/evaluation_protocol.md` | Keep | Gives benchmark commands for the report without a framework. |
| `docs/final_project_claims.md` | Keep | Prevents overclaiming and defines novelty framing. |
| `docs/completion_audit.md` | Keep | Final acceptance map and retention ledger. |
| `src/kernel.c` | Keep | Native C XNOR/popcount and threshold implementation. |
| `src/kernel_wrapper.py` | Keep | Sole ctypes boundary with dtype, shape, and library guards. |
| `src/binary_layer.py` | Keep | Thin Python layer around packed native kernels. |
| `src/scout.py` | Keep | Feature extractors and tiny scout model. |
| `src/preprocess.py` | Keep | VisDrone parsing, tile grid, labels, and feature utilities. |
| `src/routing.py` | Keep | Centralized selector logic used by recall and YOLO scripts. |
| `src/detector.py` | Keep | Box offset, NMS, and class-agnostic matching utilities. |
| `scripts/download_visdrone.py` | Keep | Official split downloader helper. |
| `scripts/verify_visdrone_dataset.py` | Keep | Dataset layout and annotation guard. |
| `scripts/make_tile_dataset.py` | Keep | Produces tile occupancy labels. |
| `scripts/verify_tile_dataset.py` | Keep | Verifies split counts, metadata, and train/val disjointness. |
| `scripts/extract_tile_features.py` | Keep | Generates bitplane and binary-XNOR tile feature caches. |
| `scripts/add_spatial_features.py` | Keep | Adds the spatial features that made the scout competitive. |
| `scripts/verify_tile_features.py` | Keep | Guards feature metadata and shape compatibility. |
| `scripts/train_scout.py` | Keep | Trains the lightweight linear/MLP scout. |
| `scripts/evaluate_scout_recall.py` | Keep | Measures tile-label recall for scout and baselines. |
| `scripts/run_yolo_tiles.py` | Keep | Main end-to-end detector/router CLI. |
| `scripts/render_scout_heatmap.py` | Keep | Optional visualization for presentation/debugging. |
| `scripts/benchmark_xnor_kernel.py` | Keep | Synthetic native-kernel benchmark. |
| `scripts/verify_packed_kernel.py` | Keep | C/Python parity, guard, and nonbinary input checks. |
| `scripts/verify_tile_contracts.py` | Keep | Fast tile/data contract checks. |
| `scripts/verify_detector_utils.py` | Keep | Fast detector utility checks. |
| `scripts/verify_tile_grid.py` | Keep | Explicit 49-tile grid contract check. |
| `scripts/verify_routing.py` | Keep | Selector determinism and oracle-greedy checks. |
| `tests/test_contracts.py` | Keep | Pytest wrapper around the fast verification scripts. |
| `docs/next_goal.md` | Delete | Goal contract was consumed into this audit and should not remain as a stale internal truth source. |
| `docs/pro_final_review.md` | Delete | Review feedback was consumed; keeping an AI-review transcript in the repo would be internal process noise. |

## LOC And Simplicity Check

Before the final documentation cleanup, tracked source/docs were 7085 lines. After removing the consumed goal contract and internal review transcript, the active tracked repo is 6271 lines. The active repo keeps the code direct: no package framework, no plugin system, no abstract base hierarchy, no notebook, no GUI, no ONNX/TensorRT branch, and no full-BNN detector branch.

The largest file is `src/kernel.c`, because the native C test harness lives beside the kernel. The largest Python script is `scripts/run_yolo_tiles.py`, because it owns the end-to-end smoke path and JSON timing output. Those are acceptable concentrations of complexity for a CLI school project, but they are also the first places to inspect if future changes grow scope.

## Remaining Work For The Report

These are not blockers for the implementation handoff:

1. Run larger YOLO sweeps, especially K=4, 8, 12, 16, and 20 over more validation images.
2. Decide whether the final report needs class-aware VisDrone mAP. If yes, use VisDrone-compatible YOLO weights or fine-tune.
3. Optimize live scout feature extraction; it is currently the largest latency cost.
4. Train and evaluate full binary-XNOR scout caches if the presentation needs the binary path as a measured scout result, not only a verified feature path.
5. Use `docs/final_project_claims.md` as the claim boundary when writing slides/report.
