# Phase 0 Inventory

Generated for the final cleanup goal. This is an evidence ledger, not a final completion audit.

## Baseline Results

| Check | Result | Evidence |
|---|---|---|
| `git status --short` | Dirty after first targeted fix | `scripts/benchmark_packed.py` replaced with `scripts/benchmark_xnor_kernel.py`; `docs/next_goal.md` updated. |
| `git ls-files` | 51 tracked files before cleanup | Includes stale root `PROD.md`, `docs/legacy_plans/`, `scripts/legacy/`, and `experiments/bnn_detector/`. |
| Tracked LOC baseline | 11,372 lines | Largest bloat source is `PROD.md` at 2,550 lines. |
| `python --version` | Fail on Mac | `python` command is not installed in this environment. Use `python3` locally or document Windows `python`. |
| `python3 --version` | Pass | Python 3.9.6 on Mac. |
| `python3 -m compileall -q .` | Pass | Current Python source compiles. |
| `make clean` | Pass on Mac | Removes native outputs. |
| `make` | Fixed on Mac | Makefile now omits `-fopenmp` on Darwin while preserving OpenMP flags for Windows/Linux. |
| `make test` | Fixed on Mac | Darwin test build now omits sanitizers that caused the local test binary to hang; kernel tests pass 55/55. |
| `python3 scripts/verify_packed_kernel.py --include-nonbinary` | Pass after native build fix | Packed kernel verifier passes, including nonbinary 0/255 input-plane handling. |
| `python3 scripts/verify_tile_contracts.py` | Pass | Tile/grid/parser/bitplane/area/split checks pass. |
| `python3 scripts/verify_detector_utils.py` | Pass | IoU/NMS/tile offset/matching checks pass. |
| `python3 scripts/verify_tile_grid.py` | Pass | 49-tile 160/80 grid covers 640x640. |
| `python3 scripts/benchmark_packed.py --help` | Confirmed broken before replacement | It sampled absent VisDrone images before argument parsing and raised `ValueError`. |
| `python3 scripts/benchmark_xnor_kernel.py --help` | Pass after replacement | Help prints without data, CUDA, or native library. |
| `python3 scripts/benchmark_xnor_kernel.py --synthetic --max-images 2 --runs 2 --warmup 1` | Pass after native build fix | Writes ignored JSON evidence under `data/results/`; smoke run measured XNOR and float32 reference kernels. |
| Active `scripts/*.py --help` | Pass | Every active top-level script supports `--help`; legacy scripts were not included. |

## Initial Retention Ledger

| Path | Action | Reason |
|---|---|---|
| `.gitignore` | rewrite | Keep, but simplify decorative/noisy comments during final cleanup. |
| `Makefile` | rewrite | Required for native kernel; must handle Windows-first and non-OpenMP Mac failure honestly. |
| `PROD.md` | delete/rewrite | Too large and stale; replace useful current content with short truth-source docs. |
| `README.md` | rewrite | Keep as main quickstart; update commands and claims after implementation. |
| `docs/binary_core_explained.md` | rewrite | Useful conceptually but stale; replace with concise `docs/binary_xnor_core.md`. |
| `docs/completion_audit.md` | rewrite | Keep final audit, but current version is pre-final. |
| `docs/current_results.md` | rewrite | Keep as smoke/results truth source; separate smoke from benchmarks. |
| `docs/legacy_plans/README.md` | delete | Historical planning material, not active pipeline truth. |
| `docs/legacy_plans/detailed_plan.md` | delete | Historical planning material, not active pipeline truth. |
| `docs/legacy_plans/full_model_plan.md` | delete | Historical planning material, not active pipeline truth. |
| `docs/legacy_plans/highlevelplan.md` | delete | Historical planning material, not active pipeline truth. |
| `docs/next_goal.md` | keep | Canonical acceptance contract until final completion. |
| `docs/pro_final_review.md` | keep/rewrite | Keep as Pro context, not command truth. |
| `docs/research_framing.md` | rewrite | Merge useful claims into final claims/evaluation docs. |
| `experiments/bnn_detector/README.md` | delete | Full BNN detector is out of scope. |
| `experiments/bnn_detector/bnn_model.py` | delete | Full BNN detector is out of scope. |
| `experiments/bnn_detector/train_bnn.py` | delete | Full BNN detector is out of scope. |
| `requirements.txt` | keep/rewrite | Required setup file; audit for Windows/PyTorch clarity. |
| `scripts/add_spatial_features.py` | keep | Active feature pipeline. |
| `scripts/benchmark_packed.py` | replace | Confirmed broken; replaced by synthetic-first XNOR benchmark. |
| `scripts/benchmark_xnor_kernel.py` | keep | Active native benchmark with working help and JSON output. |
| `scripts/download_visdrone.py` | keep | Dataset acquisition path. |
| `scripts/evaluate_scout_recall.py` | rewrite | Add missing prior/heuristic/oracle-greedy/random multi-trial evidence. |
| `scripts/extract_tile_features.py` | rewrite | Keep; improve binary metadata and avoid repeated binary layer creation. |
| `scripts/legacy/benchmark_kernel.py` | delete | Legacy script outside active path. |
| `scripts/legacy/benchmark_numpy_f32.py` | delete | Legacy script outside active path. |
| `scripts/legacy/benchmark_yolo.py` | delete | Legacy script outside active path. |
| `scripts/legacy/make_process_images.py` | delete | Legacy script outside active path. |
| `scripts/legacy/pack_data.py` | delete | Legacy script outside active path. |
| `scripts/legacy/run_first_layer.py` | delete | Legacy script outside active path. |
| `scripts/legacy/run_pipeline.py` | delete | Legacy script outside active path. |
| `scripts/legacy/sobel_demo.py` | delete | Legacy script outside active path. |
| `scripts/legacy/verify_kernel.py` | delete | Legacy script outside active path. |
| `scripts/legacy/verify_packing.py` | delete | Legacy script outside active path. |
| `scripts/make_tile_dataset.py` | keep | Active tile dataset builder. |
| `scripts/render_scout_heatmap.py` | keep/rewrite | Useful presentation artifact; ensure it remains CLI-truthful. |
| `scripts/run_yolo_tiles.py` | rewrite | Active detector/router path; needs timing phases and selector cleanup. |
| `scripts/train_scout.py` | keep/rewrite | Active scout training path; audit metadata and errors. |
| `scripts/verify_detector_utils.py` | keep | Focused verifier. |
| `scripts/verify_packed_kernel.py` | keep | Focused native verifier. |
| `scripts/verify_tile_contracts.py` | keep | Focused contract verifier. |
| `scripts/verify_tile_dataset.py` | keep | Dataset verifier. |
| `scripts/verify_tile_features.py` | keep | Feature cache verifier. |
| `scripts/verify_tile_grid.py` | keep | Focused grid verifier. |
| `scripts/verify_visdrone_dataset.py` | keep | Dataset structure verifier. |
| `setup.ps1` | rewrite | Windows-first setup must match MSYS2/UCRT64 and current commands. |
| `setup.sh` | rewrite | Keep only if honest about macOS/Linux native build requirements. |
| `src/binary_layer.py` | rewrite | Keep but simplify comments and package imports later. |
| `src/detector.py` | keep/rewrite | Active detector utilities; likely keep small. |
| `src/kernel.c` | keep | Native XNOR/popcount core. |
| `src/kernel_wrapper.py` | rewrite | Keep; narrow ctypes boundary and simplify once package layout is settled. |
| `src/preprocess.py` | keep | Active VisDrone/tile contract code. |
| `src/scout.py` | rewrite | Active feature/scout helpers; needs binary extractor reuse and metadata. |

## Confirmed First Fix

`scripts/benchmark_packed.py` was replaced with `scripts/benchmark_xnor_kernel.py`.

Why this is the simplest clean fix:

- The old script mixed dataset loading, benchmark setup, and execution before parsing `--help`.
- It depended on absent VisDrone files for a kernel benchmark that can be synthetic.
- It returned flattened float weights while the wrapper expects `[channels, kH, kW]`.
- The replacement keeps one job: synthetic native-kernel timing with JSON output.

Follow-up native-build fix:

- The Makefile now uses a no-OpenMP Darwin build so `make`, `make test`, `verify_packed_kernel.py`, and the synthetic XNOR benchmark can run on Mac.
- Windows and Linux keep the OpenMP build flags.
