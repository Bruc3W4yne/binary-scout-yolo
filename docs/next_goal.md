# Next Goal

This is the canonical follow-up goal after the first runnable prototype.

It incorporates the May 11, 2026 GPT-5.5 Pro cleanup review from:

```text
https://chatgpt.com/c/6a018ab2-a66c-8328-af47-cd8423f58633
```

The repo zip attached to Pro was the current git-tracked source of truth. Pro's second-pass correction changed the priority order:

1. End-to-end pipeline function and credible evidence come first.
2. Simplicity and efficiency come second.
3. Code/doc polish comes third.

The key correction is important: a clean-looking repo that cannot run the full scout -> tile selection -> YOLO -> merge/evaluate path is a failure. Cleanup must not become a framework rewrite, and lower LOC is only valuable when it preserves or improves correctness, performance, and readability.

## Goal Text

Ruthlessly finish `binary-scout-yolo` as a working, simple, benchmark-ready CLI pipeline.

Codex acts as implementation agent, verification engineer, cleanup auditor, and project finisher.

Primary priority:

```text
The full scout-guided selective tiling pipeline must work end-to-end and produce credible evaluation evidence.
```

Secondary priority:

```text
Keep the repo simple, efficient, and readable. Lower LOC is good only when it preserves or improves correctness, performance, and clarity. Do not reduce LOC by hiding logic, weakening tests, or removing required functionality.
```

Third priority:

```text
Polish code and docs. Remove stale, misleading, duplicated, generated-looking, or dead artifacts. A cleanup pass that increases tracked LOC is suspicious unless the added code is required for functionality, tests, timing/schema support, or honest evaluation.
```

Do not call the project done after the first working pass. Iterate until implementation, verification, cleanup, claim audit, GPT-5.5 Pro review, and post-review fixes are complete.

## Code Quality Standard

For this project, "good code" means evidence-bearing code first: every active module, script, and helper should either run the pipeline, verify the pipeline, benchmark the pipeline, or document exactly how to do those things.

Follow these standards:

- Prefer boring, direct CLI tools over a framework.
- Keep module boundaries few and obvious.
- Share selector, metadata, timing, and result-writing logic only where it prevents real duplication.
- Do not add abstractions unless they remove confusion or prevent inconsistent behavior.
- Keep the C/Python boundary narrow and heavily validated.
- Make expected user failures clear: missing data, missing kernel build, bad checkpoint, bad feature metadata, or unavailable CUDA should fail with a useful message.
- Keep tests and verifiers small, fast, and tied to real risks.
- Keep comments sparse. Explain contracts and non-obvious native/ML details; remove narration and generated-looking boilerplate.
- Keep docs short and command-true. There should be fewer active truth sources, not more.
- Delete stale material instead of archiving it, unless historical context is genuinely needed.
- Treat LOC as a signal, not a target. Net LOC growth must be justified by required functionality, verifiers, timing/schema support, or honest evaluation evidence.

## Hard Constraints

- Main deliverable: Windows-first local CLI pipeline for the RTX 4090 PC.
- Docker and notebooks are out of scope for this pass.
- Use official VisDrone train and val where available.
- Do not add broad new models or research branches.
- Do not add a full BNN detector, MobileNet scout, TensorRT, ONNX export, CUDA scout kernels, GUI, notebook demo, or CI matrix.
- Do not overclaim. COCO-pretrained YOLO smoke runs are wiring evidence, not final VisDrone accuracy.
- Do not claim binary-XNOR speed or accuracy unless measured in this repo.

## Phase 0: Preserve Function, Inventory, Identify Breakage

Run first:

```powershell
git status --short
git ls-files
python -m compileall -q .
mingw32-make clean
mingw32-make
mingw32-make test
python scripts\verify_packed_kernel.py --include-nonbinary
python scripts\verify_tile_contracts.py
python scripts\verify_detector_utils.py
python scripts\verify_tile_grid.py
python scripts\benchmark_packed.py --help
```

On macOS or Linux, use `make clean && make && make test` instead of `mingw32-make`.

Create a retention ledger for every tracked file:

- keep
- rewrite
- move/rename
- delete
- archive only if genuinely necessary

Record:

- broken commands
- stale docs
- duplicated logic
- misleading claims
- generated-looking or overexplained text
- current tracked LOC/diffstat baseline

Gate 0:

```text
Do not edit broadly until current breakage is known. Confirm or disprove the benchmark_packed.py --help failure.
```

Known local caveat: on this Mac, `python3` works while `python` may not exist. Also, `make test` was observed hanging locally after building, while Pro reported it passed in its isolated review. Treat that as a real discrepancy to investigate, not as a solved issue.

## Phase 1: Prove A Minimal End-To-End Path Before Large Cleanup

Before major restructuring, preserve or restore the shortest viable pipeline path.

Dataset and tile path:

- VisDrone verification works if data exists.
- Tile dataset creation and verification work on a small `--max-images` run if data exists.
- If data does not exist, scripts fail clearly and help text documents required paths.

Feature and scout path:

- Extract bitplane-stats features.
- Append spatial features.
- Train a tiny scout checkpoint.
- Evaluate recall for at least scout, random, and oracle-greedy.

Detector path:

- Run YOLO routing smoke for `full`, `all`, `random`, `heuristic`, `oracle-greedy`, and `scout` where data/checkpoint exist.
- Produce at least one result JSON with timing phases.

Kernel path:

- Verify XNOR/popcount kernel.
- Run synthetic kernel benchmark.

Gate 1:

```text
There must be a short-run evidence bundle, or a documented blocker with exact failing command, output, likely cause, and minimal next action. Do not proceed to cosmetic cleanup while the core pipeline is broken.
```

## Phase 2: Add Only Evaluation-Critical Missing Functionality

Required selectors:

- `full`
- `all`
- `random` with seed and multi-trial support
- `prior` trained from train split only
- `heuristic` using existing image/feature statistics, no new model
- `oracle-count`
- `oracle-greedy` set-cover style object coverage
- `scout` from cached features/checkpoint
- `scout-live` only if implemented cleanly; otherwise document as not implemented

Required result evidence:

- selected tile count
- selected area fraction
- object/tile recall where applicable
- detector detections and class-agnostic recall for YOLO smoke
- selector metadata
- seed
- command
- feature/checkpoint metadata
- dataset/split/max-images
- timing phases

Required timing phases:

- `image_load_ms`
- `resize_preprocess_ms`
- `gt_parse_ms`
- `scout_ms`
- `yolo_ms`
- `merge_nms_ms`
- `match_eval_ms`
- `pipeline_ms_excl_gt`
- `wall_ms`

Timing rules:

- Synchronize CUDA around GPU sections.
- Include warmup.
- Summaries include mean, p50, p95, min, and max.
- Do not mix ground-truth parsing into pipeline compute claims.

Gate 2:

```text
A sweep or equivalent repeated run can compare full/all/random/prior/heuristic/oracle/scout on a small dataset without code changes.
```

## Phase 3: Simplify Architecture Without Building A Framework

Convert to an importable package only if it improves reliability:

- Preferred layout: `src/binary_scout_yolo/`
- No script-level `sys.path` hacks.
- Scripts import package modules.

Keep modules few and coherent. Acceptable modules:

- `preprocess`
- `features`
- `binary_layer`
- `kernel_wrapper`
- `detector`
- `routing`
- `scout_model`
- `timing` or result helpers only if they reduce duplication

Do not create abstract base classes, registries, plugin systems, dataclass forests, or generic experiment frameworks.

Shared logic should replace duplication in scripts:

- selector logic
- checkpoint loading/scoring
- result writing
- timing summary
- feature metadata checks

Gate 3:

```text
The package must be simpler than the previous script sprawl, not merely more formal. Include a diffstat and explain any net LOC increase.
```

## Phase 4: Binary-XNOR Path Cleanup

Keep the C/Python boundary narrow:

- Only `kernel_wrapper.py` calls `ctypes`.
- The wrapper validates dtype, shape, contiguity, and library availability.
- Higher-level code does not touch `ctypes`.

Fix or replace `scripts/benchmark_packed.py`:

- Prefer rename to `scripts/benchmark_xnor_kernel.py`.
- `--help` works without data.
- Synthetic mode works.
- JSON output works.
- No inflated speedup claims.

Binary feature extraction:

- Reuse extractor/layer per run.
- Record binary metadata: filters, kernel size, threshold, seed, input channels, and weight hash.
- Support `binary-xnor-spatial` only if it is straightforward.

Live binary routing:

- Implement only if clean and verified.
- Otherwise state plainly that cached binary-XNOR features are verified but live binary routing remains future work.

Gate 4:

```text
Kernel verification, synthetic benchmark, and binary feature extraction smoke pass.
```

## Phase 5: Tests And Verifiers Focused On Real Risk

Required pure tests:

- tile/grid contracts
- detector offset/NMS/matching utilities
- routing selectors, especially oracle-greedy and random determinism
- feature metadata compatibility
- timing/result summary sanity if implemented

Required script verifiers:

- packed kernel
- tile contracts
- detector utils
- tile grid
- routing
- live feature parity if live route is claimed

Do not add large test scaffolds, mock-heavy tests, or coverage theater.

Gate 5:

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
python scripts\benchmark_xnor_kernel.py --synthetic --max-images 5 --runs 10 --warmup 2 --out data\results\smoke_xnor_kernel.json
```

Use equivalent `make` commands on macOS/Linux.

## Phase 6: Ruthless Cleanup

Delete stale files unless there is a strong reason to keep them. Prefer deletion over archive.

Remove:

- stale `PROD.md` if it conflicts with current truth
- `docs/next_goal.md` after the final implementation goal is achieved and extracted into completion docs
- `docs/legacy_plans/`
- `scripts/legacy/`
- `experiments/bnn_detector/` unless the user explicitly wants it preserved
- stale commands
- personal paths
- old project names
- misleading comments
- generic boilerplate
- duplicated helpers
- unsupported claims

Keep docs short and authoritative:

- `README.md`
- `docs/evaluation_protocol.md`
- `docs/current_results.md`
- `docs/final_project_claims.md`
- `docs/binary_xnor_core.md`
- `docs/completion_audit.md`
- `docs/pro_final_review.md` only as historical context, not as a truth source

README must contain exact working commands:

- Windows setup
- kernel build
- verifiers
- dataset prep
- feature extraction
- scout training
- recall comparison
- YOLO routing smoke
- longer benchmark command

Gate 6:

```text
Search current docs/scripts for stale command names, old project names, personal absolute paths, and dead modes. Fix all hits unless clearly inside archived historical context.
```

## Phase 7: Evidence Bundle And Benchmark Readiness

Produce or document:

- kernel synthetic benchmark JSON
- scout recall comparison JSON/table
- YOLO routing smoke JSON/table
- timing summary with required phases
- completion audit with commands run

If VisDrone data is available:

- Run `--max-images` smoke on official val.

If VisDrone data is not available:

- Do not fake performance.
- Prove synthetic/unit wiring only.
- Document exact official VisDrone commands to run later.

Longer benchmark commands must be ready but do not need to be run:

- official train+val tile prep
- full feature extraction
- scout training
- `run_yolo_sweep` over selectors and top-K values, or an equivalent documented command sequence

Gate 7:

```text
The repo can run short comparisons and is ready for long benchmarks without code changes.
```

## Phase 8: Final File-By-File Audit And Pro Review

Run `git ls-files`.

For every tracked file, justify:

- why it exists
- whether it is current
- whether it supports the pipeline, tests, docs, or setup
- whether it contains stale text or dead code

Delete or rewrite anything that fails justification.

Produce final diffstat:

- files added
- files deleted
- files renamed
- net LOC change
- justification for any net LOC increase

Run all final verification commands again.

Prepare a GPT-5.5 Pro review package:

- final tree
- diff summary
- retention ledger
- commands run and outputs
- evidence files
- known limitations
- claims intended for final report

Use GPT-5.5 Pro for final review.

If Pro flags blocking issues:

- Fix them.
- Rerun final verification.
- If a Pro objection is intentionally not fixed, ask the user to explicitly waive that exact objection.

Final stopping condition:

```text
Stop only when implementation, verification, cleanup, claim audit, file-by-file audit, Pro review, and post-review fixes are complete. Otherwise stop with exact blockers and next actions, not a vague "done."
```

## Final Acceptance Gate

The project is accepted only when all of the following are true.

End-to-end evidence exists:

- At least one scout recall comparison result.
- At least one YOLO routing smoke result.
- At least one kernel verification/benchmark result.
- Timing includes the required phase breakdown.
- Results are clearly labeled smoke vs benchmark.

Core commands pass:

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
python scripts\benchmark_xnor_kernel.py --synthetic --max-images 5 --runs 10 --warmup 2 --out data\results\smoke_xnor_kernel.json
```

Short pipeline smoke passes where data exists:

- feature extraction
- spatial append
- scout training
- recall evaluation
- YOLO routing for `full`, `all`, `random`, `heuristic`, `oracle`, and `scout`

Every tracked file is justified:

- No unreviewed stale docs.
- No legacy scripts in the active path.
- No broken advertised commands.
- No unsupported claims.
- Diffstat and LOC growth justification are documented.

GPT-5.5 Pro final review is completed:

- Codex presents the final repo state to GPT-5.5 Pro.
- Blocking Pro feedback is fixed.
- Final verification is rerun after fixes.
- The project is not "done" until Pro review agrees or the user explicitly waives remaining objections.

## Do Not Do

These are out of scope even if they sound like quality work:

| Do not do | Reason |
|---|---|
| Add a full BNN detector | Distracts from scout-guided routing and creates misleading scope. |
| Add MobileNet/TinyCNN scout baselines | Broadens model comparison before the current pipeline is finished. |
| Add TensorRT/ONNX/export paths | Optimization distraction; not required for evaluation readiness. |
| Add Docker | Immediate target is local Windows. |
| Add notebooks/demos | Explicitly out of scope for this pass. |
| Add a generic experiment framework | Bloat; scripts plus shared helpers are enough. |
| Add plugin registries or abstract base classes | Overengineering for a small project. |
| Add many docs | Fewer truth-source docs are better. |
| Archive lots of stale material | Preserves confusion; delete instead. |
| Chase LOC reduction blindly | Smaller code that hides complexity or weakens correctness is worse. |
| Add heavy mock-based tests | Prefer small tests/verifiers that catch real pipeline breakage. |
| Rewrite all code for style | Refactor only when it improves correctness, reliability, or duplication. |
| Add CI matrix | Nice later, not needed for the final school deliverable. |
| Add final mAP claims from smoke runs | Misleading unless evaluated properly with a trained detector/baseline. |
| Optimize the C kernel deeply before timing proves it matters | Measure first; avoid premature optimization. |
