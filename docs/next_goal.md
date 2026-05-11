# Next Goal Draft

This is the comprehensive follow-up goal for finalizing the project after the first runnable prototype.

GPT-5.5 Pro final review status:

```text
Submitted in ChatGPT Pro with /Users/bruc3w4yne/binary-scout-yolo-final-pass-context.zip attached.
Conversation URL: https://chatgpt.com/c/6a01716f-1a28-8332-965b-c79cc32c70cb
Result not yet consumed because Computer Use lost access to the Zen window before the final response could be read.
First task in the new goal is to recover and integrate that response.
```

## Goal Text

Finalize the binary-scout-yolo school ML project into a credible, research-framed, fully runnable CLI artifact in `/Users/bruc3w4yne/binary-scout-yolo`, using Windows RTX 4090 over SSH for heavy verification.

First recover the submitted GPT-5.5 Pro final-review conversation at:

```text
https://chatgpt.com/c/6a01716f-1a28-8332-965b-c79cc32c70cb
```

The current git-tracked repo zip was attached there. Extract Pro's critical feedback into `docs/pro_final_review.md`. If Browser/Computer Use remains blocked after reasonable retries, document the blocker and proceed from independent research plus the existing repo audit. Use GPT-5.5 Pro again at major architecture/research/review crossroads whenever available, but never invent Pro feedback.

Ground the final framing in reputable sources:

```text
SAHI / sliced inference for small objects
VisDrone dataset and detection challenge
XNOR-Net / binary neural networks
Ultralytics / YOLO evaluation metrics
Recent adaptive, density-guided, or guided slicing work
```

Continue until these phase gates are genuinely satisfied or blockers are documented with exact next actions.

1. Research and claim lock

   Update `docs/research_framing.md` and create `docs/final_project_claims.md` with a narrow defensible thesis, novelty statement, related-work notes, acceptable claims, forbidden claims, and citation links.

2. Code-quality and command-truth pass

   Remove or quarantine remaining misleading/stale material. Make README/current audit the only runnable command truth. Fix stale `PROD.md` conflicts or split `PROD.md` into current spec vs archive. Simplify naming/imports where it improves clarity without broad packaging bloat.

3. Binary scout meaning pass

   Decide and implement the simplest credible binary path. At minimum, run full-split binary-XNOR feature extraction/evaluation or document why it is too slow or weak. Compare timing and accuracy against bitplane/spatial MLP, random, oracle, and heuristic baselines. Keep CPU C/OpenMP unless evidence justifies CUDA, Numba, PyTorch quantization, or another route.

4. Scout model pass

   Evaluate bitplane stats, spatial MLP, binary-XNOR, and one small learned occupancy scout only if it materially improves the story. Pick the final scout by measured recall/latency, not aesthetics.

5. Detector/evaluation pass

   Make YOLO routing evaluation credible with K sweeps, cached vs live timing, p50/p95 latency, selected area/tile count, random/oracle/full/all-tile baselines, and, if feasible, VisDrone-compatible mAP/AP_small/AR_small via fine-tuned or suitable weights. If full mAP is too heavy, provide a documented smoke metric and exact command for the heavy benchmark.

6. Performance pass

   Optimize the live scout bottleneck first. Add timing synchronization around GPU calls. Avoid repeated image work. Report CPU scout vs GPU YOLO phase times honestly.

7. Reproducibility pass

   Ensure Windows setup, dataset download/verification, feature cache creation, training, evaluation, heatmap/demo generation, and benchmark commands work from clean instructions.

8. Demo/report artifact pass

   Create a concise demo script or command sequence that produces heatmap/image examples and result JSON/tables suitable for group presentation.

9. Final ruthless audit

   Review every tracked source/doc/script for bloat, misleading comments, dead paths, brittle assumptions, unnecessary abstraction, and unverified claims.

10. Final handoff

    Update `README.md`, `docs/current_results.md`, `docs/completion_audit.md`, `docs/final_project_claims.md`, and any result tables with what is verified, what remains unverified, exact commands, and how to present the project.

Do not mark the goal complete until the repo is clean, pushed to GitHub, Windows verification evidence is collected, and the final audit maps each gate to concrete files, commands, and results.

## Research Anchors

- SAHI establishes the slice, detect, merge pattern for small objects and reports AP gains on aerial datasets including VisDrone/xView.
- SAHI docs describe why high-resolution images resized to detector input size lose small-object detail, and why overlapping tiles plus merge are useful.
- VisDrone provides the drone detection benchmark and official train/val/test-dev splits.
- XNOR-Net motivates binary input/weight convolution with XNOR/popcount-style efficiency and CPU-friendly inference.
- Ultralytics evaluation docs frame mAP, precision/recall, speed metrics, and class-wise diagnostics as the right detector-performance vocabulary.
- Recent adaptive/guided slicing papers make the project less isolated: the research trend is reducing exhaustive tiling cost by selecting better regions.

## First 10 Implementation Tasks

1. Recover GPT-5.5 Pro final review and save it to `docs/pro_final_review.md`.

   Verification: `Test-Path docs\pro_final_review.md` on Windows or `test -f docs/pro_final_review.md` on Mac.

2. Create `docs/final_project_claims.md`.

   Verification: document contains thesis, novelty, related work, acceptable claims, forbidden claims, and citations.

3. Split or clean stale `PROD.md` command sections.

   Verification: `rg "evaluate_scout_yolo|--model bitplane-stats|scout_binary_xnor_linear|36" README.md PROD.md docs` shows no stale runnable-command truth outside archived/history sections.

4. Add a benchmark manifest format.

   Likely files: `src/metrics.py`, `scripts/run_benchmark_suite.py`, `docs/current_results.md`.

   Verification: a smoke manifest can run full/random/oracle/scout-live on `--max-images 2` and writes one consolidated JSON.

5. Add robust timing synchronization.

   Likely file: `scripts/run_yolo_tiles.py`.

   Verification: output JSON includes `load_ms`, `scout_ms`, `yolo_ms`, `merge_ms`, `latency_ms`, p50, p95, and CUDA synchronization when `device=cuda`.

6. Optimize live bitplane/spatial feature extraction.

   Likely files: `src/scout.py`, `scripts/run_yolo_tiles.py`.

   Verification: `scout-live` K=8 on a fixed small subset is faster than the previous documented scout phase without changing selected tile IDs.

7. Run full binary-XNOR val feature extraction, then decide whether full train is feasible.

   Verification: `python scripts\extract_tile_features.py --feature-mode binary-xnor --split val` and `python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor_val.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor`.

8. Train/evaluate binary-XNOR scout or document why it loses.

   Verification: K sweep JSON compares binary-XNOR vs spatial MLP vs random vs oracle.

9. Add detector K sweep.

   Verification: consolidated results for `full`, `all`, `random K=[4,8,12,16,20]`, `oracle K=[4,8,12,16,20]`, `scout K=[4,8,12,16,20]`, and `scout-live K=[8]` on a defined subset.

10. Create final demo command.

    Verification: one command produces heatmap PNG/JSON plus routed YOLO result JSON for a known VisDrone val image.
