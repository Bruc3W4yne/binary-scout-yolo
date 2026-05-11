# Next Goal Draft

This is the comprehensive follow-up goal for finalizing the project after the first runnable prototype.

GPT-5.5 Pro final review status:

```text
Submitted in ChatGPT Pro with /Users/bruc3w4yne/binary-scout-yolo-final-pass-context.zip attached.
Conversation URL: https://chatgpt.com/c/6a01716f-1a28-8332-965b-c79cc32c70cb
Result consumed and distilled into docs/pro_final_review.md.
```

## Goal Text

Finalize the binary-scout-yolo school ML project into a credible, research-framed, fully runnable CLI artifact in `/Users/bruc3w4yne/binary-scout-yolo`, using Windows RTX 4090 over SSH for heavy verification.

GPT-5.5 Pro's final-pass verdict is now integrated in `docs/pro_final_review.md`: proceed, but freeze the architecture. The next goal is an evaluation-and-integration hardening pass, not a broad modeling exploration pass. Do not build a MobileNet scout, CUDA scout kernels, a full BNN detector, or a new detector architecture until the existing pipeline is measured correctly.

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

3. Baseline and binary scout meaning pass

   Decide and implement the simplest credible binary path. At minimum, run full-split binary-XNOR feature extraction/evaluation or document why it is too slow or weak. Compare timing and accuracy against bitplane/spatial MLP, random multi-seed, spatial-prior-only, oracle-count, oracle-greedy, full-image YOLO, and all-tiles. Keep CPU C/OpenMP unless evidence justifies CUDA, Numba, PyTorch quantization, or another route.

4. Scout model pass

   Evaluate bitplane stats, spatial MLP, binary-XNOR, and binary-XNOR + spatial. Pick the final scout by measured recall/latency, not aesthetics. Do not force binary-XNOR to win; the final report can honestly say binary was useful, weak, or too slow if the evidence says so.

5. Detector/evaluation pass

   Make YOLO routing evaluation credible with K sweeps, cached vs live timing, p50/p95 latency, selected area/tile count, random/oracle/full/all-tile baselines, and, if feasible, VisDrone-compatible mAP/AP_small/AR_small via fine-tuned or suitable weights. If full mAP is too heavy, clearly label class-agnostic routing recall as a smoke metric and provide the exact command for the heavy benchmark.

6. Performance pass

   Optimize the live scout bottleneck first. Add `torch.cuda.synchronize()` around GPU-timed sections, add warmup, split ground-truth parsing/matching out of headline latency, avoid repeated image work, and report CPU scout vs GPU YOLO phase times honestly.

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

1. Clean stale current-truth docs.

   Verification: `rg "evaluate_scout_yolo|--mode binary-xnor|--model bitplane-stats|scout_binary_xnor_linear|36" README.md PROD.md docs`.

2. Fix or quarantine `scripts/benchmark_packed.py`.

   Verification: `python scripts/benchmark_packed.py --help` and `python -m compileall -q scripts/benchmark_packed.py`.

3. Add `src/routing.py`.

   Verification: `python scripts/verify_routing.py`.

4. Upgrade `scripts/evaluate_scout_recall.py`.

   Add `prior`, `oracle-greedy`, random multi-trial mean/std, selected-area fraction, and selected tile count.

   Verification: run random/prior/oracle-greedy K sweeps on the val feature cache.

5. Add spatial-prior and spatial-only baselines.

   Verification: the project can prove whether learned scouts beat simple location priors.

6. Refactor binary-XNOR feature extraction for reuse.

   Verification: binary feature extraction no longer reconstructs the binary layer unnecessarily per image, and metadata records binary params.

7. Add cached-vs-live feature verification.

   Verification: `python scripts/verify_live_features.py --feature-file <cache> --feature-mode binary-xnor --max-images 3`.

8. Fix `run_yolo_tiles.py` timing.

   Verification: CUDA and CPU smoke runs produce per-phase timing, aggregate mean/p50/p95, and headline latency excluding ground-truth parsing.

9. Add binary-XNOR live scout support.

   Verification: cached and live binary selected tile IDs match on the same deterministic sample.

10. Run the first credible routing and detector sweep.

    Verification: `data/results/sweep_val100/summary.md` and `summary.json` contain detector recall, precision, F1, TP/FP/FN, selected area, tile count, and latency mean/p50/p95 for every method.
