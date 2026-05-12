# Binary/XNOR Final Router Plan

Goal: give the native binary/XNOR heatmap scout its fairest final shot, then make an evidence-backed yes/no decision.

## Success Contract

- Deployable selectors must not use validation/test labels before evaluation.
- Final tiled detector benchmarks use `--crop-source original`.
- `topk` must reproduce current behavior.
- A positive result requires a no-GT XNOR route with mean calls <= 24, latency <= 60% of exhaustive tiling, overall and small gain recovery >= 85%, and a matched-budget improvement over current `xnor-heatmap-320-live topk` of at least 0.015 recall or small recall, or within 0.02 of oracle.
- If that is not reached after the fair routing/model pass, the final result is negative with a specific blocker.

## Work Plan

- [x] Add simple routing policies: `topk`, `tile-nms`, `mmr`, `heatmap-coverage`, `adaptive-mmr`, `adaptive-heatmap-coverage`.
- [x] Add routing diagnostics and adaptive-K summaries.
- [x] Add tests for compatibility, geometry, adaptive clamps, heatmap scaling, and benchmark crop-source checks.
- [x] Run 30-image Windows smoke benchmarks for the required route set.
- [x] Promote the best candidates to larger validation if the smoke result is promising.
- [ ] Train and benchmark a 320-native STE/XNOR scout to test whether resolution mismatch is the remaining blocker.
- [ ] Write `docs/xnor_heatmap_final_decision.md` with the final yes/no result and blocker analysis.
