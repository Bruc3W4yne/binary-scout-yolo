# Binary/XNOR Final Router Experiments

This file records curated experiments only. Scratch reasoning goes in `EXPERIMENT_NOTES.md`.

| ID | Status | Question | Change | Metric | Result |
|---|---|---|---|---|---|
| E0 | done | What does the current evidence say? | Review current 30-image smoke and oracle results. | Recall, small recall, latency, oracle gap. | Oracle K=8 recovers 63% of tiling gain, so strict low-K near-exhaustive premise is too optimistic. Current XNOR-320 K=8 underperforms heuristic on small recall and leaves a large gap to oracle. |
| E1 | pending | Does non-GT diversity reduce wasted overlapping tiles? | Add tile-NMS/MMR policies. | Matched-budget recall and small recall vs topk. | Pending. |
| E2 | pending | Does heatmap coverage greedy better use the scout spatial output? | Add heatmap-coverage policy. | Recall gain at K <= 24. | Pending. |
| E3 | pending | Does adaptive K improve cost/recall balance? | Add adaptive MMR/coverage with frozen thresholds. | Mean K, p95 K, gain recovery, latency. | Pending. |
| E4 | pending | Is 320 resolution loss fixable? | Train/fine-tune 320 STE/XNOR scout if practical. | XNOR-320 recall vs current checkpoint. | Pending. |
