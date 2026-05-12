# Binary/XNOR Final Router Experiments

This file records curated experiments only. Scratch reasoning goes in `EXPERIMENT_NOTES.md`.

| ID | Status | Question | Change | Metric | Result |
|---|---|---|---|---|---|
| E0 | done | What does the current evidence say? | Review current smoke, 100-image, and oracle results. | Recall, small recall, latency, oracle gap. | Oracle K=12 can recover useful tiling gain, so low-call routing is viable in principle. Current XNOR-320 top-K still leaves a ranking gap. |
| E1 | done | Does non-GT diversity reduce wasted overlapping tiles? | Add tile-NMS/MMR policies. | Matched-budget recall and small recall vs topk. | Yes. On 100 images, XNOR-320 tile-NMS K20 improves over top-K K20 from 0.329/0.286 to 0.348/0.309 recall/small recall at a similar cost fraction. |
| E2 | done | Does heatmap coverage greedy better use the scout spatial output? | Add heatmap-coverage policy. | Recall gain at K <= 24. | Mixed. It improves object-center coverage and 30-image YOLO recall, but added selection overhead makes tile-NMS the cleaner timed route. |
| E3 | done | Does adaptive K improve cost/recall balance? | Add adaptive MMR/coverage with frozen thresholds. | Mean K, p95 K, gain recovery, latency. | Not yet. Adaptive heatmap coverage gives high object-center coverage, but 30-image YOLO recall did not beat fixed-K tile-NMS under the latency budget. |
| E4 | active | Is 320 resolution loss fixable? | Train/fine-tune 320 STE/XNOR scout if practical. | XNOR-320 recall vs current checkpoint. | Training support added and smoke-tested locally. Windows training/benchmark pending. |
