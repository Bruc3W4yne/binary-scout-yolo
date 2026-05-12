# Final Report Handoff

This folder is the clean handoff for writing the ACIT4630 report. It uses the final oracle-rank benchmark artifacts and avoids the older exploratory numbers unless explicitly marked historical.

## What To Copy Into Overleaf

Use `docs/overleaf_project/` as the active Overleaf project.

Uploaded Overleaf project:

- https://www.overleaf.com/project/6a0387af89d45da7283d7e8a

Files to upload:

- `docs/overleaf_project/main.tex`
- `docs/overleaf_project/references.bib`
- `docs/overleaf_project/figures/recall_vs_latency.png`
- `docs/overleaf_project/figures/small_recall_vs_latency.png`
- `docs/overleaf_project/figures/xnor_k_sweep.png`
- `docs/overleaf_project/figures/small_recall_recovery_vs_latency.png`

The same report body is mirrored at `docs/acit4630_ieee_report_skeleton.tex` for local editing.

## Safe Core Story

Use this as the report spine:

1. Full-image YOLO is fast but weak on small objects: small recall `0.084` at `13.7 ms`.
2. Exhaustive tiling improves small-object recall strongly: `0.355` at `107.2 ms`.
3. Oracle greedy selection proves the tile budget has redundancy: K12 reaches `0.329` small recall at `32.9 ms`; K18 reaches `0.345` at `44.0 ms`.
4. The learned and binary scouts test whether that oracle opportunity can be approximated without labels at inference.
5. The final native XNOR result is partial validation: oracle-rank XNOR K18 reaches `0.344` recall, `0.309` small recall, and `63.8 ms`, missing the strict target; K24 reaches stronger recall but uses too many tiles/latency to be the headline.

## Claims That Are Safe

- Scout-guided selective tiling is implemented end to end.
- Exhaustive original-resolution tiling materially improves small-object recall over full-image YOLO.
- Oracle selection shows that a much smaller tile subset can recover most of the exhaustive-tiling benefit.
- The native CPU XNOR scout path is implemented and benchmarked as a real tile router.
- Oracle-rank supervision improves low-K scout-only tile coverage and improves detector-level K12 XNOR recall.
- The final binary/XNOR result is a rigorous partial validation, not a full success.

## Claims To Avoid

- Do not claim state-of-the-art detection.
- Do not claim official VisDrone mAP.
- Do not claim the binary scout is near-free.
- Do not claim the binary scout beats the GPU heatmap route on the RTX 4090.
- Do not claim the system is proven drone-real-time or edge-deployed.
- Do not claim the custom detector loss was implemented.
- Do not make K24 the main low-cost result.

## Human Writing Still Needed

- Replace `Student 1, Student 2, Student 3, Student 4` with the actual author list.
- Replace the individual contribution bullets with accurate group-member contributions.
- Rewrite the abstract after the body is finalized.
- Turn the section scaffolding into polished prose in the group's voice.
- Decide whether to include all four figures or keep only the strongest two if the IEEE page limit gets tight.
- Add any examiner-requested appendix material separately from the 6-8 page main report.

## Useful Source Files

- Final benchmark source: `artifacts/benchmarks/oracle_rank_final/summary.md`
- Report tables: `docs/report_finalization/EVIDENCE_TABLES.md`
- Figure/table source map: `docs/report_finalization/FIGURE_TABLE_CHECKLIST.md`
- Claim safety audit: `docs/report_finalization/CLAIMS_AUDIT.md`
- Final report skeleton: `docs/acit4630_ieee_report_skeleton.tex`
- Active Overleaf project: `docs/overleaf_project/`
- Uploaded Overleaf project: https://www.overleaf.com/project/6a0387af89d45da7283d7e8a

## Final Framing Sentence

This project should be presented as a successful end-to-end implementation and a useful partial validation: selective tiling works, oracle routing shows a strong opportunity, and the binary/XNOR scout is real and measurable, but the current native scout does not yet fully reach the desired low-K recall/latency target.
