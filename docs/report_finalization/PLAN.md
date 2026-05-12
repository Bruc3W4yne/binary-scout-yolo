# Final Report Readiness Plan

This is the final report/readiness pass. It is intentionally not a new research, training, detector-tuning, or native-kernel cycle.

## Phase 1 - Reconcile Evidence

Status: complete.

- Use `artifacts/benchmarks/oracle_rank_final/summary.md` as the source of truth.
- Treat older docs as historical unless explicitly updated.
- Use the exam guidelines PDF for report structure: IEEE article format, 6-8 pages excluding references/appendices, clear scientific sections, working-solution documentation, and individual contributions.
- Use the presentation as the project-intent source: binary/XNOR scout-guided selective tiling for UAV small-object detection.

## Phase 2 - Package Tables And Claims

Status: complete.

- Create final benchmark tables in Markdown and LaTeX.
- Separate oracle upper bounds, learned GPU references, objectness XNOR, and oracle-rank XNOR.
- State clearly that the final binary/XNOR result is partial validation.
- Keep unsupported claims out of report-facing text.

## Phase 3 - Generate Figures

Status: complete.

- Generate figures from existing benchmark JSON files only.
- Save figures under `docs/report_finalization/figures/` and `docs/overleaf_project/figures/`.
- Do not rerun detection, training, or benchmarks.

## Phase 4 - Update Report Materials

Status: complete.

- Update `docs/acit4630_ieee_report_skeleton.tex`.
- Mirror the active report in `docs/overleaf_project/main.tex`.
- Keep BibTeX entries real and limited to verified existing sources.

## Phase 5 - Verify And Handoff

Status: complete.

- Figure-generation script ran successfully.
- Python syntax check passed for `scripts/make_final_report_figures.py`.
- Basic LaTeX structure check passed: balanced braces and matching table/figure environments.
- Local LaTeX compile could not be run because `latexmk`, `pdflatex`, and `tectonic` are not installed on this Mac.
- Human-writing tasks are recorded in `FINAL_REPORT_HANDOFF.md`.
