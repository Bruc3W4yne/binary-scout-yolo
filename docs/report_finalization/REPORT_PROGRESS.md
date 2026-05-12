# Report Finalization Progress

## Checked

- `README.md`
- `docs/final_project_claims.md`
- `docs/xnor_heatmap_final_decision.md`
- `docs/learned_heatmap_scout.md`
- `docs/evaluation_protocol.md`
- `docs/completion_audit.md`
- `docs/current_results.md`
- `docs/final_binary_pass_results.md`
- `docs/acit4630_ieee_report_skeleton.tex`
- `docs/overleaf_project/main.tex`
- `docs/references.bib`
- `docs/overleaf_project/references.bib`
- `artifacts/benchmarks/oracle_rank_final/summary.md`
- Final benchmark JSON files under `artifacts/benchmarks/oracle_rank_final/`
- `/Users/bruc3w4yne/Downloads/Exam evaluation guidelines-ACIT4630.pdf`
- `/Users/bruc3w4yne/Downloads/Small Object Detection 2-2.pdf`

## Findings

- The final artifact is internally consistent and is the best source of truth for report numbers.
- Existing report skeletons still contained older benchmark rows from before the oracle-rank final pass.
- The exam guidelines require a 6-8 page IEEE-style scientific report with Abstract, Keywords, Introduction, Background/Related Work, Proposed Solution, Experiments, Results, Discussion, Conclusion, References, and Individual Contributions.
- The presentation frames the intended idea as binary/XNOR scout-guided selective tiling: downsample/scout, rank top-K tiles, run full detector only on selected tiles, then merge detections.
- The final implementation matches the pipeline structure but only partially validates the strongest binary/XNOR performance target.

## Changed

- Added report-finalization tracking docs.
- Added report-ready evidence tables and LaTeX snippets.
- Added `scripts/make_final_report_figures.py` to generate figures from frozen benchmark JSON files.
- Updated the IEEE report skeleton and Overleaf project to use final oracle-rank benchmark numbers.
- Added final handoff guidance for Overleaf/report writing.

## Left For User

- Replace placeholder author names.
- Fill in exact individual contributions.
- Write final prose in the group's voice.
- Decide whether to include appendices with code commands or keep the report body tighter.
- Continue writing in the uploaded Overleaf project: https://www.overleaf.com/project/6a0387af89d45da7283d7e8a

## Verification Notes

- Figure generation: passed with `python3 scripts/make_final_report_figures.py`.
- Generated figures were written to both `docs/report_finalization/figures/` and `docs/overleaf_project/figures/`.
- Python syntax check: passed with `python3 -m py_compile scripts/make_final_report_figures.py`.
- Whitespace check: passed with `git diff --check`.
- Zip integrity check: passed with `zip -T docs/acit4630_overleaf_ieee_project.zip`.
- Overleaf zip contents were checked with `zipinfo`; stale macOS `._*` resource-fork entries were removed.
- LaTeX local compile: not run because `latexmk`, `pdflatex`, and `tectonic` are not installed on this Mac.
- Basic LaTeX structure check: passed for balanced braces and matching table/figure environments in both report files.
- Overleaf upload and compile: passed. The uploaded project opened and rendered a 3-page IEEE PDF preview.
