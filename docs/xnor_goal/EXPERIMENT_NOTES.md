# Binary/XNOR Final Router Notes

## 2026-05-12

- Pro's critique: current K=36/40 results are not a meaningful selective-routing win because they select most of the 49-tile grid.
- Oracle interpretation: K=8 is not enough to reach near-exhaustive recall under the current detector/grid, even with ground-truth tile choice. However, current XNOR-320 top-K is also leaving a real ranking gap.
- Immediate engineering hypothesis: plain score-sorted top-K can waste slots on overlapping high-score neighboring tiles. Diversity-aware and heatmap-coverage selectors are the fastest fair test before retraining.
- Goal-mode shape: keep a tight 30-image feedback loop, save every benchmark artifact, and only promote promising candidates to larger validation.
- Added routing policies and diagnostics in the benchmark path. Local checks passing so far: `python3 -m compileall src scripts`, `python3 scripts/verify_detector_utils.py`, `python3 scripts/verify_routing.py`, `python3 scripts/verify_heatmap_scout.py`, and `python3 scripts/verify_xnor_heatmap_scout.py`.
- `python3 -m pytest -q` is not runnable on this Mac because `pytest` is not installed here; use the Windows environment for the project test suite.
- Windows 100-image result: learned GPU heatmap + tile-NMS meets the strict selective-routing criteria, proving the pipeline idea. Native CPU XNOR-320 + tile-NMS improves over plain top-K but narrowly misses the strict positive threshold: K20 stays inside the latency budget but reaches only about 81%/83% overall/small gain recovery; K24 reaches about 84%/86% but costs about 68% of exhaustive tiling.
- Next fair test: train/fine-tune the STE scout directly at 320 input while keeping 640 tile coordinates. This checks whether the current XNOR-320 route is limited by using a 640-trained checkpoint at 320 inference resolution.
- Added `--input-size` to `scripts/train_heatmap_scout.py`; local 320 synthetic smoke passes with `python3 scripts/train_heatmap_scout.py --synthetic-images 4 --epochs 1 --batch 2 --input-size 320 --variant ste --out /tmp/heatmap_ste_320_smoke.pt`.
