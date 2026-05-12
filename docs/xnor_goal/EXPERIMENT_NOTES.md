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
- Windows verification after the 320 training change: all four verifier scripts passed and `python -m pytest -q` reported `5 passed`.
- Trained `runs\heatmap_scout\heatmap_ste_320_e5.pt` from `heatmap_ste_e10.pt` with `--input-size 320 --epochs 5 --batch 16 --workers 4 --lr 1e-4 --device cuda`.
- 320-native scout coverage improved strongly: XNOR-320 tile-NMS K16/K20/K24 reached `0.942/0.969/0.974` object-center coverage; heatmap-coverage reached `0.980/0.994/0.996`.
- Detector benchmark did not satisfy the strict XNOR contract. K18 is inside budget at `56.9 ms` but reaches only `0.80/0.83` gain recovery. K24 reaches `0.86/0.88` gain recovery but costs `68%` of exhaustive tiling. Final blocker: CPU XNOR scout overhead plus remaining ranking gap.
