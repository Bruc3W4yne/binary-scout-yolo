# Historical Benchmark Run: `a878807`

This file is retained as historical evidence from an earlier report pass. Do not use it as the final report source of truth. The final report numbers are in `artifacts/benchmarks/oracle_rank_final/summary.md` and `docs/report_finalization/EVIDENCE_TABLES.md`.

Hardware: Windows desktop, RTX 4090 FE, CUDA inference.

Data: VisDrone validation subset unless noted. Detector results below use 100 images,
YOLOv8n COCO weights, and original-resolution tile crops.

## Gate Result

Continue. The measured pipeline supports the core project claim:

- Full-frame YOLO is fastest but has low small-object recall.
- Exhaustive tiled YOLO has the highest measured recall but requires 49 detector calls.
- Scout-guided selective tiling improves recall over full-frame YOLO and over simple
  selection baselines at low K, while using far fewer detector calls than exhaustive tiling.

Do not claim state of the art, VisDrone mAP, or that the binary-XNOR scout is the best
selector unless later binary-XNOR results support that.

## Detector Results

| Method | K | Recall | Small recall | Detector calls | Mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full image YOLO | - | 0.129 | 0.084 | 1 | 13.1 | 20.4 |
| All tiles YOLO | 49 | 0.399 | 0.355 | 49 | 102.1 | 112.8 |
| Random tiles | 8 | 0.155 | 0.142 | 8 | 23.9 | 30.4 |
| Spatial prior | 8 | 0.188 | 0.184 | 8 | 24.0 | 30.3 |
| Heuristic | 8 | 0.157 | 0.139 | 8 | 25.4 | 32.6 |
| Scout cached | 8 | 0.228 | 0.210 | 8 | 25.4 | 33.5 |
| Binary-XNOR scout cached | 8 | 0.191 | 0.168 | 8 | 25.0 | 32.1 |
| Scout live | 8 | 0.228 | 0.210 | 8 | 29.9 | 39.0 |
| Oracle greedy | 8 | 0.332 | 0.305 | 8 | 24.0 | 30.8 |
| Scout cached | 12 | 0.267 | 0.245 | 12 | 32.1 | 38.3 |
| Binary-XNOR scout cached | 12 | 0.245 | 0.217 | 12 | 31.1 | 36.5 |
| Scout live | 12 | 0.267 | 0.245 | 12 | 38.7 | 46.0 |
| Scout cached | 16 | 0.311 | 0.283 | 16 | 39.2 | 43.9 |
| Scout cached | 20 | 0.334 | 0.302 | 20 | 46.9 | 54.1 |

## Tile Routing Recall

Full validation feature cache. These are tile-level object-center recall results, not
detector results.

| Selector | K=4 | K=8 | K=12 | K=16 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.251 +/- 0.004 | 0.452 +/- 0.004 | 0.611 +/- 0.007 | 0.727 +/- 0.005 | 0.816 +/- 0.007 |
| Spatial prior | 0.295 | 0.469 | 0.542 | 0.677 | 0.740 |
| Heuristic | 0.247 | 0.447 | 0.600 | 0.727 | 0.818 |
| Scout | 0.364 | 0.516 | 0.635 | 0.735 | 0.806 |
| Oracle greedy | 0.782 | 0.953 | 0.995 | 1.000 | 1.000 |

Interpretation: the scout is strongest at low K, where selective routing matters most.
At larger K, random and heuristic selection cover enough image area that the tile-only
metric becomes less discriminative. Detector-level results are therefore the main evidence.

## Binary-XNOR Status

The C-kernel binary-XNOR path passed correctness tests, full train/validation feature
extraction, CUDA scout training, and full-validation tile-recall evaluation.

| Selector | K=4 | K=8 | K=12 | K=16 | K=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.251 +/- 0.004 | 0.452 +/- 0.004 | 0.611 +/- 0.007 | 0.727 +/- 0.005 | 0.816 +/- 0.007 |
| Spatial prior | 0.295 | 0.469 | 0.542 | 0.677 | 0.740 |
| Heuristic | 0.199 | 0.352 | 0.493 | 0.602 | 0.692 |
| Binary-XNOR scout | 0.289 | 0.491 | 0.643 | 0.760 | 0.837 |
| Oracle greedy | 0.782 | 0.953 | 0.995 | 1.000 | 1.000 |

Interpretation: the binary-XNOR scout is weaker than the spatial bitplane scout at K=4
and K=8, but it beats random/prior/heuristic at K=8 and beats random/prior/heuristic at
K=12 and above in tile recall. In detector routing, cached binary-XNOR reaches 0.191 recall
at K=8 and 0.245 at K=12, which is below the spatial scout but still above or near the
simple K=8/K=12 baselines. This supports the original binary-scout framing, while the
spatial bitplane scout remains the strongest measured router.
