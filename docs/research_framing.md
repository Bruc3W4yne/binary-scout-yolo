# Research Framing

## Project Claim

This project is best framed as:

```text
Binary/XNOR scout-guided selective tiling for efficient UAV small-object detection.
```

The novelty is the measured pipeline, not a new detector architecture:

```text
full UAV image
-> cheap scout scores overlapping high-resolution tiles
-> select top-K tiles
-> run YOLO only on selected tiles
-> merge detections
-> compare recall/latency/tile-count tradeoffs
```

## Why This Is Interesting

Small objects in UAV imagery are harmed by full-image resizing. Exhaustive sliced inference can recover detail, but it spends detector compute on every tile. The project explores the middle ground: use a cheap tile router to recover some sliced-inference benefit with fewer YOLO calls.

This makes the work a systems/measurement contribution:

```text
SAHI-like tiled inference benefit
+ binary/bitplane scout routing
+ VisDrone tile labels
+ YOLO selected-tile detection
+ recall vs latency vs tile budget ablations
```

## Defensible Claims

The current project can defend these claims:

```text
Selective routing is measurable against random, oracle, full-image, and all-tile baselines.
The scout is detector-agnostic: YOLO remains the final detector.
Binary/XNOR code is implemented and verified as a plausible edge-oriented scout feature path.
The strongest current scout is tiny: bitplane/RGB/spatial features plus a 64-hidden-unit MLP.
```

## Claims To Avoid

Do not claim:

```text
The project beats YOLO overall.
The current scout is real-time on edge hardware.
The binary scout is a full detector.
The binary kernel alone improves detection accuracy.
The method is state of the art.
```

## Best Ablation Table

The final report should include:

| Method | YOLO calls | Tile budget | Recall / mAP | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Full-image YOLO | 1 | full image | baseline | baseline | baseline |
| All tiles / SAHI-like | 49 | all tiles | upper-cost reference | high | high |
| Random top-K | K | selected | weak baseline | lower | lower |
| Oracle top-K | K | selected | upper bound | lower | lower |
| Scout top-K | K | selected | target | lower | lower |
| Live scout top-K | K | selected | honest end-to-end | includes scout | includes scout |

The key sentence to aim for is:

```text
At a tight K tile budget, the scout covers more objects than random while running YOLO on a small fraction of the exhaustive tile set.
```

## Sources

- SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection, arXiv 2202.06934: https://arxiv.org/abs/2202.06934
- SAHI docs, sliced inference for small object detection: https://obss.github.io/sahi/
- VisDrone: Detection and Tracking Meet Drones Challenge, arXiv 2001.06303: https://arxiv.org/abs/2001.06303
- VisDrone official dataset repository: https://github.com/VisDrone/VisDrone-Dataset
- YOLOv8 / Ultralytics documentation: https://docs.ultralytics.com/
- XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks, arXiv 1603.05279: https://arxiv.org/abs/1603.05279
- Binarized Neural Networks, arXiv 1602.02830: https://arxiv.org/abs/1602.02830
- FINN: A Framework for Fast, Scalable Binarized Neural Network Inference, arXiv 1612.07119: https://arxiv.org/abs/1612.07119
