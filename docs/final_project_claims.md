# Final Project Claims

## Thesis

The project implements a selective tiling pipeline for UAV small-object detection: score overlapping high-resolution tiles with a cheap scout, run YOLO only on selected tiles, merge detections back into image coordinates, and measure recall versus compute.

The contribution is the measured pipeline combination, not a new detector architecture.

## Strong Claim

The defensible claim is:

```text
Scout-guided selective tiling can be measured against full-image YOLO, all-tile inference, random top-K, train-split spatial priors, heuristics, and oracle upper bounds. In the current full-val tile-label evaluation, the tiny spatial MLP scout selects higher-value tiles than random, a train-split spatial prior, and a content-only heuristic at K=8.
```

## Binary-XNOR Claim

The native C binary path is a verified edge-oriented scout feature implementation. It proves the project has a narrow XNOR/popcount core and Python wrapper that can be tested and benchmarked.

It is not yet evidence that binary-XNOR is the fastest live scout on the target machine. That requires a full live binary-scout benchmark and comparison against the current bitplane/spatial scout.

## Claims To Avoid

Do not claim:

```text
This beats YOLO overall.
This is state of the art.
The binary kernel improves detector accuracy by itself.
The current pipeline is already drone-real-time.
COCO-pretrained YOLO smoke results are final VisDrone mAP.
```

## Novelty Framing

SAHI-style tiling, YOLO, binary neural networks, and XNOR/popcount inference are known ideas. The project is interesting because it combines them into one simple, testable routing system and reports where compute is spent.

The report should frame novelty as:

```text
cheap scout routing + selective high-resolution tiles + YOLO detection + measured recall/latency/tile-budget tradeoff
```

## Sources

- SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection, arXiv 2202.06934: https://arxiv.org/abs/2202.06934
- SAHI docs: https://obss.github.io/sahi/
- VisDrone: Detection and Tracking Meet Drones Challenge, arXiv 2001.06303: https://arxiv.org/abs/2001.06303
- VisDrone dataset: https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics YOLO docs: https://docs.ultralytics.com/
- XNOR-Net, arXiv 1603.05279: https://arxiv.org/abs/1603.05279
- Binarized Neural Networks, arXiv 1602.02830: https://arxiv.org/abs/1602.02830
