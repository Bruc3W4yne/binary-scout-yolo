# Final Project Claims

## Thesis

The project implements a selective tiling pipeline for UAV small-object detection: score overlapping tiles with a cheap scout, run YOLO only on selected tiles, merge detections back into image coordinates, and measure recall versus compute.

The contribution is the measured pipeline combination, not a new detector architecture.

The high-resolution claim only applies when tiled detector runs use `--crop-source original`. In that mode the scout still selects tiles on the 640x640 grid, but YOLO sees crops from the original image and detections are projected back to the 640x640 evaluation canvas.

## Strong Claim

The defensible claim is:

```text
Scout-guided selective tiling can be measured against full-image YOLO, all-tile inference, random top-K, train-split spatial priors, heuristics, and oracle upper bounds. In the current full-val tile-label evaluation, the tiny spatial MLP scout selects higher-value tiles than random, a train-split spatial prior, and a content-only heuristic at K=8.
```

If original-crop detector benchmarks are positive, the stronger report claim is:

```text
Selected original-resolution tile inference recovers part of the all-tile recall gain over full-image YOLO while using K detector inputs instead of 49.
```

## Binary-XNOR Claim

The native C binary path is a verified edge-oriented scout feature implementation. It proves the project has a narrow XNOR/popcount core and Python wrapper that can be tested and benchmarked.

The final binary comparison should separate:

```text
binary-xnor pure: XNOR-popcount tile summaries only
binary-xnor hybrid: XNOR-popcount tile summaries plus cheap tile geometry/spatial-prior features
```

The hybrid route is still a binary-centered scout because the image features come from packed bitplanes and C XNOR-popcount filters, but it must not be described as pure binary. Binary latency claims require `binary-xnor-live`, not cached features.

After the final binary pass, the most defensible binary result is the 8-filter live hybrid: it improves over the previous binary scout, keeps the XNOR-popcount visual path, and is faster than exhaustive tiling, but it is still slower than the spatial scout. The pure 64-filter binary route is now measured as a live detector row and proves the ablation, but it is too slow to be the practical route. The 64-filter hybrid is a useful high-recall ablation, not the deployable configuration.

## Claims To Avoid

Do not claim:

```text
This beats YOLO overall.
This is state of the art.
The binary kernel improves detector accuracy by itself.
The current pipeline is already drone-real-time.
COCO-pretrained YOLO smoke results are final VisDrone mAP.
The reported best scout result is binary-XNOR when it was produced by bitplane/spatial features.
High-resolution tiling when the detector crops came from the resized 640x640 canvas.
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
