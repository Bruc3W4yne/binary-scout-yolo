# Final Project Claims

## Thesis

The project implements a selective tiling pipeline for UAV small-object detection: score overlapping tiles with a cheap scout, run YOLO only on selected tiles, merge detections back into image coordinates, and measure recall versus compute.

The contribution is the measured pipeline combination, not a new detector architecture.

The high-resolution claim only applies when tiled detector runs use `--crop-source original`. In that mode the scout still selects tiles on the 640x640 grid, but YOLO sees crops from the original image and detections are projected back to the 640x640 evaluation canvas.

## Final Strong Claim

The defensible claim is:

```text
Scout-guided selective tiling is a useful recall/latency tradeoff for UAV small-object detection. On the final 100-image VisDrone validation benchmark, full-image YOLO reached 0.084 small-object recall at 13.7 ms, while exhaustive original-resolution tiling reached 0.355 small-object recall at 107.2 ms. Oracle tile selection showed that most of this tiling benefit can be recovered with far fewer tiles.
```

The final binary/XNOR claim is partial validation:

```text
The native CPU XNOR scout is implemented, measured, and aligned with the intended scout-router architecture, but it did not meet the strict K18 recall/latency target. Oracle-rank XNOR K18 reached 0.344 recall, 0.309 small recall, and 63.8 ms against a target of 0.355 / 0.318 / <=60.1 ms.
```

## Binary-XNOR Claim

The PowerPoint-aligned binary routes are `xnor-heatmap-live` and `xnor-heatmap-320-live`: the learned STE heatmap scout runs its binary convolution body through native packed XNOR-popcount on CPU, ranks top-K tiles, and YOLO detects only those selected crops on the GPU.

This validates the intended architecture/operation split:

```text
CPU native XNOR scout selection -> GPU YOLO tile detection -> merged boxes
```

The 640 scout is the exact-reference native route and remains slower than CUDA/PyTorch on the RTX 4090 workstation. The 320 route is the performance-aligned route used for the final oracle-rank benchmark. It recovers meaningful tiling recall at K12-K24, but the final detector-level XNOR result is not the fastest or strongest route on the desktop benchmark.

The older `binary-xnor-live` route is a separate feature/MLP ablation. It proves the repo has a narrow XNOR/popcount feature core and wrapper that can be tested and benchmarked, but it must not be presented as the final learned heatmap scout.

That older binary comparison should separate:

```text
binary-xnor pure: XNOR-popcount tile summaries only
binary-xnor hybrid: XNOR-popcount tile summaries plus cheap tile geometry/spatial-prior features
```

The hybrid route is still a binary-centered ablation because the image features come from packed bitplanes and C XNOR-popcount filters, but it must not be described as pure binary or as the PowerPoint-aligned learned scout. Latency claims for that ablation require `binary-xnor-live`, not cached features.

For the final report, lead with `xnor-heatmap-320-live` and the oracle-rank final artifacts when discussing the practical CPU-XNOR scout result. Use older 8-filter, 16-filter, 64-filter, and feature/MLP binary routes only as historical ablations if there is space.

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
