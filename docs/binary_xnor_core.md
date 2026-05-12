# Binary XNOR Core

This document covers the older `binary-xnor-live` feature/MLP ablation. The
PowerPoint-aligned final routes are `xnor-heatmap-live` and
`xnor-heatmap-320-live`, where native XNOR-popcount accelerates the learned
heatmap scout's binary convolution body.
Neither route is a full detector replacement.

## What It Does

The active binary path is:

```text
RGB image
-> 24 RGB bitplanes
-> packed uint64 channels
-> C XNOR-popcount filters
-> thresholded binary maps
-> per-tile feature summaries
```

`src/kernel.c` is a normal user-space C library. On Windows it builds to `src/kernel.dll`; on Linux/macOS-style builds it builds to `src/kernel.so`.

`src/kernel_wrapper.py` is the only Python module that calls `ctypes`. It validates shapes, dtypes, dimensions, channel counts, and library loading before passing arrays to C.

`src/binary_layer.py` wraps the packed C kernels in `BinaryConvLayer`. It owns weight packing and exposes:

```text
forward([channels, H, W] uint8) -> [filters, H, W] int32 scores
forward_packed([H, W] uint64) -> [filters, H, W] int32 scores
apply_threshold(scores) -> [filters, H, W] uint8 maps
```

`forward_packed` uses the same native kernel as `forward`; it exists so live benchmarks can time bitplane extraction, packing, XNOR-popcount, thresholding, tile summaries, and MLP scoring separately.

`src/scout.py` exposes `BinaryXnorExtractor`, which reuses one `BinaryConvLayer` across a feature extraction run and records binary metadata. This is useful as a baseline and ablation, but it is not the final learned heatmap scout:

```text
filter count
kernel size
threshold
seed
input channels
weight SHA-256
```

## What It Does Not Claim

Do not claim:

```text
The binary path is a full VisDrone detector.
The binary path beats YOLO.
The binary path is faster in the full pipeline until measured end to end.
The default random binary filters are learned object detectors.
```

The honest claim is narrower:

```text
The repo contains verified native XNOR-popcount primitives. The final route
uses them inside the learned heatmap scout; this older feature extractor remains
an ablation.
```

## Verification

Core parity and guard checks:

```powershell
mingw32-make clean
mingw32-make
python scripts\verify_packed_kernel.py --include-nonbinary
```

One-image binary feature smoke:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor --split val --max-images 1 --out-dir data\tile_features_smoke --progress-every 0
python scripts\verify_tile_features.py --feature-file data\tile_features_smoke\binary_xnor_val_n1.npz --expect-feature-dim 64 --expect-feature-mode binary-xnor
```

The binary feature smoke should produce 49 tile rows for the default 7x7 grid and include binary metadata in `metadata_json`.

Hybrid binary scout smoke:

```powershell
python scripts\extract_tile_features.py --feature-mode binary-xnor-hybrid --split val --max-images 1 --out-dir data\tile_features_smoke --progress-every 0
python scripts\verify_tile_features.py --feature-file data\tile_features_smoke\binary_xnor_hybrid_val_n1.npz --expect-feature-dim 72 --expect-feature-mode binary-xnor-hybrid
```

Filter-budget hybrid from an existing 64-filter binary cache:

```powershell
python scripts\add_spatial_features.py --features data\tile_features\binary_xnor_val.npz --out data\tile_features\binary_xnor8_hybrid_val.npz --keep-first-features 8
python scripts\verify_tile_features.py --feature-file data\tile_features\binary_xnor8_hybrid_val.npz --expect-feature-dim 16 --expect-feature-mode binary-xnor-hybrid
```

Live detector route smoke after training a binary checkpoint:

```powershell
python scripts\run_yolo_tiles.py --selector binary-xnor-live --crop-source original --top-k 8 --split val --max-images 2 --device cuda --checkpoint runs\scout_binary_xnor_hybrid_mlp\scout_binary_xnor_hybrid.pt
```

The live route must be used for binary latency claims. Cached binary feature files are useful for training and routing-quality ablations, but they hide preprocessing and XNOR-popcount time.
