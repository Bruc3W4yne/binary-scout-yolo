# Binary XNOR Core

This project uses binary/XNOR-popcount code as a cheap scout feature path, not as a full detector replacement.

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
apply_threshold(scores) -> [filters, H, W] uint8 maps
```

`src/scout.py` exposes `BinaryXnorExtractor`, which reuses one `BinaryConvLayer` across a feature extraction run and records binary metadata:

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
The repo contains a verified native XNOR-popcount feature extractor that can produce tile features for a scout/router pipeline.
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
