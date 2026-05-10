# Binary Core Explained

This document explains the inherited binary/XNOR implementation in plain terms.
It is meant to help the team understand what exists before changing it.

## Short Verdict

The current binary code is a useful low-level feature extraction layer. It is not
yet the full scout pipeline.

Keep and harden these pieces:

```text
src/kernel.c
src/kernel_wrapper.py
src/binary_layer.py
```

Treat these as demos or legacy support, not final project logic:

```text
scripts/run_first_layer.py
scripts/legacy/benchmark_kernel.py
scripts/legacy/benchmark_numpy_f32.py
scripts/legacy/pack_data.py
scripts/legacy/verify_packing.py
scripts/legacy/sobel_demo.py
```

The next real project layer should be a small scout module that turns binary
feature maps into tile scores and selected tile coordinates.

## What Binary XNOR-Popcount Means

Normal convolution multiplies numbers and sums them:

```text
pixel_value * weight_value
```

Binary convolution uses values that behave like `-1` and `+1`. Instead of doing
many floating-point multiplications, it asks whether input bits and weight bits
match.

```text
input bit agrees with weight bit     -> +1
input bit disagrees with weight bit  -> -1
```

The C code computes this as:

```text
score = 2 * popcount(matches) - total_comparisons
```

For a first layer using 24 input channels and a 3x3 filter:

```text
total_comparisons = 24 * 3 * 3 = 216
score range       = -216 to +216
```

A high positive score means "this local image patch looks like this binary
filter." A low negative score means the opposite.

## What The Current Code Actually Does

The intended useful path is:

```text
RGB image
-> split into 24 bit planes
   - R bits 0..7
   - G bits 0..7
   - B bits 0..7
-> pack those 24 binary channels into one uint64 per pixel
-> run packed XNOR-popcount filters in C
-> output integer feature maps
-> optionally threshold feature maps into binary feature maps
```

This is a binary feature extractor. It does not yet:

```text
score candidate tiles
select top-K tiles
train from VisDrone tile labels
run YOLO on selected crops
merge YOLO detections
evaluate final accuracy or latency
```

So the inherited code is a foundation for the scout, not the scout itself.

## File-By-File Map

## `src/kernel.c`

This is a normal user-space C shared library, despite the filename `kernel.c`.
It is not an operating-system kernel driver.

On Windows it should compile to:

```text
src/kernel.dll
```

On Linux or WSL2 it should compile to:

```text
src/kernel.so
```

Important functions:

```text
xnor_popcount_conv
  Simple single-channel reference-style XNOR convolution.
  Useful for teaching and tests.

xnor_packed_u8_conv
  Packed XNOR convolution for up to 8 channels.
  Useful for tests and microbenchmarks.

xnor_packed_u64_conv
  Packed XNOR convolution for up to 64 channels.
  Useful for the final binary feature path.

xnor_multi_filter_conv
  Applies many packed uint64 filters in one C call.
  This is the most important final-path C function.

pack_channels_to_u64
  Packs [channels, H, W] uint8 planes into [H, W] uint64.
  Any nonzero plane value is treated as bit 1.

threshold_i32_to_u8
  Thresholds int32 feature maps into uint8 binary maps.
  Useful helper.

sobel_conv
  Edge detection demo.
  Not part of the final binary scout claim.
```

## `src/kernel_wrapper.py`

This is the Python-to-C bridge using `ctypes`.

It loads the compiled C library and exposes Python functions for the C kernels.
This file is required for the final binary/XNOR claim because it proves Python
is calling native C code.

Current guardrail:

```text
The wrapper validates array dimensionality and expected shapes before passing
raw pointers to C.
```

Bad Python calls should raise clear Python errors instead of reaching the C
layer with mismatched buffers.

## `src/binary_layer.py`

This is the main Python abstraction over the C binary convolution functions.

`BinaryConvLayer`:

```text
input:
  [n_ch, H, W] uint8 bit planes

internal:
  packs bit planes into [H, W] uint64
  packs filter weights into [n_filters, kH, kW] uint64
  calls xnor_multi_filter_conv

output:
  [n_filters, H, W] int32 score maps
```

This is useful and should be kept, but it should be understood as a binary
feature layer. It is not currently a trained BNN detector and not a complete
scout.

Important limitation:

```text
The default filters are random.
```

Random filters are fine for smoke tests. They are not evidence that the scout
can find objects. To make this useful for the final project, tile scores must be
trained or evaluated against VisDrone tile labels.

## `scripts/run_first_layer.py`

This is a visualization/smoke script.

It loads one image, creates 24 bit planes, runs random binary filters, and saves
feature map images.

This is helpful for seeing that the binary layer runs, but it should not be
presented as the actual scout.

Better mental name:

```text
smoke_binary_layer.py
```

## `scripts/benchmark_packed.py`

This is a useful microbenchmark.

It compares packed XNOR-popcount against a naive float32 reference for similar
convolution work.

It can support a narrow claim:

```text
Packed XNOR-popcount can be faster than naive float32 convolution for equivalent binary convolution work.
```

It cannot support a broad claim:

```text
The full detection pipeline is faster or more accurate.
```

The full pipeline must be benchmarked separately.

## `scripts/legacy/pack_data.py`

This file is easy to misunderstand.

It packs bits across horizontal pixel positions:

```text
[3, 8, 640, 10] uint64
```

The current C scout path expects one `uint64` per pixel where bits represent
channels:

```text
[H, W] uint64
```

Those are both "bit-packed" layouts, but they are not the same layout.

For the final scout path, do not use `pack_data.py` as input to
`BinaryConvLayer`.

## `scripts/legacy/benchmark_kernel.py`, `scripts/legacy/benchmark_numpy_f32.py`, `scripts/legacy/sobel_demo.py`

These are Sobel edge detection demos/benchmarks.

Sobel is not the final scout. It can remain as legacy/demo code, but it should
not be central to the project story.

## Known Issues To Fix First

## 1. `pack_channels_to_u64` should clamp inputs to one bit

Current C logic effectively does this:

```c
output[i] |= (uint64_t)src[i] << shift;
```

That only works if `src[i]` is exactly `0` or `1`.

If a caller accidentally passes `255`, it can set many bits instead of one.

Safer logic:

```c
output[i] |= ((uint64_t)(src[i] != 0)) << shift;
```

This is the first code fix I recommend.

## 2. Python wrapper needs strict validation

Before calling C, `src/kernel_wrapper.py` should check:

```text
input dimensionality
weight dimensionality
kH and kW are positive odd integers
n_ch is within valid range
n_filters is within valid range
weights shape matches kH, kW, and n_filters
```

Bad input should fail as a Python `ValueError`, not as unsafe C behavior.

## 3. `BinaryConvLayer` should be hardened

It should validate:

```text
n_filters >= 1
1 <= n_ch <= 64
kH >= 1
kW >= 1
kH and kW are odd
0 <= filter_idx < n_filters
planes are 3D
planes have expected channel count
planes are binary or explicitly converted to binary
```

## 4. The main final-path kernel needs a parity test

The most important function for the project is:

```text
xnor_multi_filter_conv
```

We need a script that compares it against a slow NumPy reference on small inputs.

Recommended script:

```text
scripts/verify_packed_kernel.py
```

This should test cases like:

```text
n_ch=1,  n_filters=1, k=1
n_ch=8,  n_filters=3, k=3
n_ch=24, n_filters=5, k=3
n_ch=64, n_filters=2, k=3
```

## How This Connects To The Final Scout

The real scout should look like this:

```text
640x640 RGB image
-> rgb_to_bitplanes_u8
-> BinaryConvLayer.forward
-> binary feature maps or score maps
-> aggregate features over candidate tiles
-> produce one score per tile
-> select top-K tile indices
-> pass selected crops to YOLO later
```

The missing modules should be small and single-purpose:

```text
src/scout/bitplanes.py
  Convert RGB images into [24, H, W] bit planes.

src/scout/tile_grid.py
  Generate candidate tile coordinates.

src/scout/tile_scoring.py
  Convert feature maps into one score per tile.

src/scout/binary_scout.py
  Combine bitplanes, binary features, tile scoring, and top-K selection.

scripts/run_binary_scout.py
  CLI smoke test that writes tile scores and debug images.
```

## Recommended Step Order

Make one focused change at a time:

```text
Step 1:
  Add this explanation document.

Step 2:
  Fix pack_channels_to_u64 and add/extend a test for binary input clamping.

Step 3:
  Add strict wrapper and BinaryConvLayer validation.

Step 4:
  Add verify_packed_kernel.py for xnor_multi_filter_conv parity.

Step 5:
  Add src/scout/bitplanes.py and verify_bitplanes.py.

Step 6:
  Add src/scout/tile_grid.py and verify_tile_grid.py.

Step 7:
  Add tile scoring and BinaryScout smoke path.
```

## Open Decision: Tile Grid Coverage

`PROD.md` currently mentions a default 160x160 tile size and stride 80, but it
also uses a 6x6 grid with starts:

```text
0, 80, 160, 240, 320, 400
```

With 160x160 tiles, the last tile ends at 560. That means the far-right and
bottom 80 pixels of a 640x640 image do not receive full start-position coverage.

For full coverage, starts should include 480:

```text
0, 80, 160, 240, 320, 400, 480
```

That produces:

```text
7 x 7 = 49 candidate tiles
```

Recommended default:

```text
49 candidate tiles, top_k=12
```

This keeps compute bounded while avoiding edge blind spots.
