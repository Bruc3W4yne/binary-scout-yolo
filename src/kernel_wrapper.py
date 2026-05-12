"""
ctypes bridge for the user-space XNOR/popcount C library.

This module owns the Python-to-C safety boundary: arrays are made contiguous,
shapes are checked, and only then are raw pointers passed to src/kernel.c.
"""

import ctypes
import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_NAME = "kernel.dll" if sys.platform == "win32" else "kernel.so"
_LIB_PATH = os.path.join(_HERE, _LIB_NAME)
_C_INT_MAX = 2_147_483_647

_lib = None
_dll_dir_handles = []


def _require_int(name: str, value: int, min_value: int = 1, max_value: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def _require_kernel(kH: int, kW: int) -> tuple[int, int]:
    kH = _require_int("kH", kH, max_value=_C_INT_MAX)
    kW = _require_int("kW", kW, max_value=_C_INT_MAX)
    if kH % 2 == 0 or kW % 2 == 0:
        raise ValueError("kH and kW must be odd")
    _require_c_int_product("kH*kW", kH, kW)
    return kH, kW


def _require_channels(n_ch: int, max_channels: int) -> int:
    return _require_int("n_ch", n_ch, max_value=max_channels)


def _as_c_array(name: str, array: np.ndarray, dtype: np.dtype, ndim: int) -> np.ndarray:
    array = np.ascontiguousarray(array, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    if any(dim < 1 for dim in array.shape):
        raise ValueError(f"{name} dimensions must be positive, got shape {array.shape}")
    for i, dim in enumerate(array.shape):
        _require_int(f"{name}.shape[{i}]", dim, max_value=_C_INT_MAX)
    return array


def _require_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")


def _require_c_int_product(name: str, *values: int) -> int:
    total = 1
    for value in values:
        value = _require_int(name, value, max_value=_C_INT_MAX)
        total *= value
        if total > _C_INT_MAX:
            raise ValueError(f"{name} exceeds C int range: {total}")
    return total


def _register_windows_dll_dirs() -> None:
    """Make MinGW/MSYS2 runtime DLLs visible to Python on Windows."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    candidates = [_HERE]
    candidates.extend(os.environ.get("PATH", "").split(os.pathsep))
    candidates.extend(
        [
            r"C:\msys64\ucrt64\bin",
            r"C:\msys64\mingw64\bin",
            r"C:\msys64\clang64\bin",
        ]
    )

    seen = set()
    for path in candidates:
        if not path:
            continue
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen or not os.path.isdir(path):
            continue
        seen.add(norm)
        try:
            _dll_dir_handles.append(os.add_dll_directory(path))
        except OSError:
            pass


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"{_LIB_NAME} not found at {_LIB_PATH}. Run `make` first."
            )
        _register_windows_dll_dirs()
        try:
            _lib = ctypes.CDLL(_LIB_PATH)
        except OSError as exc:
            if sys.platform == "win32":
                raise OSError(
                    f"Found {_LIB_NAME} at {_LIB_PATH}, but Windows could not load "
                    "it or one of its MinGW/MSYS2 runtime dependencies. Make sure "
                    r"C:\msys64\ucrt64\bin is installed and available, then retry."
                ) from exc
            raise

        # int xnor_popcount_conv(uint8*, int8*, int32*, int, int, int, int)
        _lib.xnor_popcount_conv.restype = ctypes.c_int
        _lib.xnor_popcount_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int sobel_conv(uint8*, float*, int, int)
        _lib.sobel_conv.restype = ctypes.c_int
        _lib.sobel_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]

        # int xnor_packed_u8_conv(uint8*, uint8*, int32*, int, int, int, int, int)
        _lib.xnor_packed_u8_conv.restype = ctypes.c_int
        _lib.xnor_packed_u8_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int xnor_packed_u64_conv(uint64*, uint64*, int32*, int, int, int, int, int)
        _lib.xnor_packed_u64_conv.restype = ctypes.c_int
        _lib.xnor_packed_u64_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int pack_channels_to_u64(uint8*, uint64*, int, int)
        _lib.pack_channels_to_u64.restype = ctypes.c_int
        _lib.pack_channels_to_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int, ctypes.c_int,
        ]

        # int threshold_i32_to_u8(int32*, uint8*, int, int, int)
        _lib.threshold_i32_to_u8.restype = ctypes.c_int
        _lib.threshold_i32_to_u8.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        # int xnor_multi_filter_conv(uint64*, uint64*, int32*, int, int, int, int, int, int)
        _lib.xnor_multi_filter_conv.restype = ctypes.c_int
        _lib.xnor_multi_filter_conv.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,   # n_ch, n_filters
        ]

        _lib.xnor_multi_filter_conv_zero_pad.restype = ctypes.c_int
        _lib.xnor_multi_filter_conv_zero_pad.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]

        # int float32_conv_nch_u8(uint8*, float*, float*, int, int, int, int, int)
        _lib.float32_conv_nch_u8.restype = ctypes.c_int
        _lib.float32_conv_nch_u8.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]

        # int float32_conv_nch_u64(uint64*, float*, float*, int, int, int, int, int)
        _lib.float32_conv_nch_u64.restype = ctypes.c_int
        _lib.float32_conv_nch_u64.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
    return _lib


def xnor_popcount_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int,
    kW: int,
) -> np.ndarray:
    """
    Run XNOR-Popcount convolution.

    Parameters
    ----------
    input_arr   : (rows, cols) uint8 — values must be 0 or 1
    weights_arr : (kH, kW)     int8  — values must be +1 or -1
    kH, kW      : kernel dimensions (must be odd)

    Returns
    -------
    output : (rows, cols) int32
    """
    lib = _load()
    kH, kW = _require_kernel(kH, kW)

    input_arr = _as_c_array("input_arr", input_arr, np.uint8, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.int8, ndim=2)
    _require_shape("weights_arr", weights_arr, (kH, kW))

    rows, cols = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output = np.empty((rows, cols), dtype=np.int32)

    rc = lib.xnor_popcount_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_int(kH),
        ctypes.c_int(kW),
    )
    if rc != 0:
        raise ValueError(f"xnor_popcount_conv returned error code {rc}")
    return output


def xnor_packed_u8_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """XNOR-Popcount convolution, channels packed in uint8 (n_ch ≤ 8)."""
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 8)
    input_arr   = _as_c_array("input_arr", input_arr, np.uint8, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.uint8, ndim=2)
    _require_shape("weights_arr", weights_arr, (kH, kW))
    rows, cols  = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output      = np.empty((rows, cols), dtype=np.int32)
    rc = lib.xnor_packed_u8_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"xnor_packed_u8_conv returned {rc}")
    return output


def pack_channels_to_u64(planes: np.ndarray, n_ch: int) -> np.ndarray:
    """
    Pack [n_ch, H, W] uint8 planes into [H, W] uint64.
    Any nonzero input value is treated as bit 1.
    Bit i = planes[i] at each pixel. Vectorised in C with -O3.
    """
    lib = _load()
    n_ch = _require_channels(n_ch, 64)
    planes = _as_c_array("planes", planes, np.uint8, ndim=3)
    if planes.shape[0] != n_ch:
        raise ValueError(f"planes first dimension must equal n_ch={n_ch}, got {planes.shape[0]}")
    _, H, W = planes.shape
    npix = _require_c_int_product("H*W", H, W)
    output = np.empty((H, W), dtype=np.uint64)
    rc = lib.pack_channels_to_u64(
        planes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int(n_ch),
        ctypes.c_int(npix),
    )
    if rc != 0:
        raise ValueError(f"pack_channels_to_u64 returned {rc}")
    return output


def threshold_i32_to_u8(scores: np.ndarray, threshold: int = 0) -> np.ndarray:
    """
    Binarize [n, H, W] int32 score maps: 1 if score > threshold else 0.
    Vectorised in C with -O3 (avoids large numpy temporaries).
    """
    lib = _load()
    threshold = _require_int("threshold", threshold, min_value=-_C_INT_MAX - 1, max_value=_C_INT_MAX)
    scores = _as_c_array("scores", scores, np.int32, ndim=3)
    n = scores.shape[0]
    npix = _require_c_int_product("H*W", scores.shape[1], scores.shape[2])
    output = np.empty_like(scores, dtype=np.uint8)
    rc = lib.threshold_i32_to_u8(
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(n),
        ctypes.c_int(npix),
        ctypes.c_int(threshold),
    )
    if rc != 0:
        raise ValueError(f"threshold_i32_to_u8 returned {rc}")
    return output


def xnor_multi_filter_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
    n_filters: int,
) -> np.ndarray:
    """
    Apply n_filters XNOR-Popcount filters in a single C call.

    Parameters
    ----------
    input_arr   : (rows, cols) uint64 — n_ch channels packed per pixel
    weights_arr : (n_filters, kH, kW) uint64 — n_ch weight bits per position
    kH, kW      : kernel dimensions (must be odd)
    n_ch        : number of active channels (1–64)
    n_filters   : number of filters

    Returns
    -------
    output : (n_filters, rows, cols) int32
    """
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 64)
    n_filters = _require_int("n_filters", n_filters, max_value=_C_INT_MAX)
    input_arr   = _as_c_array("input_arr", input_arr, np.uint64, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.uint64, ndim=3)
    _require_shape("weights_arr", weights_arr, (n_filters, kH, kW))
    rows, cols  = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output      = np.empty((n_filters, rows, cols), dtype=np.int32)
    rc = lib.xnor_multi_filter_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch), ctypes.c_int(n_filters),
    )
    if rc != 0:
        raise ValueError(f"xnor_multi_filter_conv returned {rc}")
    return output


def xnor_multi_filter_conv_zero_pad(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
    n_filters: int,
) -> np.ndarray:
    """
    Multi-filter XNOR-Popcount convolution with PyTorch-style zero padding.

    In-bounds binary values are interpreted as ±1. Out-of-bounds padding
    contributes 0, matching torch.nn.functional.conv2d(..., padding=...).
    """
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 64)
    n_filters = _require_int("n_filters", n_filters, max_value=_C_INT_MAX)
    input_arr = _as_c_array("input_arr", input_arr, np.uint64, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.uint64, ndim=3)
    _require_shape("weights_arr", weights_arr, (n_filters, kH, kW))
    rows, cols = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output = np.empty((n_filters, rows, cols), dtype=np.int32)
    rc = lib.xnor_multi_filter_conv_zero_pad(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH), ctypes.c_int(kW),
        ctypes.c_int(n_ch), ctypes.c_int(n_filters),
    )
    if rc != 0:
        raise ValueError(f"xnor_multi_filter_conv_zero_pad returned {rc}")
    return output


def xnor_packed_u64_conv(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """XNOR-Popcount convolution, channels packed in uint64 (n_ch ≤ 64)."""
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 64)
    input_arr   = _as_c_array("input_arr", input_arr, np.uint64, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.uint64, ndim=2)
    _require_shape("weights_arr", weights_arr, (kH, kW))
    rows, cols  = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output      = np.empty((rows, cols), dtype=np.int32)
    rc = lib.xnor_packed_u64_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"xnor_packed_u64_conv returned {rc}")
    return output


def float32_conv_nch_u8(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """Float32 reference convolution from packed uint8 input (n_ch ≤ 8)."""
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 8)
    input_arr   = _as_c_array("input_arr", input_arr, np.uint8, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.float32, ndim=3)
    _require_shape("weights_arr", weights_arr, (n_ch, kH, kW))
    rows, cols  = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output      = np.empty((rows, cols), dtype=np.float32)
    rc = lib.float32_conv_nch_u8(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"float32_conv_nch_u8 returned {rc}")
    return output


def float32_conv_nch_u64(
    input_arr: np.ndarray,
    weights_arr: np.ndarray,
    kH: int, kW: int,
    n_ch: int,
) -> np.ndarray:
    """Float32 reference convolution from packed uint64 input (n_ch ≤ 64)."""
    lib = _load()
    kH, kW = _require_kernel(kH, kW)
    n_ch = _require_channels(n_ch, 64)
    input_arr   = _as_c_array("input_arr", input_arr, np.uint64, ndim=2)
    weights_arr = _as_c_array("weights_arr", weights_arr, np.float32, ndim=3)
    _require_shape("weights_arr", weights_arr, (n_ch, kH, kW))
    rows, cols  = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output      = np.empty((rows, cols), dtype=np.float32)
    rc = lib.float32_conv_nch_u64(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        weights_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows), ctypes.c_int(cols),
        ctypes.c_int(kH),   ctypes.c_int(kW),
        ctypes.c_int(n_ch),
    )
    if rc != 0:
        raise ValueError(f"float32_conv_nch_u64 returned {rc}")
    return output


def sobel_conv(input_arr: np.ndarray) -> np.ndarray:
    """
    Run Sobel edge detection.

    Parameters
    ----------
    input_arr : (rows, cols) uint8 grayscale image

    Returns
    -------
    output : (rows, cols) float32 gradient magnitude
    """
    lib = _load()

    input_arr = _as_c_array("input_arr", input_arr, np.uint8, ndim=2)
    rows, cols = input_arr.shape
    _require_c_int_product("rows*cols", rows, cols)
    output = np.empty((rows, cols), dtype=np.float32)

    rc = lib.sobel_conv(
        input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
    )
    if rc != 0:
        raise ValueError(f"sobel_conv returned error code {rc}")
    return output
