"""Render OHLC trajectories into batches of RGB images using Rust."""

from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import render as _render
from ._native import render_into as _render_into

__all__ = ["render", "render_into"]


def _prepare_ohlc(ohlc: ArrayLike) -> NDArray[np.float64]:
    data = np.asarray(ohlc)
    if data.dtype.kind not in "fiu":
        raise TypeError("ohlc must contain real numeric values")
    if data.ndim == 2 and data.shape[-1] == 4:
        data = data[np.newaxis, ...]
    if data.ndim != 3 or data.shape[-1] != 4:
        raise ValueError("ohlc must have shape (B, N, 4) or (N, 4)")
    if data.shape[1] == 0:
        raise ValueError("each chart must contain at least one candle")
    # NumPy views backed by byte buffers may be contiguous but misaligned.
    # rust-numpy requires aligned storage for borrowed float64 slices.
    data = np.ascontiguousarray(data, dtype=np.float64)
    return data if data.flags.aligned else data.copy()


def render(ohlc: ArrayLike, height: int, width: int) -> NDArray[np.uint8]:
    """Render `(B, N, 4)` OHLC data to `(B, height, width, 3)` uint8 RGB.

    Columns are open, high, low, close; candles run left to right in input
    order. A single `(N, 4)` trajectory is treated as a batch of size one.
    Each chart scales independently. Rising/unchanged candles are green,
    falling candles red, and the background black. There are no axes.

    Values must be finite and satisfy low <= open/close <= high. At least
    one candle per chart and positive integer image dimensions are required.
    An empty batch is supported. Dense candles may share pixel columns.
    """
    if isinstance(height, (bool, np.bool_)) or isinstance(width, (bool, np.bool_)):
        raise TypeError("height and width must be positive integers")
    height, width = index(height), index(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")

    return _render(_prepare_ohlc(ohlc), height, width)


def render_into(ohlc: ArrayLike, out: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Render into an existing `(B, H, W, 3)` C-contiguous uint8 RGB buffer.

    Returns the same array. Dimensions are inferred from `out`. The whole
    buffer is cleared to black before drawing, so it can be reused for the
    next batch. Invalid OHLC leaves `out` unchanged. Values and appearance
    follow `render()`. Input and output must not overlap.

    Consume or copy the images before reusing this buffer. Do not access or
    modify the input/output from another thread during this call. The GIL
    is retained while Rust workers fill caller-owned memory.
    """
    if not isinstance(out, np.ndarray) or out.dtype != np.uint8:
        raise TypeError("out must be a uint8 NumPy array")
    if out.ndim != 4 or out.shape[-1] != 3:
        raise ValueError("out must have shape (B, H, W, 3)")
    if out.shape[1] == 0 or out.shape[2] == 0:
        raise ValueError("output height and width must be positive")
    if not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError("out must be C-contiguous and writable")
    data = _prepare_ohlc(ohlc)
    if out.shape[0] != data.shape[0]:
        raise ValueError("out batch size must match ohlc")
    if np.shares_memory(data, out):
        raise ValueError("ohlc and out must not overlap")
    _render_into(data, out)
    return out
