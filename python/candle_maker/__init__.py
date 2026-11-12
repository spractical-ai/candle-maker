"""Render OHLC trajectories into batches of RGB images using Rust."""

from operator import index
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._native import render as _render
from ._native import render_into as _render_into

__all__ = ["Candle", "render", "render_into"]


class _CandleFields(TypedDict):
    open: float
    high: float
    low: float
    close: float


class Candle(_CandleFields, total=False):
    time: datetime


OHLCInput = ArrayLike | Sequence[Candle] | Sequence[Sequence[Candle]]
_MISSING = object()
_PRICE_KEYS = ("open", "high", "low", "close")


def _is_record_input(ohlc) -> bool:
    if not isinstance(ohlc, (list, tuple)) or not ohlc:
        return False
    first = ohlc[0]
    return isinstance(first, Mapping) or (
        isinstance(first, (list, tuple)) and bool(first) and isinstance(first[0], Mapping)
    )


def _prepare_records(ohlc):
    samples = [ohlc] if isinstance(ohlc[0], Mapping) else ohlc
    prices, times = [], []
    candles = len(samples[0])
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)):
            raise TypeError(f"sample {sample_index} must be a sequence of candle dictionaries")
        if len(sample) != candles:
            raise ValueError("dictionary batches must have the same number of candles per sample")
        sample_prices, sample_times = [], []
        awareness = None
        for candle_index, candle in enumerate(sample):
            where = f"sample {sample_index}, candle {candle_index}"
            if not isinstance(candle, Mapping):
                raise TypeError(f"{where} must be a candle dictionary")
            try:
                values = tuple(candle[key] for key in _PRICE_KEYS)
            except KeyError as error:
                raise ValueError(f"{where} is missing required key {error.args[0]!r}") from None
            if any(isinstance(value, (bool, np.bool_)) or
                   not isinstance(value, (int, float, np.integer, np.floating)) for value in values):
                raise TypeError(f"{where}: OHLC values must be real numbers")
            sample_prices.append(values)
            value = candle.get("time", _MISSING)
            if value is not _MISSING:
                if not isinstance(value, datetime):
                    raise TypeError(f"{where}: time must be a datetime object")
                aware = value.utcoffset() is not None
                if awareness is not None and awareness != aware:
                    raise ValueError(f"sample {sample_index}: do not mix naive and timezone-aware datetimes")
                awareness = aware
                if aware:
                    try:
                        value = value.astimezone(timezone.utc)
                    except (OverflowError, ValueError) as error:
                        raise ValueError(f"{where}: time cannot be represented in UTC") from error
                sample_times.append(value)
        if sample_times and len(sample_times) != candles:
            raise ValueError(f"sample {sample_index}: supply time for every candle or omit it for all candles")
        prices.append(sample_prices)
        times.append(sample_times or None)
    return _prepare_ohlc(prices), times


def _time_ticks(times, width):
    result = []
    for sample in times:
        if sample is None:
            result.append(None)
            continue
        microseconds = any(value.microsecond for value in sample)
        seconds = microseconds or any(value.second for value in sample)
        time_chars = 15 if microseconds else 8 if seconds else 5
        label_width = max(10, time_chars) * 6 - 1
        count = min(5, max(1, (width - 72) // (label_width + 12) + 1), len(sample))
        indices = [(len(sample) - 1) // 2] if count == 1 else [
            (len(sample) - 1) * i // (count - 1) for i in range(count)
        ]
        ticks = []
        for candle_index in indices:
            value = sample[candle_index]
            date = f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
            clock = f"{value.hour:02d}:{value.minute:02d}"
            if seconds:
                clock += f":{value.second:02d}"
            if microseconds:
                clock += f".{value.microsecond:06d}"
            ticks.append((candle_index, date + "\n" + clock))
        result.append(ticks)
    return result if any(sample is not None for sample in result) else None


def _prepare_input(ohlc, width, use_axis):
    if _is_record_input(ohlc):
        data, times = _prepare_records(ohlc)
        return data, _time_ticks(times, width) if use_axis else None
    return _prepare_ohlc(ohlc), None


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


def _prepare_axis(use_axis: bool) -> bool:
    if not isinstance(use_axis, (bool, np.bool_)):
        raise TypeError("use_axis must be a boolean")
    return bool(use_axis)


def render(ohlc: OHLCInput, height: int, width: int, *, use_axis: bool = False) -> NDArray[np.uint8]:
    """Render `(B, N, 4)` OHLC data to `(B, height, width, 3)` uint8 RGB.

    Columns are open, high, low, close; candles run left to right in input
    order. A single `(N, 4)` trajectory is treated as a batch of size one.
    Each chart scales independently. Rising/unchanged candles are green,
    falling candles red, and the background black. Set use_axis=True for
    price ticks on the left and zero-based candle indices along the bottom.
    Axes use fixed bitmap labels and require height >= 64 and width >= 112.
    A list of open/high/low/close dictionaries is one sample; a nested list
    is a batch. Optional datetime time values replace index labels with date
    and time labels. Supply time for all or no candles within each sample.
    Aware datetimes display in UTC; naive datetimes display as supplied.

    Values must be finite and satisfy low <= open/close <= high. At least
    one candle per chart and positive integer image dimensions are required.
    An empty batch is supported. Dense candles may share pixel columns.
    """
    if isinstance(height, (bool, np.bool_)) or isinstance(width, (bool, np.bool_)):
        raise TypeError("height and width must be positive integers")
    height, width = index(height), index(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")

    use_axis = _prepare_axis(use_axis)
    data, time_ticks = _prepare_input(ohlc, width, use_axis)
    return _render(data, height, width, use_axis, time_ticks)


def render_into(ohlc: OHLCInput, out: NDArray[np.uint8], *, use_axis: bool = False) -> NDArray[np.uint8]:
    """Render into an existing `(B, H, W, 3)` C-contiguous uint8 RGB buffer.

    Returns the same array. Dimensions are inferred from `out`. The whole
    buffer is cleared to black before drawing, so it can be reused for the
    next batch. Invalid OHLC leaves `out` unchanged. Values and appearance
    follow `render()`, including optional price/index axes via use_axis=True.
    Input and output must not overlap.

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
    use_axis = _prepare_axis(use_axis)
    data, time_ticks = _prepare_input(ohlc, out.shape[2], use_axis)
    if out.shape[0] != data.shape[0]:
        raise ValueError("out batch size must match ohlc")
    if np.shares_memory(data, out):
        raise ValueError("ohlc and out must not overlap")
    _render_into(data, out, use_axis, time_ticks)
    return out
