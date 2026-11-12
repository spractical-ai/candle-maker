# Candle Maker

Fast candlestick images for Python, rendered by a Rust engine. Designed for
generating batches of model inputs: black backgrounds, green/red candles,
and optional price/index axes. No GPU or plotting framework is needed.

Pass **`(B, N, 4)` OHLC data** and receive **`(B, H, W, 3)` RGB images**.
Each image contains the whole trajectory of its sample, from left to right.
This is a focused raster renderer, not a full mplfinance API replacement.

## Install

Requires **Python 3.10+**, **Rust 1.83+**, and a native compiler/linker.
On Windows, install Visual Studio C++ Build Tools with a Windows SDK.
Windows x64 / CPython 3.12 has been tested; Linux and macOS are not yet verified.

```sh
git clone https://github.com/spractical-ai/candle-maker.git
cd candle-maker
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` on Windows
PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install .
```

This compiles the Rust extension in release mode and installs NumPy automatically.
From another Python project, use `python -m pip install /path/to/candle-maker`.
The distribution name is `candle-maker`; the Python import is `candle_maker`.

To distribute without requiring Rust on the receiving machine, install the
development extras and run `maturin build --release --locked`. Install the
resulting wheel from `target/wheels/` on a matching Python version, OS, and CPU
architecture with `python -m pip install path/to/package.whl`.

## Use

```python
import numpy as np
from candle_maker import render, render_into

# Two samples, three candles each. Columns: open, high, low, close.
ohlc = np.array([
    [[10, 14, 9, 13], [13, 15, 10, 11], [11, 16, 10, 15]],
    [[20, 23, 18, 19], [19, 24, 17, 22], [22, 25, 20, 21]],
], dtype=np.float64)

images = render(ohlc, height=256, width=512)
assert images.shape == (2, 256, 512, 3)
assert images.dtype == np.uint8

# Add price ticks and candle indices; the image dimensions stay the same.
with_axis = render(ohlc, height=256, width=512, use_axis=True)

# For repeated batches with fixed B/H/W, allocate once.
out = np.empty((2, 256, 512, 3), dtype=np.uint8)
render_into(ohlc, out, use_axis=True)  # Returns out; H/W come from its shape.
# Consume out before rendering the next batch into it.
```

Input may be numeric array-like data; a single `(N, 4)` sample also works.
Use contiguous `float64` input to avoid conversion overhead. Output is always
C-contiguous RGB: black `(0, 0, 0)`, green `(0, 255, 0)` for `close >= open`,
and red `(255, 0, 0)` otherwise. Each sample scales independently using its
lowest low and highest high, with approximately 5% vertical padding.

Both functions also accept a **list of candle dictionaries** for one trajectory,
or a nested list for a batch with the same number of candles per sample:

```python
from datetime import datetime

candles = [
    {"open": 10, "high": 14, "low": 9, "close": 13, "time": datetime(2024, 1, 1, 9, 30)},
    {"open": 13, "high": 15, "low": 10, "close": 11, "time": datetime(2024, 1, 1, 10, 30)},
]
dated_images = render(candles, height=256, width=512, use_axis=True)
dated_out = np.empty_like(dated_images)
render_into(candles, dated_out, use_axis=True)
```

`open`, `high`, `low`, and `close` are required numeric keys; extra keys are
ignored. Optional `time` must be a Python `datetime`, supplied for all or no
candles within each sample. With axes enabled, timestamps produce date/time
labels; without them, labels remain candle indices. Naive datetimes display as
supplied; aware datetimes display in UTC. Input order and equal candle spacing
are preserved. See [datetime rules and input benchmarks](docs/dictionary-input.md).

Both functions default to **`use_axis=False`**, preserving the original pixels.
Set **`use_axis=True`** for gray price ticks on the left and zero-based candle
indices along the bottom, or date/time labels for timestamped dictionaries.
The Rust engine draws a built-in 5x7 bitmap font; no font files or
extra Python packages are needed. Labels use compact decimal/scientific notation.
Axis mode requires **H >= 64, W >= 112** and reserves 64/8/8/22 pixels for
left/right/top/bottom margins (32 bottom pixels for date/time labels) within the
requested image dimensions. The entire
trajectory fits in the remaining plot area; small images have denser candles.

Values must be finite and satisfy `low <= open/close <= high`. At least one
candle and positive image dimensions are required; empty batches are supported.
When there are more candles than image columns, candles can overlap.

`render()` returns independent storage. `render_into()` clears and overwrites
its writable, C-contiguous `uint8` output on every call; consume or copy the
previous images first. Invalid OHLC leaves that buffer unchanged. Input/output
must not overlap or be accessed concurrently during rendering. Output memory
is `B * H * W * 3` bytes, so split large jobs into batches.

To save a preview, install Pillow and use
`Image.fromarray(images[0]).save("chart.png")` after `from PIL import Image`.

## Similar charts, faster generation

The same 64-candle BTC trajectory, rendered at **256 x 512 (H x W)**,
first without axes:

![Rust and mplfinance renders of the same OHLC trajectory](docs/assets/mplfinance-comparison.png)

And with `use_axis=True`:

![Rust and mplfinance renders with price and candle-index axes](docs/assets/mplfinance-comparison-axes.png)

Timestamped dictionaries use date/time labels instead:

![Rust and mplfinance renders with datetime axes](docs/assets/mplfinance-comparison-datetime.png)

Left: candle-maker. Center: mplfinance with fresh figures. Right: mplfinance
with a reused figure/canvas. Appearance is closely matched; the renderers
are **not pixel-identical**, including differences in font and tick rasterization.

Earlier array-only throughput in **images/second**, batch size 256, median of five rounds:

| Axes | Image H x W | Output mode | mplfinance | candle-maker | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Off | 128 x 128 | Fresh | 53 | 118,921 | 2,225x |
| Off | 128 x 128 | Reused | 177 | 352,259 | 1,990x |
| Off | 256 x 512 | Fresh | 53 | 21,067 | 396x |
| Off | 256 x 512 | Reused | 150 | 39,108 | 260x |
| On | 128 x 128 | Fresh | 31 | 105,200 | 3,388x |
| On | 128 x 128 | Reused | 63 | 318,457 | 5,023x |
| On | 256 x 512 | Fresh | 26 | 20,978 | 810x |
| On | 256 x 512 | Reused | 52 | 39,258 | 760x |

Measured on an Intel Core i7-1165G7, Windows 11, Python 3.12.3; 256 distinct
windows of 64 candles, matched colors/dimensions/padding, antialiasing disabled.
Both engines produce actual contiguous RGB batches. Reused modes preallocate
output; mplfinance also retains its figure, axes, and canvas. Rust uses Rayon
workers; mplfinance runs in one process. With axes and one Rust worker, reused
output at 256 x 512 was still about **683x** faster than mplfinance.

These numbers describe this OHLC-to-RGB pipeline, not general Rust/Python
performance or training speed. Imports, data preparation, PNG encoding, and
disk I/O are excluded. Hardware, resolution, and batch size affect throughput.
See [benchmark details, versions, and raw timings](docs/benchmarks/README.md).

Current input-format comparison, **axes on and output reused**, with the same
256 samples, 64 candles, batch size 256, and five-round medians:

| Input | Image H x W | mplfinance/s | candle-maker/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| NumPy | 128 x 128 | 64.2 | 353,030 | 5,499x |
| NumPy | 256 x 512 | 50.9 | 40,490 | 795x |
| Dictionaries | 128 x 128 | 60.1 | 5,867 | 98x |
| Dictionaries | 256 x 512 | 49.7 | 5,124 | 103x |
| Dictionaries + datetime | 128 x 128 | 57.1 | 5,158 | 90x |
| Dictionaries + datetime | 256 x 512 | 34.7 | 4,245 | 122x |

Dictionary timings include validation and conversion on every call in both
engines; records are constructed before timing. The array control uses prebuilt
DataFrames. Numeric arrays remain the fast path.
[Full fresh/reused results and method](docs/dictionary-input.md).

## Where the speed comes from

- **Direct rasterization:** Rust draws bodies, wicks, and optional bitmap labels
  into RGB memory, avoiding general plotting objects, text layout, and an
  RGBA-to-RGB copy.
- **Batching and parallelism:** one Python call handles many samples; Rayon
  distributes larger batches across CPU workers.
- **Less memory work:** `render()` transfers its pixel allocation to NumPy
  without another pixel copy. `render_into()` reuses storage between batches.
- **Cached geometry and release builds:** candle positions are cached per
  calling thread for up to 4,096 candles; thin LTO optimizes the native code.

`render()` snapshots input and releases the Python GIL while Rust works.
`render_into()` retains the GIL while borrowing caller-owned memory; its Rust
workers still draw in parallel. Reusing memory accelerates image generation;
the effect on training depends on the rest of your pipeline.

## Develop and benchmark

Inside the activated environment, with Rust available:

```sh
python -m pip install ".[dev]"
maturin develop --release --locked
python -m pytest
cargo test --locked
python examples/benchmark.py
```

For the mplfinance comparison:

```sh
python -m pip install ".[benchmark]"
python examples/benchmark_mplfinance.py --csv path/to/prices.csv
python examples/benchmark_mplfinance.py --csv path/to/prices.csv --use-axis
python examples/benchmark_mplfinance.py --csv path/to/prices.csv --use-axis --input-format dict-time
```

The comparison CSV needs a header, then timestamp/open/high/low/close as its
first five columns. Both engines receive the same axis setting. Results and
previews go to ignored `output/mplfinance_comparison/` or `output/mplfinance_axes/`.
`--input-format dict` benchmarks dictionaries without timestamps; `dict-time`
uses ISO-format timestamps from the CSV's first column. Dictionary modes time
input conversion in both engines and use separate output directories.
Use `--rayon-threads 1 --engines rust rust_reuse` for the single-worker control.
`examples/benchmark_batch_sizes.py` sweeps batch sizes and supports `--reuse-output`.
`examples/render_csv.py path/to/prices.csv --use-axis` saves PNG previews (requires lowercase
`open`, `high`, `low`, `close` column names and Pillow).

Datasets in `sample_data/`, build products, and local toolchains are intentionally
ignored. Supply your own CSV; the basic usage example and synthetic benchmark
need no dataset. `scripts/use-local-rust.ps1` is an optional helper for a toolchain
already installed in `.tools/`; normal Rust installations do not need it.
