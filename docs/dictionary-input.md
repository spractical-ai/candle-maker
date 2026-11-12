# Dictionary and datetime input

`render()` and `render_into()` accept the original numeric arrays, a list of
candle dictionaries for one trajectory, or a nested list of trajectories for
a batch. Batched trajectories must have the same number of candles. The output
remains contiguous uint8 RGB with shape `(B, H, W, 3)`.

Each dictionary requires numeric `open`, `high`, `low`, and `close` fields.
Key order does not matter; extra keys are ignored. Booleans, strings, complex
values, nonfinite prices, and inconsistent OHLC are rejected. The optional
public `Candle` TypedDict is available for type hints.

## Time and axes

- `time` is optional. If present, use a Python `datetime` object, not a string,
  date-only value, Unix timestamp, or `numpy.datetime64`.
- Within one trajectory, supply `time` on every candle or omit it from all.
  Different samples in a batch can independently have or omit timestamps.
- Naive datetimes display as supplied. Aware datetimes normalize to UTC.
  Mixing naive and aware values within one sample is rejected. Different aware
  timezones are supported; values outside the representable UTC range fail.
- Input order is preserved, including irregular gaps or reversed timestamps.
  Candle spacing remains equal; elapsed time does not stretch the plot.
- `use_axis=False` keeps the original axis-free pixels. Provided timestamps are
  still validated, so invalid data behaves consistently in both modes.
- With `use_axis=True`, timestamped samples get two-line `YYYY-MM-DD` / `HH:MM`
  labels. Seconds appear if any candle has nonzero seconds; six microsecond
  digits appear if any candle has nonzero microseconds. Small images use fewer
  ticks to prevent overlapping date labels; a single tick labels a middle candle.
- The total image dimensions stay fixed. Date labels reserve 32 bottom pixels
  instead of 22; the entire trajectory fits the remaining plot area. Minimum
  axis-enabled dimensions remain H >= 64 and W >= 112.

Timestamp formatting and dictionary-to-array conversion happen in the Python
adapter. Rust draws the candles, axes, and bitmap glyphs. Normalized labels are
owned by Rust while rendering; no Python datetime objects are borrowed by its
workers. `render_into()` validates the entire input before clearing output,
and can switch between dated, indexed, and axis-free batches without stale pixels.

## Matched benchmark

The dictionary benchmark repeats the same 256 distinct, 64-candle BTC windows,
batch size 256, two image sizes, and five shuffled rounds for:

1. NumPy input with candle-index axes, as a control.
2. Dictionaries without time, with candle-index axes.
3. Dictionaries with real datetime values, with date/time axes.

Record construction and CSV reading are outside timing. Dictionary validation,
NumPy normalization, selected datetime-label formatting, and output rendering
are inside candle-maker's timed calls. mplfinance's dictionary-to-DataFrame and
DatetimeIndex conversion and label formatting are also timed. The array control
uses prebuilt DataFrames, as in the earlier benchmark. Colors/styles are prepared
outside timing in every mode. Reused mplfinance retains figure/axes/canvas, and
both reused pipelines retain their output buffers.

Actual RGB output, `canvas.draw()`, RGBA-to-RGB copying, and normal disposal are
timed; PNG encoding, disk writes, imports, and forced GC between configurations
are excluded. Rust repeats the same corpus 16 times per round, normalized by all
images rendered. Default Rust worker pooling and one mplfinance process are used;
these are pipeline measurements, not language-wide speed claims. The dictionary
adapter adds CPU work on every call; contiguous float64 arrays remain the fast path.

The corpus's OHLC and selected timestamp hashes, Python-wrapper/native hashes,
versions, summaries, and every raw round are recorded. Fonts and individual pixels
can differ; previews check that fresh/reused mplfinance outputs agree after changing
samples and that every output has the expected dimensions and palette.

## Recorded results

Intel Core i7-1165G7, Windows 11, Python 3.12.3. Axes are enabled throughout;
arrays/plain dictionaries use index labels, timestamped dictionaries use date/time
labels. Image dimensions include margins; date labels need ten more bottom pixels.
Rates are rounded, while ratios use unrounded medians.

| Input | Image H x W | Output | mplfinance/s | candle-maker/s | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| NumPy | 128 x 128 | Fresh | 31.5 | 106,322 | 3,378x |
| NumPy | 128 x 128 | Reused | 64.2 | 353,030 | 5,499x |
| NumPy | 256 x 512 | Fresh | 26.5 | 20,991 | 793x |
| NumPy | 256 x 512 | Reused | 50.9 | 40,490 | 795x |
| Dictionaries | 128 x 128 | Fresh | 28.9 | 5,396 | 186x |
| Dictionaries | 128 x 128 | Reused | 60.1 | 5,867 | 98x |
| Dictionaries | 256 x 512 | Fresh | 26.5 | 4,629 | 175x |
| Dictionaries | 256 x 512 | Reused | 49.7 | 5,124 | 103x |
| Dictionaries + datetime | 128 x 128 | Fresh | 30.8 | 4,905 | 160x |
| Dictionaries + datetime | 128 x 128 | Reused | 57.1 | 5,158 | 90x |
| Dictionaries + datetime | 256 x 512 | Fresh | 20.5 | 3,862 | 189x |
| Dictionaries + datetime | 256 x 512 | Reused | 34.7 | 4,245 | 122x |

- [Array control](benchmarks/dictionary-array-results.json)
- [Dictionaries without time](benchmarks/dictionary-dict-results.json)
- [Dictionaries with datetime](benchmarks/dictionary-dict-time-results.json)
- [Combined summary CSV](benchmarks/dictionary-summary.csv)
- [Exact environment versions](benchmarks/python-dependencies.txt)
- [Datetime comparison image](assets/mplfinance-comparison-datetime.png)

## Re-run

In the clean environment described in the main README:

```sh
python examples/benchmark_mplfinance.py --use-axis --batches 256 --input-format array
python examples/benchmark_mplfinance.py --use-axis --batches 256 --input-format dict
python examples/benchmark_mplfinance.py --use-axis --batches 256 --input-format dict-time
```

Add `--csv path/to/prices.csv` to supply your dataset. Dictionary-time mode reads
ISO-format datetimes from the first CSV column. Numeric columns remain open,
high, low, close in columns 2-5. New output directories are suffixed with the
dictionary format, so these modes do not overwrite array reports.
