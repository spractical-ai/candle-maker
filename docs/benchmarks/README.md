# Recorded mplfinance comparison

These are the measurements quoted in the main README, recorded on Windows 11
with an Intel Core i7-1165G7 (4 cores / 8 logical CPUs) and Python 3.12.3.
They compare a specialized candle rasterizer with mplfinance's public plotting
API, including a version that reuses its figure, axes, and Agg canvas.

## Method

- Select 256 distinct, evenly distributed 64-candle windows from the local
  68,149-row BTCUSDT hourly CSV. Every engine receives the same price values.
- Render at 128 x 128 and 256 x 512, with batches 1, 32, and 256; report the
  median throughput across five shuffled rounds after warm-up.
- Match black backgrounds, pure green/red bodies, approximately 70% body width,
  one-pixel wicks, price limits, and padding. Disable antialiasing and all axes,
  text, grid, and volume. The pixels are similar, not identical.
- Fresh modes allocate a new RGB batch; mplfinance creates/closes each figure.
  Reused modes allocate and first-touch output before timing; mplfinance also
  retains its figure/axes/canvas, replacing the candle collections each time.
- Include validation, drawing, normal output disposal, and, for mplfinance,
  `canvas.draw()` plus the RGBA-to-RGB copy. Exclude imports, CSV reading,
  DataFrame/date/color/style preparation, PNG encoding, and disk writes.
- Keep automatic garbage collection enabled; force collection between
  configurations outside timing. Deferred cyclic disposal can favor mplfinance.
- Repeat the corpus 16 times per Rust round to stabilize short timings;
  normalize by all images actually rendered. mplfinance uses one pass per round.
- Use Rust's default Rayon worker pool and one mplfinance process. Run a
  separate single-worker Rust control. This does not test multiprocessing or
  every possible Matplotlib optimization.

## Evidence

- [Full results](mplfinance-results.json): metadata, visual checks, summaries,
  native/input hashes, and all per-round timings.
- [Single-worker control](single-worker-results.json).
- [Summary CSV](summary.csv).
- [Exact Python dependency versions](python-dependencies.txt). Install the local
  candle-maker source separately; the benchmark used its 0.1.0 release wheel.
- [Comparison image](../assets/mplfinance-comparison.png): saved directly by the
  benchmark, with Rust / mplfinance fresh / mplfinance reused from left to right.

## Re-run

Use a clean Python environment and the Rust build prerequisites described in
the main README. The original dataset is local and intentionally not published;
provide a CSV with timestamp/open/high/low/close as its first five columns.
The recorded input hash identifies the original corpus, so another dataset
will produce a different hash and potentially different timings.

```sh
python -m pip install -r docs/benchmarks/python-dependencies.txt
python -m pip install .
python examples/benchmark_mplfinance.py --csv path/to/prices.csv
python examples/benchmark_mplfinance.py --csv path/to/prices.csv --engines rust rust_reuse --rayon-threads 1 --output output/mplfinance_single_worker
```

Alternatively, install `".[benchmark]"` for compatible dependency ranges rather
than the exact recorded versions. New reports, timings, and previews are saved
under `output/`; the checked-in evidence is not overwritten.
