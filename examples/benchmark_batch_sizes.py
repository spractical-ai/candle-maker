"""Compare batch sizes using fixed work and shuffled, repeated measurements."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from candle_maker import _native, render, render_into

ROOT = Path(__file__).resolve().parents[1]


def load_windows(path, candles, count):
    prices = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4))
    if prices.ndim != 2 or len(prices) < candles + count - 1:
        raise ValueError("CSV has too few rows for the requested distinct windows")
    windows = np.lib.stride_tricks.sliding_window_view(prices, candles, axis=0)
    indices = np.linspace(0, len(windows) - 1, count, dtype=np.intp)
    return np.ascontiguousarray(windows[indices].transpose(0, 2, 1)), len(prices)


def measure(data, batch, height, width, render_function, out=None):
    chunks = [data[start : start + batch] for start in range(0, len(data), batch)]
    calls_ms = []
    start = perf_counter()
    for chunk in chunks:
        call_start = perf_counter()
        images = render_function(chunk, height, width) if out is None else render_into(chunk, out)
        del images  # Include disposal; never retain two image batches at once.
        calls_ms.append((perf_counter() - call_start) * 1000)
    return perf_counter() - start, median(calls_ms)


def save_plot(rows, path, candles):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping plot: install matplotlib to generate it.", flush=True)
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for height, width in sorted({(row["height"], row["width"]) for row in rows}):
        series = [row for row in rows if row["height"] == height and row["width"] == width]
        batches = [row["batch_size"] for row in series]
        throughput = [row["charts_per_second"] / 1000 for row in series]
        label = f"{height}×{width} RGB"
        line, = axes[0].plot(batches, throughput, "o-", label=label)
        axes[0].fill_between(
            batches,
            [row["throughput_p10"] / 1000 for row in series],
            [row["throughput_p90"] / 1000 for row in series],
            color=line.get_color(), alpha=0.14,
        )
        axes[1].plot(batches, [row["median_batch_ms"] for row in series], "o-", label=label)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Batch size (charts per call)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Throughput (thousand charts / second)")
    axes[0].set_title("Fixed workload; shading: 10th–90th percentile of rounds")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median call + disposal latency (ms)")
    axes[1].set_title("Time before the next batch can start")
    figure.suptitle(f"Rust candlestick pipeline • {candles} candles per chart • warmed release build")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "sample_data/BTCUSDT-1h.csv")
    parser.add_argument("--candles", type=int, default=64)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--batches", type=int, nargs="+", default=[2**i for i in range(12)])
    parser.add_argument("--resolutions", nargs="+", default=["128x128", "256x512"])
    parser.add_argument("--output", type=Path, default=ROOT / "output/batch_size_benchmark")
    parser.add_argument("--reuse-output", action="store_true", help="measure render_into with preallocated output")
    parser.add_argument("--baseline-package", type=Path, help="directory containing a saved package __init__.py and native extension")
    args = parser.parse_args()
    render_function, native = render, _native
    if args.baseline_package:
        if args.reuse_output:
            parser.error("baseline-package and reuse-output cannot be combined")
        package = args.baseline_package.resolve()
        spec = importlib.util.spec_from_file_location(
            "candle_benchmark_baseline", package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        render_function, native = baseline.render, baseline._native
    if min(args.candles, args.samples, args.rounds, *args.batches) <= 0:
        parser.error("candles, samples, rounds, and batch sizes must be positive")
    if any(args.samples % batch for batch in args.batches):
        parser.error("samples must be divisible by every batch size")
    resolutions = [tuple(map(int, size.lower().split("x"))) for size in args.resolutions]
    if any(len(size) != 2 or min(size) <= 0 for size in resolutions):
        parser.error("resolutions must be positive HxW pairs")
    data, csv_rows = load_windows(args.csv, args.candles, args.samples)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS", "default"),
        "native_extension": native.__file__,
        "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "csv": str(args.csv), "csv_rows": csv_rows,
        "candles_per_chart": args.candles, "charts_per_pass": args.samples,
        "rounds": args.rounds, "input_dtype": str(data.dtype),
        "input_contiguous": data.flags.c_contiguous,
        "window_selection": "distinct overlapping windows evenly spaced across CSV",
        "mode": "render_into" if args.reuse_output else "render",
        "timed": "render_into() including buffer clearing, without allocation/disposal" if args.reuse_output else "render() including disposal; per-pass loop and timer overhead included",
        "excluded": "CSV loading, window preparation, warmup, PNG encoding, disk writes",
        "memory_metric": "output buffer bytes, excludes input, Rust snapshot and allocator overhead",
    }
    try:
        import psutil

        metadata["physical_memory_bytes"] = psutil.virtual_memory().total
        metadata["available_memory_before_bytes"] = psutil.virtual_memory().available
        metadata["physical_cpu_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    configurations = [(height, width, batch) for height, width in resolutions for batch in sorted(set(args.batches))]
    for height, width, batch in configurations:
        # Warm up allocation paths and the thread pool outside measured rounds.
        for _ in range(3):
            images = render_function(data[:batch], height, width)
            assert images.shape == (batch, height, width, 3)
            assert images.dtype == np.uint8
            del images
    print(f"Testing {len(configurations)} configurations × {args.rounds} rounds; {args.samples} charts/pass", flush=True)
    records = {config: [] for config in configurations}
    rng = random.Random(42)
    for round_number in range(args.rounds):
        order = configurations.copy()
        rng.shuffle(order)
        for height, width, batch in order:
            out = None
            if args.reuse_output:
                out = np.empty((batch, height, width, 3), dtype=np.uint8)
                render_into(data[:batch], out)  # Allocation and first-touch outside timing.
            seconds, batch_ms = measure(data, batch, height, width, render_function, out)
            del out
            records[height, width, batch].append({
                "round": round_number + 1, "pass_seconds": seconds,
                "charts_per_second": args.samples / seconds,
                "median_batch_ms": batch_ms,
            })
        print(f"Round {round_number + 1}/{args.rounds} complete", flush=True)
    rows, raw = [], []
    for (height, width, batch), runs in records.items():
        throughput = [run["charts_per_second"] for run in runs]
        rows.append({
            "height": height, "width": width, "batch_size": batch,
            "median_batch_ms": median(run["median_batch_ms"] for run in runs),
            "median_pass_ms": 1000 * median(run["pass_seconds"] for run in runs),
            "charts_per_second": median(throughput),
            "throughput_p10": float(np.percentile(throughput, 10)),
            "throughput_p90": float(np.percentile(throughput, 90)),
            "output_mib": batch * height * width * 3 / (1024**2),
        })
        raw.extend({"height": height, "width": width, "batch_size": batch, **run} for run in runs)
    result = {"metadata": metadata, "summary": rows, "rounds": raw}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("H×W         Batch   Batch ms    Charts/sec    Output MiB", flush=True)
    for row in rows:
        print(f'{row["height"]}×{row["width"]:<7} {row["batch_size"]:5} {row["median_batch_ms"]:10.3f} {row["charts_per_second"]:13,.0f} {row["output_mib"]:13.2f}', flush=True)
    save_plot(rows, args.output / "batch_size_performance.png", args.candles)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
