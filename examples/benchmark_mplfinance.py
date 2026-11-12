"""Compare real RGB rendering against mplfinance, with fresh and reused buffers."""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/mplfinance-mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from candle_maker import _native, render, render_into
from benchmark_batch_sizes import load_windows

DPI = 100
ENGINES = ["mplfinance", "mplfinance_reuse", "rust", "rust_reuse"]
MARKET_COLORS = mpf.make_marketcolors(
    up="#00ff00", down="#ff0000", edge="inherit", wick="inherit", alpha=1.0,
)
STYLE = mpf.make_mpf_style(
    base_mpf_style="classic", marketcolors=MARKET_COLORS,
    facecolor="black", figcolor="black", gridstyle="",
    rc={"axes.grid": False, "patch.antialiased": False, "lines.antialiased": False,
        "text.antialiased": False,
        "font.weight": "normal", "axes.labelweight": "normal", "figure.titleweight": "normal"},
)


def format_price(value):
    """Match the Rust renderer's compact numeric labels (font shapes differ)."""
    if value == 0:
        return "0"
    label = f"{value:.2f}".rstrip("0").rstrip(".")
    if 0.01 <= abs(value) < 1e6 and len(label) <= 9:
        return label
    for precision in [2, 1]:
        mantissa, exponent = f"{value:.{precision}e}".split("e")
        label = f"{mantissa}e{int(exponent)}"
        if len(label) <= 9:
            return label
    return label


class MpfRenderer:
    def __init__(self, height, width, reuse, use_axis=False):
        self.height, self.width, self.reuse, self.use_axis = height, width, reuse, use_axis
        self.figure = self.axis = self.canvas = None
        if reuse:
            self.figure = Figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor="black")
            self.canvas = FigureCanvasAgg(self.figure)
            self.axis = self.figure.add_axes([0, 0, 1, 1], facecolor="black")

    def draw(self, frame, colors, out):
        candles = len(frame)
        low, high = frame["Low"].min(), frame["High"].max()
        plot_height = self.height - 30 if self.use_axis else self.height
        plot_width = self.width - 72 if self.use_axis else self.width
        padding = min(int(plot_height * 0.05), (plot_height - 1) // 2)
        span = plot_height - 1 - 2 * padding
        margin = (high - low) * (padding + 0.5) / max(span, 1)
        ylim = (float(low - margin), float(high + margin)) if low != high else (float(low - 1), float(high + 1))
        kwargs = dict(
            type="candle", style=STYLE, volume=False, axisoff=not self.use_axis,
            xlim=(-0.5, candles - 0.5), ylim=ylim,
            update_width_config={"candle_width": 0.7, "candle_linewidth": 72 / DPI},
            marketcolor_overrides=colors, warn_too_much_data=candles + 1,
        )
        figure = None
        try:
            if self.reuse:
                figure, axis, canvas = self.figure, self.axis, self.canvas
                # Keep the expensive axes/tick objects. Only replace the two
                # candle collections; mpf.plot receives explicit limits below.
                for collection in list(axis.collections):
                    collection.remove()
                axis.dataLim = Bbox.null()
                mpf.plot(frame, ax=axis, **kwargs)
            else:
                figure, axes = mpf.plot(
                    frame, figsize=(self.width / DPI, self.height / DPI),
                    returnfig=True, scale_padding=0, **kwargs,
                )
                figure.set_dpi(DPI)
                figure.set_size_inches(self.width / DPI, self.height / DPI, forward=False)
                axis, canvas = axes[0], figure.canvas
                for item in axes:
                    item.set_position([0, 0, 1, 1])
                    item.set_axis_off()
            figure.set_facecolor("black")
            axis.set_facecolor("black")
            axis.set_axis_off()
            axis.grid(False)
            axis.set_xlim(-0.5, candles - 0.5)
            axis.set_ylim(*ylim)
            if self.use_axis:
                axis.set_axis_on()
                axis.set_position([64 / self.width, 22 / self.height,
                                   plot_width / self.width, plot_height / self.height])
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.yaxis.tick_left()
                axis.xaxis.tick_bottom()
                for name, spine in axis.spines.items():
                    spine.set_visible(name in ["left", "bottom"])
                    spine.set_color("#c8c8c8")
                    spine.set_linewidth(72 / DPI)
                    spine.set_antialiased(False)
                count = min(5, max(2, plot_width // 48 + 1), candles)
                indices = [0] if count == 1 else [(candles - 1) * i // (count - 1) for i in range(count)]
                axis.set_xticks(indices, [str(index) for index in indices])
                count = 1 if low == high else min(5, max(2, plot_height // 24 + 1))
                values = [low] if count == 1 else [high * (1 - i / (count - 1)) + low * (i / (count - 1)) for i in range(count)]
                axis.set_yticks(values, [format_price(value) for value in values])
                axis.tick_params(axis="both", colors="#c8c8c8", labelsize=6,
                                 length=3 * 72 / DPI, width=72 / DPI, pad=3 * 72 / DPI,
                                 labelrotation=0, top=False, right=False)
                for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
                    label.set_fontfamily("DejaVu Sans Mono")
                    label.set_fontweight("normal")
                    label.set_antialiased(False)
                for line in [*axis.get_xticklines(), *axis.get_yticklines()]:
                    line.set_antialiased(False)
            for collection in axis.collections:
                collection.set_antialiased(False)
                if isinstance(collection, LineCollection):
                    collection.set_linewidth(72 / DPI)  # One-pixel wick.
                    collection.set_capstyle("butt")
                elif isinstance(collection, PolyCollection):
                    collection.set_linewidth(0)  # Filled bodies, no extra outline.
            canvas.draw()  # Mandatory: creating artists alone is not rendering.
            rgba = np.asarray(canvas.buffer_rgba())
            if rgba.shape != (self.height, self.width, 4):
                raise AssertionError(f"unexpected Agg buffer shape: {rgba.shape}")
            np.copyto(out, rgba[:, :, :3])  # Own contiguous RGB, not a canvas view.
        finally:
            if figure is not None and not self.reuse:
                plt.close(figure)

    def batch(self, frames, colors, out):
        for index, (frame, palette) in enumerate(zip(frames, colors)):
            self.draw(frame, palette, out[index])


def measure(engine, data, frames, colors, batch, height, width, renderer, passes, use_axis=False):
    chunks = [(start, start + batch) for start in range(0, len(data), batch)]
    reused = engine.endswith("_reuse")
    out = np.empty((batch, height, width, 3), dtype=np.uint8) if reused else None
    if reused:
        if engine == "rust_reuse":
            render_into(data[:batch], out, use_axis=use_axis)
        else:
            renderer.batch(frames[:batch], colors[:batch], out)
    # Collect old Matplotlib cycles before timing, so they cannot be charged to
    # the next engine. Automatic GC remains enabled during every measurement.
    gc.collect()
    start = perf_counter()
    for _ in range(passes):
        for first, last in chunks:
            if engine == "rust":
                images = render(data[first:last], height, width, use_axis=use_axis)
            elif engine == "rust_reuse":
                images = render_into(data[first:last], out, use_axis=use_axis)
            else:
                images = out if reused else np.empty((batch, height, width, 3), dtype=np.uint8)
                renderer.batch(frames[first:last], colors[first:last], images)
            del images
    elapsed = perf_counter() - start
    return {"seconds": elapsed, "charts": len(data) * passes,
            "charts_per_second": len(data) * passes / elapsed,
            "mean_batch_ms": elapsed * 1000 / (len(chunks) * passes)}


def preview(data, frames, colors, height, width, output, use_axis=False):
    rust_image = render(data[:1], height, width, use_axis=use_axis)[0]
    fresh = np.empty_like(rust_image)
    reused = np.empty_like(rust_image)
    MpfRenderer(height, width, False, use_axis).draw(frames[0], colors[0], fresh)
    reused_renderer = MpfRenderer(height, width, True, use_axis)
    # Verify that changing prices/labels and then reusing the canvas does not
    # retain old artists. This check and preview saving are outside timing.
    reused_renderer.draw(frames[-1], colors[-1], reused)
    reused_renderer.draw(frames[0], colors[0], reused)
    np.testing.assert_array_equal(fresh, reused)
    images = [rust_image, fresh, reused]
    labels = ["Rust", "mplfinance fresh", "mplfinance reused"]
    checks = []
    allowed = {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
    if use_axis:
        allowed.add((200, 200, 200))
    required = allowed.copy()
    if use_axis:
        # Agg covers the endpoints of tick markers by half a pixel even when
        # antialiasing is disabled; retain those actual pixels in the comparison.
        allowed.add((100, 100, 100))
    for label, image in zip(labels, images):
        palette = {tuple(map(int, value)) for value in np.unique(image.reshape(-1, 3), axis=0)}
        assert image.shape == (height, width, 3)
        assert image.dtype == np.uint8 and image.flags.c_contiguous
        assert required <= palette <= allowed, (label, palette)
        assert not image[0].any() and not image[-1].any(), label
        if use_axis:
            gray = np.all(image == (200, 200, 200), axis=-1)
            assert gray[8:height - 22, :57].any() and gray[height - 15:, 64:].any(), label
        checks.append({"engine": label, "shape": list(image.shape), "colors": [list(c) for c in sorted(palette)],
                       "matches_rust_pixels": bool(np.array_equal(image, rust_image))})
        Image.fromarray(image).save(output / f"{label.replace(' ', '_')}_{height}x{width}.png")
    contact = Image.new("RGB", (3 * width + 32, height + 36), (32, 32, 32))
    draw = ImageDraw.Draw(contact)
    font = ImageFont.load_default(size=12)
    for i, (label, image) in enumerate(zip(labels, images)):
        x = 8 + i * (width + 8)
        draw.text((x, 5), label, fill="white", font=font)
        contact.paste(Image.fromarray(image), (x, 28))
    contact.save(output / f"visual_comparison_{height}x{width}.png")
    return checks


def save_plot(rows, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, (height, width) in zip(axes, sorted({(r["height"], r["width"]) for r in rows})):
        for engine in ENGINES:
            series = [r for r in rows if r["engine"] == engine and r["height"] == height and r["width"] == width]
            if series:
                axis.plot([r["batch_size"] for r in series], [r["charts_per_second"] for r in series], "o-", label=engine)
        axis.set(xscale="log", yscale="log", xlabel="Batch size", ylabel="Charts / second", title=f"{height}×{width} RGB")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Rust vs mplfinance • same 64-candle BTC windows • real RGB output")
    figure.savefig(output / "performance.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--candles", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rust-passes", type=int, default=16, help="repeat the same corpus to reduce noise for the faster engines")
    parser.add_argument("--engines", choices=ENGINES, nargs="+", default=ENGINES)
    parser.add_argument("--rayon-threads", type=int)
    parser.add_argument("--use-axis", action="store_true", help="draw matched price/index axes in both engines")
    parser.add_argument("--csv", type=Path, default=ROOT / "sample_data/BTCUSDT-1h.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("output/mplfinance_axes" if args.use_axis else "output/mplfinance_comparison")
    if min(args.samples, args.candles, args.rounds, args.rust_passes, *args.batches) < 1:
        parser.error("sizes and repeat counts must be positive")
    if any(args.samples % b for b in args.batches):
        parser.error("samples must be divisible by every batch size")
    if args.rayon_threads is not None:
        if args.rayon_threads < 1:
            parser.error("rayon-threads must be positive")
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    data, csv_rows = load_windows(args.csv, args.candles, args.samples)
    dates = pd.date_range("2000-01-01", periods=args.candles, freq="h")
    # Input adapters and static styles are prepared once, outside timed regions.
    frames = [pd.DataFrame(sample, columns=["Open", "High", "Low", "Close"], index=dates) for sample in data]
    colors = [["#00ff00" if close >= open_ else "#ff0000" for open_, _, _, close in sample] for sample in data]
    resolutions = [(128, 128), (256, 512)]
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "numpy": np.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__, "mplfinance": mpf.__version__,
        "backend": matplotlib.get_backend(), "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "samples": args.samples, "candles": args.candles, "csv_rows": csv_rows,
        "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "rounds": args.rounds, "rust_passes": args.rust_passes,
        "use_axis": args.use_axis,
        "scope": "real RGB batches, including canvas.draw(), RGBA-to-RGB copy, and normal figure close; automatic GC enabled",
        "excluded": "imports, CSV reading, window selection, DataFrame/DatetimeIndex/color/style preparation, preview saving, PNG encoding, disk writes, forced GC between configurations",
        "fresh": "both engines allocate independent B,H,W,3 outputs per call; mplfinance makes/closes a figure per image",
        "reuse": "both engines reuse output; mplfinance also reuses its figure, external axes, and Agg canvas; only old candle collections are removed before replotting",
        "threads": "Rust uses configured Rayon pool; mplfinance runs in one process without a multiprocessing wrapper",
        "rust_repetitions": f"Rust repeats the identical {args.samples}-window corpus more times per round for stable timing; throughput is normalized by charts actually rendered",
        "appearance": "black, pure green/red, antialiasing disabled, no grid/volume, 70% candle bodies, one-pixel wicks, matching x/price limits and vertical padding",
        "axes": "gray left price and bottom zero-based candle-index ticks; matching tick values/counts, 64/8/8/22-pixel left/right/top/bottom margins; Rust 5x7 bitmap font vs Matplotlib 6pt DejaVu Sans Mono" if args.use_axis else "no axes or labels",
        "rasterization": "same requested appearance and exact output dimensions; Matplotlib and Rust need not produce bit-identical pixels",
    }
    visual = {f"{h}x{w}": preview(data, frames, colors, h, w, args.output, args.use_axis) for h, w in resolutions}
    renderers = {(h, w, engine): MpfRenderer(h, w, engine.endswith("_reuse"), args.use_axis)
                 for h, w in resolutions for engine in args.engines if engine.startswith("mplfinance")}
    configurations = [(h, w, batch, engine) for h, w in resolutions for batch in args.batches for engine in args.engines]
    records = {config: [] for config in configurations}
    rng = random.Random(91)
    print(f"{len(configurations)} configurations × {args.rounds} rounds; {args.samples} distinct windows; versions: mpf={mpf.__version__}, mpl={matplotlib.__version__}", flush=True)
    for round_index in range(args.rounds):
        order = configurations.copy()
        rng.shuffle(order)
        for h, w, batch, engine in order:
            passes = args.rust_passes if engine.startswith("rust") else 1
            result = measure(engine, data, frames, colors, batch, h, w, renderers.get((h, w, engine)), passes, args.use_axis)
            records[h, w, batch, engine].append({"round": round_index + 1, **result})
        print(f"Round {round_index + 1}/{args.rounds} complete", flush=True)
    rows, raw = [], []
    for (h, w, batch, engine), runs in records.items():
        rates = [r["charts_per_second"] for r in runs]
        rows.append({"height": h, "width": w, "batch_size": batch, "engine": engine,
                     "charts_per_second": median(rates), "median_mean_batch_ms": median(r["mean_batch_ms"] for r in runs),
                     "throughput_p10": float(np.percentile(rates, 10)), "throughput_p90": float(np.percentile(rates, 90))})
        raw.extend({"height": h, "width": w, "batch_size": batch, "engine": engine, **r} for r in runs)
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "visual_checks": visual, "summary": rows, "rounds": raw}, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f'{row["height"]}x{row["width"]} B={row["batch_size"]:3} {row["engine"]:18} {row["charts_per_second"]:12,.1f} charts/s {row["median_mean_batch_ms"]:10.3f} ms/batch', flush=True)
    save_plot(rows, args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
