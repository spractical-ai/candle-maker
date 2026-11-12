"""Measure end-to-end rendering, including input snapshot and output allocation."""

import argparse
from statistics import median
from time import perf_counter

import numpy as np

from candle_maker import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candles", type=int, default=64)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--use-axis", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.candles, args.height, args.width, args.repeats) <= 0:
        parser.error("all arguments must be positive")
    rng = np.random.default_rng(42)
    opens = 100 + rng.normal(size=(args.batch_size, args.candles)).cumsum(axis=1)
    closes = opens + rng.normal(size=opens.shape)
    highs = np.maximum(opens, closes) + rng.random(opens.shape)
    lows = np.minimum(opens, closes) - rng.random(opens.shape)
    data = np.stack([opens, highs, lows, closes], axis=-1)
    render(data, args.height, args.width, use_axis=args.use_axis)  # Warm up the native thread pool.
    times = []
    for _ in range(args.repeats):
        start = perf_counter()
        images = render(data, args.height, args.width, use_axis=args.use_axis)
        times.append(perf_counter() - start)
        del images
    elapsed = median(times)
    print(f"{data.shape} -> ({args.batch_size}, {args.height}, {args.width}, 3)")
    print(f"Median: {elapsed * 1000:.3f} ms/batch; {args.batch_size / elapsed:,.0f} charts/s")


if __name__ == "__main__":
    main()
