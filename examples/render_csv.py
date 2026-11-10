"""Render consecutive windows from a CSV with open, high, low, close columns."""

import argparse
import csv
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image

from candle_maker import render

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=ROOT / "sample_data/BTCUSDT-1h.csv")
    parser.add_argument("--candles", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if args.candles < 1 or args.batch_size < 1:
        parser.error("--candles and --batch-size must be positive")
    count = args.batch_size * args.candles
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = [[float(row[key]) for key in ("open", "high", "low", "close")] for row in islice(reader, count)]
    if len(rows) != count:
        parser.error(f"need {count} candles, found {len(rows)}")
    data = np.array(rows, dtype=np.float64).reshape(args.batch_size, args.candles, 4)
    images = render(data, height=args.height, width=args.width)
    args.output.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        path = args.output / f"chart_{i}.png"
        Image.fromarray(image).save(path)
        print(path)
    print(f"Rendered {data.shape} -> {images.shape}, dtype={images.dtype}")


if __name__ == "__main__":
    main()
