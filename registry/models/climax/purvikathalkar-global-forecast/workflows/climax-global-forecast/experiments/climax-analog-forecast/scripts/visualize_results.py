#!/usr/bin/env python
"""Create a compact weighted-RMSE comparison chart from metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = metrics["results"]
    labels = []
    values = []
    for row in rows:
        label = row["method"] if row["k"] is None else "{} k={}".format(row["method"], row["k"])
        labels.append(label)
        values.append(float(row["weighted_rmse"]))
    maximum = max(values) if values else 1.0
    width = 760
    left = 190
    chart_width = 540
    row_height = 36
    height = 70 + row_height * len(rows)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 40 + index * row_height
        parts.append('<text x="10" y="{}" font-family="sans-serif" font-size="13">{}</text>'.format(y + 13, label))
        bar_width = chart_width * value / maximum
        parts.append('<rect x="{}" y="{}" width="{:.2f}" height="18" fill="#3a86a8"/>'.format(left, y, bar_width))
        parts.append('<text x="{:.2f}" y="{}" font-family="sans-serif" font-size="12">{:.4f}</text>'.format(left + bar_width + 6, y + 14, value))
    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
