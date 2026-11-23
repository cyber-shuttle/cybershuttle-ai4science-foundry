#!/usr/bin/env python3
"""Rebuild SatMAE's unpublished *_rgb_crop_0.jpg chips from raw fMoW-RGB.

SatMAE val_62classes.csv and CustomDatasetFromImagesTemporal both assume
filenames ending in `_rgb_crop_0.jpg` (the loader globs the last 15
characters). The public SpaceNet fMoW-RGB dump only has full scenes
(`*_rgb.jpg`) plus a sidecar JSON whose `bounding_boxes[0].box` is
`[x, y, width, height]`.

This writes the first AOI crop next to each RGB scene so the official CSV
paths and temporal glob keep working. It does not invent labels.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

_OVERWRITE = False


def _init_worker(overwrite: bool) -> None:
    global _OVERWRITE
    _OVERWRITE = overwrite
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_one(rgb_path: str) -> str:
    src = Path(rgb_path)
    dst = src.with_name(src.name.replace("_rgb.jpg", "_rgb_crop_0.jpg"))
    if dst.is_file() and not _OVERWRITE:
        return "exists"

    meta = src.with_name(src.name.replace("_rgb.jpg", "_rgb.json"))
    if not meta.is_file():
        return "no_json"
    try:
        info = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return "bad_json"

    boxes = info.get("bounding_boxes") or []
    if not boxes:
        return "no_box"
    box = boxes[0].get("box")
    if not box or len(box) < 4:
        return "no_box"

    x, y, w, h = (int(round(float(v))) for v in box[:4])
    if w <= 0 or h <= 0:
        return "empty_box"

    try:
        with Image.open(src) as im:
            left = max(0, x)
            upper = max(0, y)
            right = min(im.width, x + w)
            lower = min(im.height, y + h)
            if right <= left or lower <= upper:
                return "empty_box"
            crop = im.crop((left, upper, right, lower)).convert("RGB")
            crop.save(dst, format="JPEG", quality=95)
    except (OSError, ValueError):
        return "bad_image"
    return "wrote"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True, help="Directory of fMoW-RGB val scenes")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    val_root = Path(args.val_root).resolve()
    rgb_files = sorted(str(p) for p in val_root.rglob("*_rgb.jpg") if "_msrgb" not in p.name)
    if not rgb_files:
        raise SystemExit(f"no *_rgb.jpg under {val_root}")

    counts = {
        "wrote": 0,
        "exists": 0,
        "no_json": 0,
        "bad_json": 0,
        "no_box": 0,
        "empty_box": 0,
        "bad_image": 0,
    }
    workers = max(1, args.workers)
    print(f"val_root={val_root} n_rgb={len(rgb_files)} workers={workers}", flush=True)

    def consume(statuses):
        for i, status in enumerate(statuses, 1):
            counts[status] = counts.get(status, 0) + 1
            if i == 1 or i % 1000 == 0 or i == len(rgb_files):
                print(f"progress {i}/{len(rgb_files)} {counts}", flush=True)

    if workers == 1:
        _init_worker(args.overwrite)
        consume(crop_one(path) for path in rgb_files)
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=(args.overwrite,)) as pool:
            consume(pool.imap_unordered(crop_one, rgb_files, chunksize=16))

    print("crop_counts", counts, flush=True)
    if counts["wrote"] + counts["exists"] == 0:
        raise SystemExit("no AOI crops written")


if __name__ == "__main__":
    main()
