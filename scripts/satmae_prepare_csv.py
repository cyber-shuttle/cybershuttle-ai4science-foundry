#!/usr/bin/env python3
"""Rebuild the `image_path` column of an fMoW-Sentinel metadata CSV.

The CSVs shipped with fMoW-Sentinel (val.csv / train.csv) have an
`image_path` column that does not point at the real on-disk layout.
The correct path, per the upstream SatMAE README, is:

    fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif

This script rewrites that column (relative to --data-root) and optionally
writes a class-stratified subset for a quick fallback/sanity run.
"""
import argparse
import os

import pandas as pd


def build_image_path(split: str, category: str, location_id, image_id) -> str:
    base = f"{category}_{location_id}"
    return os.path.join(
        "fmow-sentinel", split, category, base, f"{base}_{image_id}.tif"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to input val.csv / train.csv")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                         help="Split name used in the rebuilt path (default: val)")
    parser.add_argument("--data-root", default="data",
                         help="Root dir the rebuilt image_path will be joined to at load "
                              "time (not written into the csv itself, just used to validate "
                              "files exist unless --skip-exists-check is set)")
    parser.add_argument("--out", required=True, help="Output csv path")
    parser.add_argument("--subset", type=int, default=None,
                         help="If set, write a class-stratified random subset of N rows total "
                              "(approximately N/num_classes per class) instead of the full csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-exists-check", action="store_true",
                         help="Skip verifying that rebuilt image files exist on disk")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"category", "location_id", "image_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")

    df["image_path"] = [
        build_image_path(args.split, cat, loc, img)
        for cat, loc, img in zip(df["category"], df["location_id"], df["image_id"])
    ]

    if not args.skip_exists_check:
        sample = df.sample(min(20, len(df)), random_state=args.seed)
        missing_files = [
            p for p in sample["image_path"]
            if not os.path.exists(os.path.join(args.data_root, p))
        ]
        if missing_files:
            print(f"WARNING: {len(missing_files)}/{len(sample)} sampled rebuilt paths do not "
                  f"exist under --data-root={args.data_root}. Example: {missing_files[0]}\n"
                  "Double-check --split and that the imagery has been downloaded/extracted.")

    if args.subset is not None:
        n_classes = df["category"].nunique()
        per_class = max(1, args.subset // n_classes)
        df = (
            df.groupby("category", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=args.seed))
            .reset_index(drop=True)
        )
        print(f"Stratified subset: requested {args.subset}, got {len(df)} rows "
              f"across {n_classes} classes (~{per_class}/class)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
