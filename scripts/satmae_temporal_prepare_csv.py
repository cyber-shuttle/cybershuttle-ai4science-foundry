#!/usr/bin/env python3
"""Rewrite SatMAE fMoW-Temporal CSV paths to a local data root.

Upstream CustomDatasetFromImagesTemporal treats column 1 as a path relative
to the directory that contains the CSV, and column 2 as an ISO-like timestamp.
This helper:

  * makes those relative paths point at $SATMAE_TEMPORAL_WORKDIR/data
  * reports how many image files actually exist on disk
  * optionally writes a class-stratified subset for the fallback eval

It does not invent labels or timestamps. Rows whose images are missing are
kept in the full prepared CSV (so you can see the gap) and dropped from a
subset unless --keep-missing is set.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _first_existing(candidates):
    for path in candidates:
        if path is not None and Path(path).is_file():
            return str(path)
    return None


def _name_variants(name: str):
    names = [name]
    if name.endswith("_rgb_crop_0.jpg"):
        names.append(name.replace("_rgb_crop_0.jpg", "_rgb.jpg"))
    elif name.endswith("_rgb.jpg") and "_msrgb" not in name:
        names.append(name.replace("_rgb.jpg", "_rgb_crop_0.jpg"))
    return names


def resolve_image_path(raw: str, data_root: Path) -> str:
    raw = str(raw).strip()
    if os.path.isabs(raw) and Path(raw).is_file():
        return raw

    name = Path(raw).name
    rel = raw.lstrip("./")
    rels = []
    for variant in _name_variants(name):
        rels.append(str(Path(rel).with_name(variant)) if rel else variant)
    candidates = []
    for item in rels:
        item_name = Path(item).name
        candidates.extend([
            data_root / item,
            data_root / "fmow" / item,
            data_root / Path(*Path(item).parts[1:]) if len(Path(item).parts) > 1 else None,
            data_root / "fmow" / "val" / Path(*Path(item).parts[2:]) if "val" in Path(item).parts else None,
        ])
        if "/val/" in item or item.startswith("val/"):
            tail = item.split("val/", 1)[-1]
            candidates.append(data_root / "fmow" / "val" / tail)
            candidates.append(data_root / "val" / tail)
        candidates.append(data_root / "fmow" / "val" / item_name)

    found = _first_existing(candidates)
    if found:
        return os.path.relpath(found, data_root)

    # Keep a normalized relative guess so the missing-file report is readable.
    if rel.startswith("fmow/"):
        return rel
    if rel.startswith("val/"):
        return str(Path("fmow") / rel)
    return str(Path("fmow") / "val" / name)


def stratified_subset(df: pd.DataFrame, n: int, seed: int, label_col: str) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    parts = []
    grouped = df.groupby(label_col, dropna=False)
    # Round-robin leftover so small classes still appear.
    base = max(1, n // max(df[label_col].nunique(), 1))
    for _, group in grouped:
        take = min(len(group), base)
        parts.append(group.sample(n=take, random_state=seed) if take < len(group) else group)
    out = pd.concat(parts, axis=0)
    if len(out) < n:
        remain = df.drop(out.index)
        extra = remain.sample(n=min(n - len(out), len(remain)), random_state=seed)
        out = pd.concat([out, extra], axis=0)
    return out.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Upstream val_62classes.csv")
    parser.add_argument("--data-root", required=True, help="Directory that will sit next to the prepared CSV")
    parser.add_argument("--out", required=True, help="Prepared CSV written here")
    parser.add_argument("--subset-out", default=None, help="Optional stratified subset CSV")
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-missing", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    df = pd.read_csv(args.csv, header=0)
    if df.shape[1] < 2:
        raise SystemExit(f"{args.csv} needs at least a label column and an image-path column")

    label_col = df.columns[0]
    path_col = df.columns[1]

    resolved = df[path_col].map(lambda p: resolve_image_path(p, data_root))
    df = df.copy()
    df[path_col] = resolved
    exists = [(data_root / p).is_file() or Path(p).is_file() for p in df[path_col]]
    df["_exists"] = exists

    n_ok = int(sum(exists))
    print(f"rows={len(df)} existing_images={n_ok} missing={len(df) - n_ok}")
    print(f"data_root={data_root}")
    if n_ok:
        print("example_ok", df.loc[df["_exists"], path_col].iloc[0])
    if n_ok < len(df):
        print("example_missing", df.loc[~df["_exists"], path_col].iloc[0])

    prepared = df.drop(columns=["_exists"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # CustomDatasetFromImagesTemporal joins paths onto dirname(csv).
    # Write the prepared CSV inside data_root so relative paths resolve.
    prepared.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    if args.subset_out:
        usable = df if args.keep_missing else df[df["_exists"]]
        usable = usable.drop(columns=["_exists"])
        if usable.empty:
            raise SystemExit("No existing images; cannot write a subset. Check the S3 download and path mapping.")
        subset = stratified_subset(usable, args.subset_size, args.seed, label_col)
        subset_path = Path(args.subset_out)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(subset_path, index=False)
        print(f"wrote subset n={len(subset)} labels={subset[label_col].nunique()} -> {subset_path}")


if __name__ == "__main__":
    main()
