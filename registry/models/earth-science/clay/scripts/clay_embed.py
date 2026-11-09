#!/usr/bin/env python3
"""Embed 256 x 256 NAIP chips with Clay v1.5.

For each chip, Clay returns a list of 1,024 numbers that summarizes it (the
"class" embedding at the end of the encoder, the same kind Clay published).

Which chips: exactly the chips in Clay's reference parquet (--reference), in
the same row order. Each chip's pixel window is found by projecting its
outline onto the tile's pixel grid, and the run stops if any outline does not
land on whole pixels. So output row i always matches reference row i.

Model inputs per chip (see build_inputs()), prepared the way Clay prepared the
data behind its published NAIP embeddings (Clay-foundation/stacchip,
stacchip/processors/prechip.py @ 5660f1a, lines 152-158):
  pixels  4 bands red, green, blue, nir, normalized with the NAIP mean/std
          from Clay's configs/metadata.yaml
  time    [sin week, cos week, sin hour, cos hour]; week = ISO week * 2pi/52,
          hour = 12 * 2pi/24 (NAIP has no time of day; Clay assumes noon)
  latlon  [sin lon, cos lon, sin lat, cos lat] of the chip center, in radians
  gsd     1.0 (metadata.yaml); waves = NAIP wavelengths in micrometers

Output (--out, Parquet), one row per chip:
  chip_index, row_off, col_off   position in the reference / the tile
  geometry                       chip outline (WKB, lon/lat) computed here
  embeddings                     1,024 float32 numbers
Run details (inputs, versions, GPU, runtime, peak memory) are stored in the
file metadata under "clay_run" and in <out>.json.

--dry-run prepares every input without importing torch or loading the model,
so the chip matching and preprocessing can be tested on any machine.

Example (GPU node, env active, from the Clay model folder):
  python scripts/clay_embed.py \\
      --tile $CLAY_WORKDIR/data/m_3211002_se_12_1_20150530.tif \\
      --reference $CLAY_WORKDIR/data/m_3211002_se_12_1_20150530.parquet \\
      --checkpoint $CLAY_WORKDIR/weights/clay-v1.5.ckpt \\
      --metadata $CLAY_WORKDIR/clay-model/configs/metadata.yaml \\
      --out $CLAY_WORKDIR/results/<run>/embeddings.parquet
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
import yaml
from pyproj import Transformer
from rasterio.windows import Window

CHIP = 256                      # pixels; Clay v1.5 input size
PATCH = 8                       # Clay v1.5 patch size -> (256/8)^2 = 1024 patches
EMBED_DIM = 1024
DOLLS = [16, 32, 64, 128, 256, 768, 1024]   # as in Clay's examples for v1.5
HOUR = 12                       # NAIP has no time of day; Clay's data preparation assumes noon
MAX_PIXEL_ERROR = 0.01          # how far a reference outline may sit off the pixel grid


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile", required=True, type=Path, help="NAIP GeoTIFF (bands red, green, blue, nir)")
    p.add_argument("--reference", required=True, type=Path, help="Clay's published embeddings for the same tile")
    p.add_argument("--metadata", required=True, type=Path, help="Clay configs/metadata.yaml (made absolute)")
    p.add_argument("--checkpoint", type=Path, help="clay-v1.5.ckpt (not needed with --dry-run)")
    p.add_argument("--out", type=Path, help="output .parquet (a .json with run details is written next to it)")
    p.add_argument("--limit", type=int, help="only the first N chips (smoke test)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="prepare inputs only; no torch, no model")
    args = p.parse_args(argv)
    if not args.dry_run and (args.checkpoint is None or args.out is None):
        p.error("--checkpoint and --out are required unless --dry-run")
    return args


def read_reference(path):
    """Chip outlines and file metadata (including the acquisition date) of Clay's parquet."""
    pf = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
    geoms = shapely.from_wkb(pf.read(columns=["geometry"]).column("geometry").to_pylist())
    if "date" not in meta:
        raise SystemExit(f"ERROR: {path} has no 'date' in its metadata")
    date = dt.datetime.strptime(meta["date"], "%Y%m%d").date()
    return geoms, meta, date


def load_naip_metadata(path):
    naip = yaml.safe_load(Path(path).read_text())["naip"]
    bands = naip["band_order"]
    return {
        "bands": bands,
        "mean": np.array([naip["bands"]["mean"][b] for b in bands], dtype=np.float32),
        "std": np.array([naip["bands"]["std"][b] for b in bands], dtype=np.float32),
        "waves": [float(naip["bands"]["wavelength"][b]) for b in bands],
        "gsd": float(naip["gsd"]),
    }


def match_chips(geoms, src):
    """Pixel window (col_off, row_off) of each reference outline in the tile."""
    to_tile = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    inverse = ~src.transform
    windows, worst = [], 0.0
    for i, g in enumerate(geoms):
        lon, lat = np.asarray(g.exterior.coords).T
        x, y = to_tile.transform(lon, lat)
        cols, rows = inverse * (np.asarray(x), np.asarray(y))
        c0, r0 = float(cols.min()), float(rows.min())
        col, row = round(c0), round(r0)
        err = max(abs(c0 - col), abs(r0 - row),
                  abs(float(cols.max()) - c0 - CHIP), abs(float(rows.max()) - r0 - CHIP))
        if err > MAX_PIXEL_ERROR or col < 0 or row < 0 or col + CHIP > src.width or row + CHIP > src.height:
            raise SystemExit(f"ERROR: reference chip {i} is not a {CHIP}x{CHIP} window on the tile grid "
                             f"(col {c0:.3f}, row {r0:.3f}, error {err:.4f} px)")
        windows.append((col, row))
        worst = max(worst, err)
    return windows, worst


def chip_polygon(src, col, row, to_lonlat):
    """Chip outline in lon/lat, built the same way as Clay's chip outlines."""
    x0, y0 = src.transform * (col, row)
    x1, y1 = src.transform * (col + CHIP, row + CHIP)
    box = shapely.box(x0, y0, x1, y1)
    lon, lat = to_lonlat.transform(*np.asarray(box.exterior.coords).T)
    return shapely.Polygon(np.column_stack([lon, lat]))


def encode_time(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = HOUR * 2 * np.pi / 24
    return [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)]


def encode_latlon(poly):
    """[sin lon, cos lon, sin lat, cos lat] of the center of the chip's bounds."""
    minx, miny, maxx, maxy = poly.bounds
    lon, lat = math.radians((minx + maxx) / 2), math.radians((miny + maxy) / 2)
    return [math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)]


def build_inputs(src, windows, polys, date, meta):
    """Yield (pixels [C,H,W] float32, time [4], latlon [4]) for each chip."""
    t = np.array(encode_time(date), dtype=np.float32)
    for (col, row), poly in zip(windows, polys, strict=True):
        raw = src.read(window=Window(col, row, CHIP, CHIP))
        if raw.shape != (len(meta["bands"]), CHIP, CHIP):
            raise SystemExit(f"ERROR: window ({col}, {row}) read shape {raw.shape}")
        # Same arithmetic as torchvision v2.Normalize(mean, std) in Clay's examples.
        px = (raw.astype(np.float32) - meta["mean"][:, None, None]) / meta["std"][:, None, None]
        yield px, t, np.array(encode_latlon(poly), dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    out = {"python": sys.version.split()[0]}
    for pkg in ["torch", "claymodel", "timm", "lightning", "numpy", "rasterio", "pyproj", "shapely", "pyarrow"]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def load_encoder(checkpoint, metadata_path, device):
    import torch
    from claymodel.module import ClayMAEModule   # importing this sets matmul precision to "medium"

    module = ClayMAEModule.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        model_size="large",
        metadata_path=str(metadata_path),
        dolls=DOLLS,
        doll_weights=[1] * len(DOLLS),
        mask_ratio=0.0,     # no masking at inference
        shuffle=False,      # keep patch order
    )                       # strict=True: any missing/unexpected weight raises
    module.eval()
    encoder = module.model.encoder.to(device).eval()
    info = {
        "teacher": module.hparams.get("teacher"),
        "model_size": module.hparams.get("model_size"),
        "mask_ratio": module.hparams.get("mask_ratio"),
        "checkpoint_load": "strict (no missing or unexpected keys)",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    return encoder, info


def run_model(encoder, inputs, n, meta, device, batch_size):
    import torch

    gsd = torch.tensor(meta["gsd"], device=device)
    waves = torch.tensor(meta["waves"], device=device)
    out = np.empty((n, EMBED_DIM), dtype=np.float32)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start, done, batch = time.perf_counter(), 0, []

    def flush():
        nonlocal done
        px, t, ll = (np.stack(x) for x in zip(*batch, strict=True))
        cube = {
            "pixels": torch.from_numpy(px).to(device),
            "time": torch.from_numpy(t).to(device),
            "latlon": torch.from_numpy(ll).to(device),
            "gsd": gsd,
            "waves": waves,
        }
        with torch.no_grad():
            encoded, *_ = encoder(cube)
        expected = (len(batch), 1 + (CHIP // PATCH) ** 2, EMBED_DIM)
        if tuple(encoded.shape) != expected:
            raise SystemExit(f"ERROR: encoder output {tuple(encoded.shape)}, expected {expected}")
        out[done:done + len(batch)] = encoded[:, 0, :].float().cpu().numpy()   # class token
        done += len(batch)
        batch.clear()

    for item in inputs:
        batch.append(item)
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    stats = {"inference_seconds": round(time.perf_counter() - start, 3)}
    if device.startswith("cuda"):
        stats["peak_cuda_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        stats["gpu"] = torch.cuda.get_device_name(0)
        stats["gpu_compute_capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
    return out, stats


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    meta = load_naip_metadata(args.metadata)
    geoms, ref_meta, date = read_reference(args.reference)

    with rasterio.open(args.tile) as src:
        if src.count != len(meta["bands"]) or src.dtypes[0] != "uint8":
            raise SystemExit(f"ERROR: expected a {len(meta['bands'])}-band uint8 NAIP tile, got "
                             f"{src.count} bands of {src.dtypes[0]}")
        windows, worst = match_chips(geoms, src)
        if args.limit:
            geoms, windows = geoms[:args.limit], windows[:args.limit]
        to_lonlat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        polys = [chip_polygon(src, c, r, to_lonlat) for c, r in windows]
        geometry_gap = max(shapely.hausdorff_distance(a, b) for a, b in zip(polys, geoms, strict=True))
        inputs = build_inputs(src, windows, polys, date, meta)

        run = {
            "script": "scripts/clay_embed.py",
            "command": sys.argv if argv is None else ["clay_embed.py", *argv],
            "host": socket.gethostname(),
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "inputs": {"date": date.isoformat(), "iso_week": date.isocalendar().week, "hour": HOUR,
                       "latlon": "[sin lon, cos lon, sin lat, cos lat] of the chip center",
                       "normalized": True, "gsd": meta["gsd"], "waves_um": meta["waves"]},
            "n_chips": len(windows),
            "chip_grid_max_error_px": worst,
            "geometry_max_gap_deg": geometry_gap,
            "tile": {"path": str(args.tile.resolve()), "sha256": sha256(args.tile), "crs": str(src.crs),
                     "width": src.width, "height": src.height},
            "reference": {"path": str(args.reference.resolve()), "sha256": sha256(args.reference),
                          "metadata": {k: v for k, v in ref_meta.items() if k != "geo"}},
            "metadata_yaml": str(args.metadata.resolve()),
            "versions": versions(),
            "platform": platform.platform(),
        }
        print(f"chips: {len(windows)}  grid error: {worst:.2e} px  geometry gap vs reference: {geometry_gap:.2e} deg")
        print("inputs:", json.dumps(run["inputs"]))

        if args.dry_run:
            stacked = list(inputs)
            px = np.stack([x[0] for x in stacked])
            run["dry_run"] = {
                "pixels_shape": list(px.shape),
                "pixels_mean_per_band": px.mean(axis=(0, 2, 3)).round(4).tolist(),
                "pixels_std_per_band": px.std(axis=(0, 2, 3)).round(4).tolist(),
                "time_first": stacked[0][1].tolist(),
                "latlon_first": stacked[0][2].tolist(),
            }
            print(json.dumps(run["dry_run"], indent=2))
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.with_suffix(".json").write_text(json.dumps(run, indent=2) + "\n")
            print("DRY RUN OK")
            return

        encoder, model_info = load_encoder(args.checkpoint, args.metadata.resolve(), args.device)
        run["model"] = dict(model_info, checkpoint=str(args.checkpoint.resolve()))
        print("model:", json.dumps(run["model"]))
        emb, stats = run_model(encoder, inputs, len(windows), meta, args.device, args.batch_size)

    if not np.isfinite(emb).all():
        raise SystemExit("ERROR: embeddings contain NaN or Inf")
    run.update(stats)
    run["batch_size"] = args.batch_size
    run["total_seconds"] = round(time.perf_counter() - t0, 3)

    table = pa.table({
        "chip_index": pa.array(range(len(windows)), pa.int32()),
        "row_off": pa.array([r for _, r in windows], pa.int32()),
        "col_off": pa.array([c for c, _ in windows], pa.int32()),
        "geometry": pa.array(shapely.to_wkb(polys).tolist(), pa.binary()),
        "embeddings": pa.array(list(emb), pa.list_(pa.float32())),
    }).replace_schema_metadata({"clay_run": json.dumps(run)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"wrote {args.out}  shape {emb.shape}  {stats}")
    print("EMBED COMPLETED")


if __name__ == "__main__":
    main()
