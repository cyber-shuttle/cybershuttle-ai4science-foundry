#!/usr/bin/env python
"""Prepare WeatherBench geopotential NetCDF files for single-variable forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


VARIABLE_NAME = "geopotential_500"
HOURS_PER_YEAR = 8760
GRID_SHAPE = (32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Extracted geopotential_5.625deg or geopotential_500_5.625deg NetCDF directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-start-year", type=int, default=2013)
    parser.add_argument("--val-start-year", type=int, default=2015)
    parser.add_argument("--test-start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2017, help="Exclusive end year")
    parser.add_argument("--num-shards", type=int, default=8)
    return parser.parse_args()


def find_netcdf_files(input_dir: Path) -> List[Path]:
    if input_dir.is_file():
        files = [input_dir] if input_dir.suffix.lower() in {".nc", ".nc4", ".netcdf"} else []
    elif input_dir.is_dir():
        files = sorted(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".nc", ".nc4", ".netcdf"}
            and path.stem.lower() != "constants"
        )
    else:
        raise FileNotFoundError("Geopotential input path does not exist: {}".format(input_dir))

    if not files:
        raise FileNotFoundError(
            "No geopotential NetCDF files found under {}".format(input_dir)
        )
    return files


def _coordinate_name(names: List[str], options: Tuple[str, ...], role: str) -> str:
    for name in options:
        if name in names:
            return name
    raise ValueError("Could not identify the {} coordinate (tried {})".format(role, options))


def select_geopotential_500(dataset: xr.Dataset) -> Tuple[xr.DataArray, str, str, str]:
    """Select the 500 hPa field from either a pressure-level or single-level file."""
    variable_name = next(
        (name for name in ("z", "geopotential", VARIABLE_NAME) if name in dataset.data_vars),
        None,
    )
    if variable_name is None:
        raise KeyError(
            "NetCDF must contain one of the geopotential fields: z, geopotential, geopotential_500"
        )

    field = dataset[variable_name]
    level_dim = next(
        (name for name in ("level", "pressure_level", "plev") if name in field.dims),
        None,
    )
    if level_dim is not None:
        if level_dim not in dataset.coords:
            raise ValueError("Pressure dimension {!r} has no coordinate values".format(level_dim))
        levels = np.asarray(dataset[level_dim].values, dtype=np.float64)
        matches = np.flatnonzero(np.isclose(levels, 500.0))
        if matches.size != 1:
            raise ValueError(
                "Expected exactly one 500 hPa level in {!r}; found levels {}".format(
                    level_dim, levels.tolist()
                )
            )
        field = field.isel({level_dim: int(matches[0])}, drop=True)

    time_dim = _coordinate_name(list(field.dims), ("time", "valid_time"), "time")
    lat_name = _coordinate_name(list(field.dims), ("lat", "latitude"), "latitude")
    lon_name = _coordinate_name(list(field.dims), ("lon", "longitude"), "longitude")
    if any(field[name].ndim != 1 for name in (time_dim, lat_name, lon_name)):
        raise ValueError("Time, latitude, and longitude coordinates must be one-dimensional")
    field = field.transpose(time_dim, lat_name, lon_name)
    return field.astype(np.float32), time_dim, lat_name, lon_name


def validate_year_ranges(args: argparse.Namespace) -> Dict[str, range]:
    if not (
        args.train_start_year < args.val_start_year
        < args.test_start_year
        < args.end_year
    ):
        raise ValueError(
            "Year ranges must satisfy train-start < val-start < test-start < exclusive end-year"
        )
    if args.num_shards <= 0 or HOURS_PER_YEAR % args.num_shards:
        raise ValueError("--num-shards must be a positive divisor of {}".format(HOURS_PER_YEAR))
    return {
        "train": range(args.train_start_year, args.val_start_year),
        "val": range(args.val_start_year, args.test_start_year),
        "test": range(args.test_start_year, args.end_year),
    }


def year_array(
    field: xr.DataArray, time_dim: str, year: int, lat_name: str, lon_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    year_field = field.where(field[time_dim].dt.year == year, drop=True)
    if year_field.sizes.get(time_dim, 0) < HOURS_PER_YEAR:
        raise ValueError(
            "Expected at least {} hourly values for {}, found {}".format(
                HOURS_PER_YEAR, year, year_field.sizes.get(time_dim, 0)
            )
        )
    # Match the pinned ClimaX converter's 365-day year convention.
    values = np.asarray(year_field.isel({time_dim: slice(0, HOURS_PER_YEAR)}).values)
    if values.shape[1:] != GRID_SHAPE:
        raise ValueError(
            "Expected the 5.625-degree grid {}; found {}".format(GRID_SHAPE, values.shape[1:])
        )
    if not np.isfinite(values).all():
        raise ValueError("Geopotential values for {} contain NaN or infinity".format(year))
    lat = np.asarray(year_field[lat_name].values, dtype=np.float32)
    lon = np.asarray(year_field[lon_name].values, dtype=np.float32)
    return values[:, None], lat, lon


def write_split(
    field: xr.DataArray,
    time_dim: str,
    lat_name: str,
    lon_name: str,
    years: range,
    split: str,
    output_dir: Path,
    num_shards: int,
) -> Tuple[float, float, int]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_size = HOURS_PER_YEAR // num_shards
    climate_sum = np.zeros(GRID_SHAPE, dtype=np.float64)
    total_hours = 0
    total_sum = 0.0
    total_square_sum = 0.0
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None

    for year in years:
        values, year_lat, year_lon = year_array(field, time_dim, year, lat_name, lon_name)
        if lat is not None and (not np.array_equal(lat, year_lat) or not np.array_equal(lon, year_lon)):
            raise ValueError("Latitude/longitude coordinates changed in {}".format(year))
        lat, lon = year_lat, year_lon
        flat = values.astype(np.float64, copy=False)
        total_sum += float(flat.sum())
        total_square_sum += float(np.square(flat).sum())
        climate_sum += flat.sum(axis=(0, 1))
        total_hours += values.shape[0]

        for shard_id in range(num_shards):
            start = shard_id * shard_size
            end = start + shard_size
            np.savez_compressed(
                split_dir / "{}_{}.npz".format(year, shard_id),
                **{VARIABLE_NAME: values[start:end]},
            )

    if total_hours == 0:
        raise ValueError("No years selected for split {!r}".format(split))
    climatology = climate_sum / total_hours
    np.savez_compressed(
        split_dir / "climatology.npz",
        **{VARIABLE_NAME: climatology[None].astype(np.float32)},
    )
    mean = total_sum / (total_hours * GRID_SHAPE[0] * GRID_SHAPE[1])
    variance = total_square_sum / (total_hours * GRID_SHAPE[0] * GRID_SHAPE[1]) - mean**2
    return mean, float(np.sqrt(max(variance, 0.0))), total_hours


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    splits = validate_year_ranges(args)
    files = find_netcdf_files(args.input_dir.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty: {}. Choose a new directory.".format(output_dir)
        )

    if len(files) == 1:
        opened = xr.open_dataset(files[0])
    else:
        opened = xr.open_mfdataset(files, combine="by_coords", parallel=False)
    try:
        field, time_dim, lat_name, lon_name = select_geopotential_500(opened)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_mean, train_std, train_hours = write_split(
            field, time_dim, lat_name, lon_name, splits["train"], "train", output_dir, args.num_shards
        )
        if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std <= 0:
            raise ValueError("Training normalization statistics are invalid")
        for split in ("val", "test"):
            write_split(
                field, time_dim, lat_name, lon_name, splits[split], split, output_dir, args.num_shards
            )
        np.savez_compressed(
            output_dir / "normalize_mean.npz",
            **{VARIABLE_NAME: np.asarray([train_mean], dtype=np.float32)},
        )
        np.savez_compressed(
            output_dir / "normalize_std.npz",
            **{VARIABLE_NAME: np.asarray([train_std], dtype=np.float32)},
        )
        latitude = np.asarray(opened[lat_name].values, dtype=np.float32)
        longitude = np.asarray(opened[lon_name].values, dtype=np.float32)
        if latitude.shape != (GRID_SHAPE[0],) or longitude.shape != (GRID_SHAPE[1],):
            raise ValueError("Latitude/longitude coordinates do not match {}".format(GRID_SHAPE))
        np.save(output_dir / "lat.npy", latitude)
        np.save(output_dir / "lon.npy", longitude)
    finally:
        opened.close()

    return {
        "input_dir": str(args.input_dir.resolve()),
        "source_files": [str(path) for path in files],
        "output_dir": str(output_dir),
        "variable": VARIABLE_NAME,
        "grid_shape": list(GRID_SHAPE),
        "hours_per_year": HOURS_PER_YEAR,
        "shards_per_year": args.num_shards,
        "train_hours": train_hours,
        "normalization_mean": train_mean,
        "normalization_std": train_std,
        "split_years": {name: list(years) for name, years in splits.items()},
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
