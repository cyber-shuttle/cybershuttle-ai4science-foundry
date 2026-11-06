#!/usr/bin/env python
"""Generate simple 72-hour geopotential baselines and a forecast comparison plot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from forecast_visualization import write_forecast_svg


VARIABLE_NAME = "geopotential_500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shard", type=Path, help="Shard filename or path; defaults to first sorted shard")
    parser.add_argument("--variable", default=VARIABLE_NAME)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--lead-hours", type=int, default=72)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-stride", type=int, default=24)
    return parser.parse_args()


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(latitude)).astype(np.float64)
    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or weights.mean() <= 0:
        raise ValueError("Latitude coordinates do not produce valid cosine weights")
    weights /= weights.mean()
    errors = np.square(prediction.astype(np.float64) - target.astype(np.float64))
    sample_mse = np.mean(errors * weights[None, :, None], axis=(-2, -1))
    return float(np.sqrt(sample_mse).mean())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_shard(data_dir: Path, split: str, shard_arg: Optional[Path]) -> Path:
    split_dir = data_dir / split
    if shard_arg is None:
        candidates = sorted(path for path in split_dir.glob("*.npz") if path.name != "climatology.npz")
        if not candidates:
            raise FileNotFoundError("No data shards found under {}".format(split_dir))
        return candidates[0]
    if shard_arg.is_absolute():
        return shard_arg
    if shard_arg.is_file():
        return shard_arg
    split_candidate = split_dir / shard_arg
    if split_candidate.is_file() or shard_arg.parent == Path("."):
        return split_candidate
    data_candidate = data_dir / shard_arg
    return data_candidate


def load_field(shard: Path, variable: str) -> np.ndarray:
    with np.load(shard) as archive:
        if variable not in archive.files:
            raise KeyError("Variable {!r} is missing from {}".format(variable, shard))
        values = np.asarray(archive[variable], dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(
            "Expected {!r} with shape [time, 1, lat, lon]; found {}".format(
                variable, values.shape
            )
        )
    if not np.isfinite(values).all():
        raise ValueError("Input shard contains non-finite values: {}".format(shard))
    return values[:, 0]


def load_climatology(data_dir: Path, variable: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    path = data_dir / "train" / "climatology.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "Training climatology is missing: {}. Re-run forecast preprocessing.".format(path)
        )
    with np.load(path) as archive:
        if variable not in archive.files:
            raise KeyError("Training climatology is missing variable {!r}".format(variable))
        values = np.asarray(archive[variable], dtype=np.float32)
    if values.shape == (1,) + expected_shape:
        values = values[0]
    if values.shape != expected_shape:
        raise ValueError(
            "Training climatology has shape {}; expected {}".format(values.shape, expected_shape)
        )
    if not np.isfinite(values).all():
        raise ValueError("Training climatology contains non-finite values")
    return values


def run_baseline(
    data_dir: Path,
    output_dir: Path,
    split: str = "test",
    shard_arg: Optional[Path] = None,
    variable: str = VARIABLE_NAME,
    time_index: int = 0,
    lead_hours: int = 72,
    num_samples: int = 32,
    sample_stride: int = 24,
) -> Dict[str, Any]:
    if lead_hours <= 0 or num_samples <= 0 or sample_stride <= 0 or time_index < 0:
        raise ValueError("Lead, sample count, and stride must be positive; time index must be non-negative")
    if variable != VARIABLE_NAME:
        raise ValueError("This reduced-data workflow supports only {!r}".format(VARIABLE_NAME))

    data_dir = data_dir.expanduser().resolve()
    shard = resolve_shard(data_dir, split, shard_arg).expanduser().resolve()
    if not shard.is_file():
        raise FileNotFoundError("Input shard not found: {}".format(shard))
    values = load_field(shard, variable)
    sample_indices = time_index + np.arange(num_samples, dtype=np.int64) * sample_stride
    target_indices = sample_indices + lead_hours
    if target_indices[-1] >= values.shape[0]:
        raise ValueError(
            "Selected input/target indices end at {}; shard has {} time steps".format(
                target_indices[-1], values.shape[0]
            )
        )

    inputs = values[sample_indices]
    target = values[target_indices]
    persistence = inputs.copy()
    climatology = load_climatology(data_dir, variable, values.shape[1:])
    climatology_samples = np.broadcast_to(climatology, target.shape).copy()

    latitude = np.asarray(np.load(data_dir / "lat.npy"), dtype=np.float32)
    longitude = np.asarray(np.load(data_dir / "lon.npy"), dtype=np.float32)
    if latitude.shape != (values.shape[1],) or longitude.shape != (values.shape[2],):
        raise ValueError("Latitude/longitude coordinates do not match the data grid")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Latitude/longitude coordinates contain non-finite values")

    persistence_rmse = weighted_rmse(persistence, target, latitude)
    climatology_rmse = weighted_rmse(climatology_samples, target, latitude)
    improvement = (
        float(1.0 - persistence_rmse / climatology_rmse)
        if climatology_rmse > 0 else None
    )
    prediction_shape = (num_samples, 1) + values.shape[1:]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "baseline-forecast.npz"
    np.savez_compressed(
        archive_path,
        target=target[:, None],
        input=inputs[:, None],
        persistence=persistence[:, None],
        climatology=climatology_samples[:, None],
        variable=np.asarray([variable]),
        lead_hours=np.asarray(lead_hours, dtype=np.int32),
        input_indices=sample_indices,
        target_indices=target_indices,
        lat=latitude,
        lon=longitude,
    )

    figure_path = output_dir / "baseline-forecast.svg"
    write_forecast_svg(
        figure_path,
        [
            ("Input at initialization", inputs[0]),
            ("Observed target at +{} h".format(lead_hours), target[0]),
            ("Persistence baseline", persistence[0]),
            ("Training climatology baseline", climatology),
        ],
    )
    metrics: Dict[str, Any] = {
        "kind": "baseline_only_not_a_ClimaX_forecast",
        "smoke_test": {
            "prediction_shape": list(prediction_shape),
            "all_values_finite": True,
            "status": "passed",
        },
        "evaluation": {
            "variable": variable,
            "split": split,
            "shard": str(shard),
            "lead_hours": lead_hours,
            "sample_count": num_samples,
            "sample_stride_hours": sample_stride,
            "input_indices": sample_indices.tolist(),
            "target_indices": target_indices.tolist(),
            "prediction_shape": list(prediction_shape),
            "climatology_source": "training split",
            "latitude_weighted_rmse": {
                "persistence": persistence_rmse,
                "training_climatology": climatology_rmse,
                "persistence_relative_improvement_over_climatology": improvement,
            },
        },
        "outputs": {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "visualization": str(figure_path),
        },
        "input_sha256": sha256_file(shard),
        "python_version": sys.version.split()[0],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    args = parse_args()
    run_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        shard_arg=args.shard,
        variable=args.variable,
        time_index=args.time_index,
        lead_hours=args.lead_hours,
        num_samples=args.num_samples,
        sample_stride=args.sample_stride,
    )


if __name__ == "__main__":
    main()
