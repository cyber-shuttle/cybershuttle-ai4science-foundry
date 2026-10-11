#!/usr/bin/env python
"""Run and evaluate one ClimaX global forecast from a prepared ERA5 shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml


EXPECTED_CHECKPOINT_SHA256 = "6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(npz: Any, name: str, expected_shape: Tuple[int, int, int, int]) -> np.ndarray:
    if name not in npz.files:
        raise KeyError("Required variable {!r} is absent from the shard".format(name))
    array = np.asarray(npz[name], dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(
            "Variable {!r} must have shape [time, 1, lat, lon]; got {}".format(name, array.shape)
        )
    if array.shape != expected_shape:
        raise ValueError(
            "Variable {!r} has shape {}; expected {}".format(name, array.shape, expected_shape)
        )
    return array


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or weights.mean() <= 0:
        raise ValueError("Latitude coordinates do not produce valid cosine weights")
    weights = weights / weights.mean()
    error_squared = np.square(prediction.astype(np.float64) - target.astype(np.float64))
    return float(np.sqrt(np.mean(error_squared * weights[None, :, None])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climax-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--lead-hours", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.climax_source.resolve()
    source_src = source / "src"
    if not source_src.is_dir():
        raise FileNotFoundError("ClimaX source must contain src/: {}".format(source))
    sys.path.insert(0, str(source_src))

    from climax.arch import ClimaX
    from climax.global_forecast.module import GlobalForecastModule

    if not args.config.is_file():
        raise FileNotFoundError("Config not found: {}".format(args.config))
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: got {}, expected {}".format(
                checkpoint_hash, EXPECTED_CHECKPOINT_SHA256
            )
        )

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_config = config["model"]["net"]["init_args"]
    data_config = config["data"]
    variables = list(data_config["variables"])
    out_variables = list(data_config["out_variables"])
    if not variables or not out_variables:
        raise ValueError("The forecast config must define input and output variables")
    missing_outputs = sorted(set(out_variables) - set(variables))
    if missing_outputs:
        raise ValueError("Output variables must also be model inputs: {}".format(missing_outputs))
    if args.lead_hours <= 0:
        raise ValueError("--lead-hours must be a positive integer")
    hours_each_step = int(data_config.get("hrs_each_step", 1))
    if args.lead_hours % hours_each_step:
        raise ValueError(
            "--lead-hours must be divisible by data.hrs_each_step ({})".format(hours_each_step)
        )
    lead_steps = args.lead_hours // hours_each_step

    shard = args.shard
    if shard is None:
        candidates = sorted((args.data_dir / args.split).glob("*.npz"))
        candidates = [path for path in candidates if path.name != "climatology.npz"]
        if not candidates:
            raise FileNotFoundError("No .npz shards found under {}".format(args.data_dir / args.split))
        shard = candidates[0]
    elif not shard.is_absolute():
        shard = args.data_dir / args.split / shard
    if not shard.is_file():
        raise FileNotFoundError("Selected shard not found: {}".format(shard))

    if args.time_index < 0:
        raise ValueError("--time-index must be non-negative")
    with np.load(shard) as archive:
        if not archive.files:
            raise ValueError("Input shard is empty: {}".format(shard))
        first_name = variables[0]
        if first_name not in archive.files:
            raise KeyError("Required variable {!r} is absent from {}".format(first_name, shard))
        first = np.asarray(archive[first_name], dtype=np.float32)
        if first.ndim != 4 or first.shape[1] != 1:
            raise ValueError(
                "Variable {!r} must have shape [time, 1, lat, lon]; got {}".format(first_name, first.shape)
            )
        time_count, _, height, width = first.shape
        expected_shape = (time_count, 1, height, width)
        if args.time_index + lead_steps >= time_count:
            raise ValueError(
                "Input index {} plus lead {} steps exceeds {} timesteps in {}".format(
                    args.time_index, lead_steps, time_count, shard
                )
            )

        raw_by_variable: Dict[str, np.ndarray] = {}
        for variable in variables:
            raw_by_variable[variable] = load_array(archive, variable, expected_shape)

    with np.load(args.data_dir / "normalize_mean.npz") as archive:
        missing = [name for name in variables if name not in archive.files]
        if missing:
            raise KeyError("Normalization means are missing variables: {}".format(missing))
        means = [np.asarray(archive[name], dtype=np.float32).reshape(-1) for name in variables]
    with np.load(args.data_dir / "normalize_std.npz") as archive:
        missing = [name for name in variables if name not in archive.files]
        if missing:
            raise KeyError("Normalization standard deviations are missing variables: {}".format(missing))
        standard_deviations = [
            np.asarray(archive[name], dtype=np.float32).reshape(-1) for name in variables
        ]
    mean = np.concatenate(means)
    std = np.concatenate(standard_deviations)
    if mean.size != len(variables) or std.size != len(variables):
        raise ValueError("Expected one normalization value per input variable")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Input normalization statistics must be finite with positive standard deviations")

    out_means: List[float] = []
    out_stds: List[float] = []
    with np.load(args.data_dir / "normalize_mean.npz") as archive:
        for variable in out_variables:
            if variable not in archive.files:
                raise KeyError("Output normalization mean missing for {!r}".format(variable))
            out_means.append(float(np.asarray(archive[variable]).reshape(-1)[0]))
    with np.load(args.data_dir / "normalize_std.npz") as archive:
        for variable in out_variables:
            if variable not in archive.files:
                raise KeyError("Output normalization standard deviation missing for {!r}".format(variable))
            out_stds.append(float(np.asarray(archive[variable]).reshape(-1)[0]))
    if not np.isfinite(out_means).all() or not np.isfinite(out_stds).all() or np.any(np.array(out_stds) <= 0):
        raise ValueError("Output normalization statistics must be finite with positive standard deviations")

    lat = np.asarray(np.load(args.data_dir / "lat.npy"), dtype=np.float32)
    lon = np.asarray(np.load(args.data_dir / "lon.npy"), dtype=np.float32)
    if lat.shape != (height,) or lon.shape != (width,):
        raise ValueError(
            "lat/lon dimensions {} and {} do not match input grid {} x {}".format(
                lat.shape, lon.shape, height, width
            )
        )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Latitude and longitude coordinates contain non-finite values")

    target_index = args.time_index + lead_steps
    x_raw = np.concatenate(
        [raw_by_variable[name][args.time_index] for name in variables], axis=0
    )
    x_normalized = (x_raw - mean[:, None, None]) / std[:, None, None]
    target = np.concatenate(
        [raw_by_variable[name][target_index] for name in out_variables], axis=0
    )
    persistence = np.concatenate(
        [raw_by_variable[name][args.time_index] for name in out_variables], axis=0
    )

    climatology_path = args.data_dir / args.split / "climatology.npz"
    with np.load(climatology_path) as archive:
        missing = [name for name in out_variables if name not in archive.files]
        if missing:
            raise KeyError("Climatology is missing output variables: {}".format(missing))
        climatology = np.concatenate(
            [np.asarray(archive[name], dtype=np.float32).reshape(1, height, width) for name in out_variables],
            axis=0,
        )
    if not all(
        np.isfinite(array).all()
        for array in (x_normalized, target, persistence, climatology)
    ):
        raise ValueError("Input, target, or baseline data contains non-finite values")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(42)
    net = ClimaX(**model_config)
    model = GlobalForecastModule(net=net, pretrained_path=str(args.checkpoint))
    model.to(device)
    model.eval()
    x_tensor = torch.from_numpy(x_normalized[None]).to(device=device, dtype=torch.float32)
    lead_times = torch.tensor([args.lead_hours / 100.0], device=device, dtype=torch.float32)

    start_time = time.monotonic()
    with torch.no_grad():
        _, prediction_normalized = model.net(
            x_tensor,
            None,
            lead_times,
            variables,
            out_variables,
            metric=None,
            lat=None,
        )
    runtime_seconds = time.monotonic() - start_time
    prediction = (
        prediction_normalized[0].detach().cpu().numpy()
        * np.asarray(out_stds, dtype=np.float32)[:, None, None]
        + np.asarray(out_means, dtype=np.float32)[:, None, None]
    )
    expected_output_shape = (len(out_variables), height, width)
    if prediction.shape != expected_output_shape or target.shape != expected_output_shape:
        raise ValueError(
            "Prediction/target shapes {} / {} do not match expected {}".format(
                prediction.shape, target.shape, expected_output_shape
            )
        )
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction or target contains non-finite values")

    scores: Dict[str, Dict[str, float]] = {}
    for index, variable in enumerate(out_variables):
        scores[variable] = {
            "climax_weighted_rmse": weighted_rmse(prediction[index], target[index], lat),
            "persistence_weighted_rmse": weighted_rmse(persistence[index], target[index], lat),
            "climatology_weighted_rmse": weighted_rmse(climatology[index], target[index], lat),
            "prediction_min": float(prediction[index].min()),
            "prediction_max": float(prediction[index].max()),
            "target_min": float(target[index].min()),
            "target_max": float(target[index].max()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "climax-forecast.npz"
    np.savez_compressed(
        result_path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        persistence=persistence.astype(np.float32),
        climatology=climatology.astype(np.float32),
        variables=np.asarray(out_variables),
        lat=lat,
        lon=lon,
        lead_hours=np.asarray(args.lead_hours, dtype=np.int32),
        input_index=np.asarray(args.time_index, dtype=np.int64),
        target_index=np.asarray(target_index, dtype=np.int64),
    )

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    metrics: Dict[str, Any] = {
        "smoke_test": {
            "prediction_shape": list(prediction.shape),
            "expected_shape": list(expected_output_shape),
            "all_prediction_values_finite": bool(np.isfinite(prediction).all()),
            "all_target_values_finite": bool(np.isfinite(target).all()),
            "status": "passed",
        },
        "evaluation": {
            "lead_hours": args.lead_hours,
            "split": args.split,
            "shard": str(shard.resolve()),
            "input_index": args.time_index,
            "target_index": target_index,
            "variables": out_variables,
            "latitude_weighted_rmse": scores,
        },
        "run": {
            "runtime_seconds_forward_pass": runtime_seconds,
            "device": str(device),
            "gpu_name": gpu_name,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "climax_source": str(source),
            "climax_source_commit": os.environ.get("CLIMAX_COMMIT", "not-recorded"),
            "config": str(args.config.resolve()),
            "output_file": str(result_path.resolve()),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
