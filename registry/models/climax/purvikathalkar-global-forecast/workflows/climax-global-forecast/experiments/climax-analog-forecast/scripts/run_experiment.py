#!/usr/bin/env python
"""Run raw-field and pooled-ClimaX nearest-analog 72-hour forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


VARIABLE_NAME = "geopotential_500"


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, latitude: np.ndarray) -> float:
    weights = np.clip(np.cos(np.deg2rad(latitude)), 0.0, None).astype(np.float64)
    weights /= weights.mean()
    errors = np.square(prediction.astype(np.float64) - target.astype(np.float64))
    sample_mse = np.mean(errors * weights[None, :, None], axis=(-2, -1))
    return float(np.sqrt(sample_mse).mean())


def load_split(data_dir: Path, split: str, variable: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = data_dir / split
    shards = sorted(path for path in split_dir.glob("*.npz") if path.name != "climatology.npz")
    if not shards:
        raise FileNotFoundError("No data shards found under {}".format(split_dir))
    arrays = []
    for shard in shards:
        with np.load(shard) as archive:
            if variable not in archive.files:
                raise KeyError("{} is missing {!r}".format(shard, variable))
            values = np.asarray(archive[variable], dtype=np.float32)
        if values.ndim != 4 or values.shape[1] != 1:
            raise ValueError("{} has unexpected shape {}".format(shard, values.shape))
        arrays.append(values[:, 0])
    values = np.concatenate(arrays, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("{} values contain non-finite values".format(split))
    latitude = np.asarray(np.load(data_dir / "lat.npy"), dtype=np.float32)
    longitude = np.asarray(np.load(data_dir / "lon.npy"), dtype=np.float32)
    if latitude.shape != (values.shape[1],) or longitude.shape != (values.shape[2],):
        raise ValueError("Coordinates do not match {} grid".format(split))
    return values, latitude, longitude


def cosine_retrieve(query: np.ndarray, database: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0 or k > database.shape[0]:
        raise ValueError("k={} must be between 1 and {}".format(k, database.shape[0]))
    query = np.asarray(query, dtype=np.float64)
    database = np.asarray(database, dtype=np.float64)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    database_norm = np.linalg.norm(database, axis=1, keepdims=True)
    if np.any(query_norm == 0) or np.any(database_norm == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors")
    similarity = (query / query_norm) @ (database / database_norm).T
    indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    scores = np.take_along_axis(similarity, indices, axis=1)
    return indices, scores


def load_embedding_archive(path: Path, variable: str) -> Tuple[np.ndarray, np.ndarray]:
    if path is None:
        raise FileNotFoundError("ClimaX embedding archive was not provided")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("ClimaX embedding archive not found: {}".format(path))
    with np.load(path) as archive:
        if "pooled_embeddings" not in archive.files or "sample_indices" not in archive.files:
            raise KeyError("{} lacks pooled_embeddings/sample_indices".format(path))
        embeddings = np.asarray(archive["pooled_embeddings"], dtype=np.float32)
        indices = np.asarray(archive["sample_indices"], dtype=np.int64)
        if "variable" in archive.files:
            saved_variable = str(np.asarray(archive["variable"]).reshape(-1)[0])
            if saved_variable != variable:
                raise ValueError("Embedding variable {} != {}".format(saved_variable, variable))
    if embeddings.ndim != 2 or embeddings.shape[0] != indices.shape[0]:
        raise ValueError("Invalid pooled embedding shapes in {}".format(path))
    return embeddings, indices


def write_svg(output_path: Path, panels: Sequence[Tuple[str, np.ndarray]]) -> None:
    workflow_scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(workflow_scripts) not in sys.path:
        sys.path.insert(0, str(workflow_scripts))
    from forecast_visualization import write_forecast_svg

    write_forecast_svg(output_path, panels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable", default=VARIABLE_NAME)
    parser.add_argument("--lead-hours", type=int, default=72)
    parser.add_argument("--time-step-hours", type=int, default=6)
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-stride-hours", type=int, default=24)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--reference-embeddings", type=Path)
    parser.add_argument("--evaluation-embeddings", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    if args.lead_hours <= 0 or args.time_step_hours <= 0:
        raise ValueError("lead-hours and time-step-hours must be positive")
    if args.lead_hours % args.time_step_hours != 0:
        raise ValueError("lead-hours must be divisible by time-step-hours")
    lead_steps = args.lead_hours // args.time_step_hours
    stride_steps = max(1, args.sample_stride_hours // args.time_step_hours)
    if args.sample_stride_hours % args.time_step_hours != 0:
        raise ValueError("sample-stride-hours must be divisible by time-step-hours")

    reference, latitude, longitude = load_split(
        args.data_dir, args.reference_split, args.variable
    )
    evaluation, eval_latitude, eval_longitude = load_split(
        args.data_dir, args.evaluation_split, args.variable
    )
    if not np.array_equal(latitude, eval_latitude) or not np.array_equal(longitude, eval_longitude):
        raise ValueError("Reference and evaluation coordinates differ")
    if args.time_index < 0 or args.num_samples <= 0:
        raise ValueError("time-index must be non-negative and num-samples positive")

    evaluation_indices = args.time_index + np.arange(args.num_samples) * stride_steps
    evaluation_targets = evaluation_indices + lead_steps
    if evaluation_targets[-1] >= evaluation.shape[0]:
        raise ValueError(
            "Evaluation target index {} exceeds {} time steps".format(
                int(evaluation_targets[-1]), evaluation.shape[0]
            )
        )
    eval_inputs = evaluation[evaluation_indices]
    eval_targets = evaluation[evaluation_targets]

    reference_initial = np.arange(0, reference.shape[0] - lead_steps)
    reference_targets = reference_initial + lead_steps
    raw_database = reference[reference_initial]
    raw_future = reference[reference_targets]

    flat_database = raw_database.reshape(raw_database.shape[0], -1)
    flat_queries = eval_inputs.reshape(eval_inputs.shape[0], -1)
    raw_mean = flat_database.mean(axis=0)
    raw_std = flat_database.std(axis=0)
    raw_std = np.where(raw_std > 1e-12, raw_std, 1.0)
    raw_database_std = (flat_database - raw_mean) / raw_std
    raw_queries_std = (flat_queries - raw_mean) / raw_std

    persistence = eval_inputs.copy()
    climatology_path = args.data_dir / args.reference_split / "climatology.npz"
    with np.load(climatology_path) as archive:
        if args.variable not in archive.files:
            raise KeyError("{} lacks {!r}".format(climatology_path, args.variable))
        climatology = np.asarray(archive[args.variable], dtype=np.float32)
    climatology = climatology[0] if climatology.shape[0] == 1 else climatology
    if climatology.shape != eval_inputs.shape[1:] or not np.isfinite(climatology).all():
        raise ValueError("Training climatology shape or values are invalid")
    climatology_forecasts = np.broadcast_to(climatology, eval_targets.shape).copy()

    metrics: Dict[str, object] = {
        "status": "completed",
        "python_version": sys.version.split()[0],
        "seed": args.seed,
        "configuration": {
            "variable": args.variable,
            "lead_hours": args.lead_hours,
            "time_step_hours": args.time_step_hours,
            "reference_split": args.reference_split,
            "evaluation_split": args.evaluation_split,
            "evaluation_initial_indices": evaluation_indices.tolist(),
            "evaluation_target_indices": evaluation_targets.tolist(),
            "sample_count": args.num_samples,
            "sample_stride_hours": args.sample_stride_hours,
            "k_values": args.k_values,
            "leakage_prevention": "chronological train/test split; reference never uses evaluation shard",
        },
    }
    table: List[Dict[str, object]] = []
    table.append({"method": "Persistence", "k": None,
                  "weighted_rmse": weighted_rmse(persistence, eval_targets, latitude)})
    table.append({"method": "Climatology", "k": None,
                  "weighted_rmse": weighted_rmse(climatology_forecasts, eval_targets, latitude)})

    retrieval: Dict[str, object] = {"raw": {}, "climax": {}}
    forecasts: Dict[str, np.ndarray] = {
        "persistence": persistence,
        "climatology": climatology_forecasts,
    }

    for k in args.k_values:
        indices, scores = cosine_retrieve(raw_queries_std, raw_database_std, k)
        forecast = np.mean(raw_future[indices], axis=1)
        forecasts["raw_k{}".format(k)] = forecast
        retrieval["raw"][str(k)] = {
            "indices": indices.tolist(),
            "scores": scores.tolist(),
        }
        table.append({"method": "Raw ERA5 analog", "k": k,
                      "weighted_rmse": weighted_rmse(forecast, eval_targets, latitude)})

    pooled: Optional[np.ndarray] = None
    if args.reference_embeddings is not None or args.evaluation_embeddings is not None:
        if args.reference_embeddings is None or args.evaluation_embeddings is None:
            raise ValueError("Provide both --reference-embeddings and --evaluation-embeddings")
        ref_embeddings, ref_embedding_indices = load_embedding_archive(
            args.reference_embeddings, args.variable
        )
        eval_embeddings, eval_embedding_indices = load_embedding_archive(
            args.evaluation_embeddings, args.variable
        )
        if not np.array_equal(eval_embedding_indices, evaluation_indices):
            raise ValueError(
                "Evaluation embedding indices {} do not match requested indices {}. "
                "For --time-step-hours 6, --sample-stride-hours 24 means 4 array steps; "
                "set EMBED_SAMPLE_STRIDE=4 in the embedding SLURM job.".format(
                    eval_embedding_indices.tolist(), evaluation_indices.tolist()
                )
            )
        # Reference archives may intentionally contain only a subset of states.
        ref_map = {int(index): position for position, index in enumerate(ref_embedding_indices)}
        if len(ref_map) != len(ref_embedding_indices):
            raise ValueError("Reference embedding indices contain duplicates")
        available_initial_indices = np.asarray(
            [int(index) for index in reference_initial if int(index) in ref_map],
            dtype=np.int64,
        )
        if available_initial_indices.size == 0:
            raise ValueError("Reference embedding archive has no usable initial-state indices")
        ref_embedding_vectors = ref_embeddings[
            [ref_map[int(index)] for index in available_initial_indices]
        ]
        for k in args.k_values:
            indices, scores = cosine_retrieve(eval_embeddings, ref_embedding_vectors, k)
            retrieved_initial_indices = available_initial_indices[indices]
            forecast = np.mean(reference[retrieved_initial_indices + lead_steps], axis=1)
            forecasts["climax_k{}".format(k)] = forecast
            retrieval["climax"][str(k)] = {
                "indices": indices.tolist(),
                "reference_initial_indices": retrieved_initial_indices.tolist(),
                "scores": scores.tolist(),
            }
            table.append({"method": "ClimaX embedding analog", "k": k,
                          "weighted_rmse": weighted_rmse(forecast, eval_targets, latitude)})

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "analog-forecast.npz",
        target=eval_targets,
        input=eval_inputs,
        persistence=persistence,
        climatology=climatology_forecasts,
        raw_forecasts=np.stack([forecasts["raw_k{}".format(k)] for k in args.k_values]),
        k_values=np.asarray(args.k_values, dtype=np.int32),
        input_indices=evaluation_indices,
        target_indices=evaluation_targets,
    )
    if pooled is not None:
        pass
    if "climax_k{}".format(args.k_values[0]) in forecasts:
        np.savez_compressed(
            output_dir / "climax-analog-forecast.npz",
            **{"forecast_k{}".format(k): forecasts["climax_k{}".format(k)] for k in args.k_values},
        )

    metrics["results"] = table
    metrics["retrieval"] = retrieval
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if eval_inputs.shape[0] > 0:
        first_panels = [
            ("Input at initialization", eval_inputs[0]),
            ("Observed target at +{} h".format(args.lead_hours), eval_targets[0]),
            ("Persistence", persistence[0]),
            ("Training climatology", climatology),
        ]
        for k in args.k_values:
            first_panels.append(("Raw analog k={}".format(k), forecasts["raw_k{}".format(k)][0]))
            if "climax_k{}".format(k) in forecasts:
                first_panels.append(("ClimaX analog k={}".format(k), forecasts["climax_k{}".format(k)][0]))
        write_svg(output_dir / "forecast-comparison.svg", first_panels)

        analog_panels = [("Current input", eval_inputs[0])]
        raw_positions = retrieval["raw"][str(args.k_values[0])]["indices"][0]
        for position in raw_positions:
            analog_panels.append(
                ("Raw analog initial", raw_database[int(position)])
            )
            analog_panels.append(
                ("Raw analog future", raw_future[int(position)])
            )
        analog_panels.append(("Raw analog average", forecasts["raw_k{}".format(args.k_values[0])][0]))
        write_svg(output_dir / "analog-retrieval.svg", analog_panels)

    return metrics


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    print(json.dumps(run_experiment(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
