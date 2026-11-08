#!/usr/bin/env python
"""Validate a saved ClimaX embedding archive and optionally compare a repeat run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError("Embedding archive not found: {}".format(path))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "spatial_embeddings",
        "pooled_embeddings",
        "variable",
        "sample_indices",
        "lat",
        "lon",
        "lead_time_hours",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError("Embedding archive is missing arrays: {}".format(sorted(missing)))
    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError("Array {!r} contains NaN or infinite values".format(name))
    return arrays


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Metrics file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    smoke_test = metrics.get("smoke_test", {})
    if smoke_test.get("status") != "passed":
        raise ValueError("Recorded smoke test did not pass")
    return metrics


def compare_archives(reference: Dict[str, np.ndarray], repeat: Dict[str, np.ndarray]) -> None:
    if set(reference) != set(repeat):
        raise ValueError(
            "Archive arrays differ: reference={}, repeat={}".format(
                sorted(reference), sorted(repeat)
            )
        )
    for name, first in reference.items():
        second = repeat[name]
        if first.shape != second.shape:
            raise ValueError(
                "Array {!r} shape mismatch: {} vs {}".format(name, first.shape, second.shape)
            )
        if np.issubdtype(first.dtype, np.number) and np.issubdtype(second.dtype, np.number):
            try:
                np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
            except AssertionError as error:
                raise ValueError("Repeat comparison failed for {!r}: {}".format(name, error))
        elif not np.array_equal(first, second):
            raise ValueError("Non-numeric array {!r} differs between runs".format(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original climax-embeddings.npz")
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Metrics JSON (defaults to metrics.json next to the original archive)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Optional independently generated archive to compare with the original",
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    arrays = load_archive(archive_path)
    metrics_path = args.metrics or archive_path.parent / "metrics.json"
    metrics = load_metrics(metrics_path)
    smoke_test = metrics["smoke_test"]
    expected_shapes = {
        "spatial_embeddings": tuple(smoke_test["spatial_embedding_shape"]),
        "pooled_embeddings": tuple(smoke_test["pooled_embedding_shape"]),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                "Array {!r} has shape {}; metrics expect {}".format(
                    name, arrays[name].shape, expected_shape
                )
            )
    if not smoke_test.get("all_embedding_values_finite"):
        raise ValueError("Metrics do not record finite embedding values")
    if not smoke_test.get("nonconstant_embedding"):
        raise ValueError("Metrics do not record a nonconstant representation")
    if not smoke_test.get("repeat_inference_deterministic"):
        raise ValueError("Metrics do not record deterministic repeated inference")
    if float(arrays["spatial_embeddings"].std()) <= 0:
        raise ValueError("Spatial embeddings are constant")
    expected_indices = np.asarray(metrics["representation"]["sample_indices"])
    if not np.array_equal(arrays["sample_indices"], expected_indices):
        raise ValueError("Archive sample indices differ from metrics.json")
    expected_variable = metrics["representation"]["variable"]
    if arrays["variable"].item() != expected_variable:
        raise ValueError("Archive variable differs from metrics.json")

    actual_hash = sha256_file(archive_path)
    expected_hash = metrics.get("run", {}).get("output_sha256")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "Archive SHA-256 mismatch: got {}, metrics record {}".format(
                actual_hash, expected_hash
            )
        )

    print("Archive:", archive_path)
    print("SHA-256:", actual_hash)
    for name, array in arrays.items():
        print(
            "{}: shape={}, dtype={}, finite={}".format(
                name,
                array.shape,
                array.dtype,
                bool(np.isfinite(array).all())
                if np.issubdtype(array.dtype, np.number)
                else "not numeric",
            )
        )

    if args.compare:
        repeat_path = args.compare.resolve()
        repeat_arrays = load_archive(repeat_path)
        compare_archives(arrays, repeat_arrays)
        print("Repeat archive arrays match within rtol=1e-5, atol=1e-6.")
        print("Repeat SHA-256:", sha256_file(repeat_path))
    print("Embedding artifact validation passed.")


if __name__ == "__main__":
    main()
