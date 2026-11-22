"""Inspect an existing CEBRA cache; never download data."""

import argparse
import json
from pathlib import Path

import numpy as np

DATASET_ID = "rat-hippocampus-single-achilles"
SOURCE = "https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/datasets/hippocampus.py"


def load_achilles(root):
    import cebra.datasets
    import cebra.datasets.hippocampus  # registers the named dataset

    dataset = cebra.datasets.init(DATASET_ID, root=str(root), download=False)
    neural = dataset.neural.detach().cpu().numpy()
    behavior = dataset.continuous_index.detach().cpu().numpy()
    if neural.shape != (10178, 120) or behavior.shape != (10178, 3):
        raise ValueError("Cache does not match the specified Achilles baseline shapes.")
    if not np.isfinite(neural).all() or not np.isfinite(behavior).all():
        raise ValueError("Nonfinite neural or behavioral values.")
    return dataset, neural, behavior


def summarize(values):
    return {
        "shape": list(values.shape), "dtype": str(values.dtype),
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "std": float(values.std()),
        "zero_proportion": float(np.mean(values == 0)),
        "integer_like": bool(np.allclose(values, np.rint(values), rtol=0, atol=1e-6)),
        "example_rows": values[:3].tolist(),
    }


def main():
    import inspect
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    dataset, neural, behavior = load_achilles(args.data_root)
    print(json.dumps({"neural": summarize(neural), "behavior": summarize(behavior)}, indent=2))
    print("Installed loader documentation:", inspect.getdoc(type(dataset)))
    print("Public source:", SOURCE)
    # Print numeric metadata only; never dump paths or arbitrary object reprs.
    tokens = ("time", "stamp", "sampling", "bin", "duration", "unit", "rate", "segment")
    sources = [vars(dataset)]
    cached = joblib.load(args.data_root / "rat_hippocampus" / "achilles.jl")
    if isinstance(cached, dict):
        print("Cache field names:", sorted(cached))
        sources.append(cached)
    for metadata in sources:
        for key, value in metadata.items():
            if any(token in key.lower() for token in tokens):
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                if isinstance(value, (np.ndarray, int, float, list, tuple)):
                    array = np.asarray(value)
                    if array.dtype.kind in "biuf" and array.size:
                        print(key, json.dumps({"shape": list(array.shape),
                                              "first_values": array.ravel()[:8].tolist()}))
    print("Verify bin coordinates, gaps, target timing, and column/unit mapping before conversion.")


if __name__ == "__main__":
    main()
