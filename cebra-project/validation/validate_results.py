"""Check saved hippocampus embeddings and plot them by animal position."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/hippocampus"))
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Run the workflow first; {results_dir} does not exist.")
    summary_path = results_dir / "validation_summary.json"
    summary = {
        "status": "running",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "expected": "Finite embeddings of shape (N, 3) aligned with N behavioral rows.",
        "visual_review": "PENDING: inspect the position-colored plot and record observations.",
        "scope": "Descriptive validation on the training samples; no held-out decoding claim.",
    }
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metadata = json.loads((results_dir / "run_metadata.json").read_text())
        if metadata["status"] != "success":
            raise ValueError("The workflow metadata does not report a successful run.")
        with np.load(results_dir / "embeddings.npz", allow_pickle=False) as saved:
            embeddings = saved["embeddings"]
            behavior = saved["behavior"]
            sample_indices = saved["sample_indices"]

        summary["embedding_shape"] = list(embeddings.shape)
        summary["behavior_shape"] = list(behavior.shape)
        summary["embedding_finite"] = bool(np.isfinite(embeddings).all())
        summary["behavior_finite"] = bool(np.isfinite(behavior).all())
        if embeddings.ndim != 2 or embeddings.shape[1] != 3 or len(embeddings) == 0:
            raise ValueError("The embedding must be nonempty and two-dimensional with three columns.")
        if behavior.shape != (len(embeddings), 3):
            raise ValueError("Behavior must have aligned position, right and left columns.")
        if not summary["embedding_finite"] or not summary["behavior_finite"]:
            raise ValueError("The embeddings and labels must have no NaN or Inf values.")
        if list(embeddings.shape) != [metadata["neural_shape"][0], 3]:
            raise ValueError("The embedding does not match the recorded neural sample count.")
        if not np.array_equal(sample_indices, np.arange(len(embeddings))):
            raise ValueError("Sample indices must preserve the original temporal order.")

        position = behavior[:, 0]
        summary["position_range_m"] = [float(position.min()), float(position.max())]
        summary["sample_order_verified"] = True
        summary["sample_count"] = len(embeddings)
        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(111, projection="3d")
        points = axes.scatter(
            embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
            c=position, cmap="viridis", s=2, alpha=0.7, rasterized=True,
        )
        figure.colorbar(points, ax=axes, shrink=0.7, label="Animal position (m)")
        axes.set(xlabel="CEBRA 1", ylabel="CEBRA 2", zlabel="CEBRA 3")
        axes.set_title("Achilles: CEBRA embedding colored by position (training samples)")
        figure_path = results_dir / "embedding_by_position.png"
        figure.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        summary["figure"] = str(figure_path)
        summary["status"] = "numeric checks passed; visual review pending"
        print(json.dumps(summary, indent=2), flush=True)
        print("Review spatial color structure; no exact orientation or figure match is required.")
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print("Validation summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
