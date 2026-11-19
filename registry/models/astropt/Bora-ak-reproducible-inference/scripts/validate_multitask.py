"""Probe three galaxy properties with one frozen AstroPT embedding pass."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt
from datasets import load_dataset
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms


MODEL = "Smith42/astroPT_v2.0"
CHECKPOINT = "astropt/095M"
DATASET = "Smith42/galaxies"
REVISION = "v2.0"
SPLIT = "test"
SMOOTH = "smooth-or-featured_smooth_fraction"
FEATURED = "smooth-or-featured_featured-or-disk_fraction"
TARGETS = ("redshift", "elpetro_mass_log", SMOOTH, FEATURED)
THRESHOLD = 0.7


def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def numeric(value):
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def metrics_regression(actual, predicted):
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def regression_task(name, targets, embeddings, seed, output_dir):
    # Both quantities are physical and must be positive. NaNs and infinities are excluded.
    valid = np.isfinite(targets) & (targets > 0)
    indices = np.flatnonzero(valid)
    result = {
        "target_field": name,
        "usable_examples": int(len(indices)),
        "excluded_examples": int(len(targets) - len(indices)),
        "filter": "finite and > 0",
    }
    if len(indices) < 8:
        result["skipped_reason"] = "Fewer than eight usable targets (at least two test values required for R2)"
        return result

    train, test = train_test_split(indices, test_size=0.25, random_state=seed)
    probe = make_pipeline(StandardScaler(), LinearRegression())
    probe.fit(embeddings[train], targets[train])
    predicted = probe.predict(embeddings[test])
    baseline_value = float(np.mean(targets[train]))
    baseline = np.full(len(test), baseline_value)
    result.update({
        "train_examples": int(len(train)),
        "test_examples": int(len(test)),
        "linear_probe": metrics_regression(targets[test], predicted),
        "mean_baseline": {
            "value": baseline_value,
            **metrics_regression(targets[test], baseline),
        },
    })

    plt.figure(figsize=(5, 5))
    plt.scatter(targets[test], predicted, s=9, alpha=0.5)
    low = float(min(targets[test].min(), predicted.min()))
    high = float(max(targets[test].max(), predicted.max()))
    plt.plot([low, high], [low, high], "--", color="black", linewidth=1)
    plt.xlabel(f"Actual {name}")
    plt.ylabel(f"Predicted {name}")
    plt.tight_layout()
    filename = f"multitask_{name}_predicted_vs_actual.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    result["plot"] = filename
    return result


def morphology_labels(smooth, featured, threshold=THRESHOLD):
    valid = (
        np.isfinite(smooth) & np.isfinite(featured)
        & (smooth >= 0) & (smooth <= 1)
        & (featured >= 0) & (featured <= 1)
    )
    labels = np.full(len(smooth), -1, dtype=np.int8)
    labels[valid & (smooth >= threshold) & (smooth > featured)] = 0
    labels[valid & (featured >= threshold) & (featured > smooth)] = 1
    return labels, valid


def choose_morphology_threshold(smooth, featured):
    """Inspect vote retention before choosing the predeclared confidence rule."""
    diagnostics = {}
    for threshold in (0.7, 0.6):
        labels, valid = morphology_labels(smooth, featured, threshold)
        counts = {
            "smooth": int(np.sum(labels == 0)),
            "featured_or_disk": int(np.sum(labels == 1)),
            "ambiguous": int(np.sum(valid & (labels < 0))),
            "invalid": int(np.sum(~valid)),
        }
        diagnostics[str(threshold)] = counts
        if (
            min(counts["smooth"], counts["featured_or_disk"]) >= 20
            and (counts["smooth"] + counts["featured_or_disk"]) / len(labels) >= 0.2
        ):
            return threshold, diagnostics
    return 0.7, diagnostics


def morphology_task(smooth, featured, embeddings, seed, threshold, diagnostics):
    labels, valid = morphology_labels(smooth, featured, threshold)
    counts = {
        "smooth": int(np.sum(labels == 0)),
        "featured_or_disk": int(np.sum(labels == 1)),
        "ambiguous_excluded": int(np.sum(valid & (labels < 0))),
        "invalid_vote_fractions_excluded": int(np.sum(~valid)),
    }
    indices = np.flatnonzero(labels >= 0)
    result = {
        "target_fields": [SMOOTH, FEATURED],
        "threshold": threshold,
        "threshold_selection": "Use 0.7 if both classes have >=20 and >=20% of all selected rows are retained; otherwise try 0.6 with the same criteria; if neither qualifies, retain 0.7 and skip classification if either class has <4",
        "threshold_diagnostics": diagnostics,
        "label_rule": f"smooth=0 if smooth fraction >= {threshold} and > featured fraction; featured/disk=1 if featured fraction >= {threshold} and > smooth fraction; otherwise exclude",
        "class_counts": counts,
        "usable_examples": int(len(indices)),
        "excluded_examples": int(len(labels) - len(indices)),
        "retained_fraction": float(len(indices) / len(labels)),
    }
    if min(counts["smooth"], counts["featured_or_disk"]) < 4:
        result["skipped_reason"] = f"Fewer than four examples in at least one class at threshold {threshold}"
        return result

    train, test = train_test_split(
        indices, test_size=0.25, random_state=seed, stratify=labels[indices]
    )
    probe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )
    probe.fit(embeddings[train], labels[train])
    predicted = probe.predict(embeddings[test])
    probabilities = probe.predict_proba(embeddings[test])[:, 1]
    majority = int(np.bincount(labels[train]).argmax())
    baseline = np.full(len(test), majority, dtype=np.int8)
    result.update({
        "train_examples": int(len(train)),
        "test_examples": int(len(test)),
        "train_class_counts": {
            "smooth": int(np.sum(labels[train] == 0)),
            "featured_or_disk": int(np.sum(labels[train] == 1)),
        },
        "test_class_counts": {
            "smooth": int(np.sum(labels[test] == 0)),
            "featured_or_disk": int(np.sum(labels[test] == 1)),
        },
        "logistic_probe": {
            "accuracy": float(accuracy_score(labels[test], predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], predicted)),
            "f1_featured_or_disk": float(f1_score(labels[test], predicted, pos_label=1)),
        },
        "majority_baseline": {
            "class": "smooth" if majority == 0 else "featured_or_disk",
            "accuracy": float(accuracy_score(labels[test], baseline)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test], baseline)),
            "f1_featured_or_disk": float(f1_score(labels[test], baseline, pos_label=1, zero_division=0)),
        },
    })
    if len(np.unique(labels[test])) == 2:
        result["logistic_probe"]["roc_auc"] = float(roc_auc_score(labels[test], probabilities))
    else:
        result["logistic_probe"]["roc_auc_skipped_reason"] = "Test split contains one class"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.num_examples < 1 or args.batch_size < 1:
        parser.error("--num-examples and --batch-size must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the validated AstroPT inference workflow")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    print("=== AstroPT Multi-Task Validation ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}; requested galaxies: {args.num_examples}", flush=True)
    print("Morphology candidates: 0.7, then 0.6 if retention is insufficient", flush=True)
    print("Loading frozen AstroPT model...", flush=True)
    model = load_astropt(MODEL, path=CHECKPOINT).to("cuda")
    model.eval()
    galproc = GalaxyImageDataset(
        None, spiral=True,
        transform={"images": transforms.Compose([transforms.Lambda(normalise)])},
        modality_registry=model.modality_registry,
    )

    stream = load_dataset(DATASET, split=SPLIT, revision=REVISION, streaming=True)
    missing = set(("image", *TARGETS)) - set(stream.column_names)
    if missing:
        raise ValueError(f"Required dataset fields missing: {sorted(missing)}")
    # Materialize only selected examples so the streaming connection closes before GPU work.
    rows = list(stream.select_columns(("image", *TARGETS)).take(args.num_examples))
    del stream
    if not rows:
        raise ValueError("Dataset stream returned no galaxies")
    targets = {field: np.array([numeric(row[field]) for row in rows]) for field in TARGETS}
    threshold, diagnostics = choose_morphology_threshold(targets[SMOOTH], targets[FEATURED])
    labels, valid_votes = morphology_labels(targets[SMOOTH], targets[FEATURED], threshold)
    print(f"Vote distribution by threshold: {json.dumps(diagnostics)}", flush=True)
    print(
        f"Chosen threshold {threshold}: smooth={np.sum(labels == 0)}, "
        f"featured/disk={np.sum(labels == 1)}, "
        f"ambiguous={np.sum(valid_votes & (labels < 0))}, "
        f"invalid={np.sum(~valid_votes)}",
        flush=True,
    )

    processed = []
    for row in rows:
        galaxy = galproc.process_galaxy(
            torch.from_numpy(np.array(row["image"]).swapaxes(0, 2)).to(float)
        ).to(torch.float)
        processed.append({
            "images": galaxy,
            "images_positions": torch.arange(0, len(galaxy), dtype=torch.long),
        })
    del rows
    loader = DataLoader(processed, batch_size=args.batch_size, num_workers=0)
    embeddings = []
    torch.cuda.reset_peak_memory_stats()
    embedding_start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            z = model.generate_embeddings(batch)["images"]
            embeddings.append(z.detach().cpu().numpy())
    torch.cuda.synchronize()
    embedding_runtime = time.perf_counter() - embedding_start
    X = np.concatenate(embeddings, axis=0)
    if X.shape[0] != len(processed) or not np.isfinite(X).all():
        raise ValueError("Embedding count mismatch or non-finite embeddings")
    del processed, embeddings

    embedding_path = args.output_dir / "multitask_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=X,
        redshift=targets["redshift"],
        elpetro_mass_log=targets["elpetro_mass_log"],
        smooth_fraction=targets[SMOOTH],
        featured_fraction=targets[FEATURED],
        num_examples=np.array(len(X)),
        embedding_dimension=np.array(X.shape[1]),
    )
    tasks = {
        "redshift": regression_task("redshift", targets["redshift"], X, args.seed, args.output_dir),
        "stellar_mass": regression_task("elpetro_mass_log", targets["elpetro_mass_log"], X, args.seed, args.output_dir),
        "morphology": morphology_task(
            targets[SMOOTH], targets[FEATURED], X, args.seed, threshold, diagnostics
        ),
    }
    results = {
        "model": MODEL,
        "checkpoint": CHECKPOINT,
        "astropt_package_version": importlib.metadata.version("astropt"),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": SPLIT,
        "requested_examples": args.num_examples,
        "embedded_examples": int(len(X)),
        "embedding_dimension": int(X.shape[1]),
        "seed": args.seed,
        "test_fraction": 0.25,
        "embedding_runtime_seconds": embedding_runtime,
        "total_runtime_seconds": time.perf_counter() - start,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "tasks": tasks,
    }
    result_path = args.output_dir / "multitask_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")

    print("\n=== Multi-Task Results ===")
    print(f"Embedding shape: ({results['embedded_examples']}, {results['embedding_dimension']})")
    print(f"Embedding runtime: {results['embedding_runtime_seconds']:.2f} s")
    print(f"Total runtime: {results['total_runtime_seconds']:.2f} s")
    for name in ("redshift", "stellar_mass"):
        task = results["tasks"][name]
        print(f"\n{name} ({task['target_field']}): {task['usable_examples']} usable")
        if "skipped_reason" in task:
            print(f"Train/test: skipped ({task['skipped_reason']})")
            continue
        print(f"Train/test: {task['train_examples']}/{task['test_examples']}")
        for label, key in (("Linear probe", "linear_probe"), ("Mean baseline", "mean_baseline")):
            metrics = task[key]
            print(
                f"{label}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, "
                f"R²={metrics['r2']:.6f}"
            )

    morphology = results["tasks"]["morphology"]
    counts = morphology["class_counts"]
    print(f"\nMorphology threshold: {morphology['threshold']}")
    print(
        f"Smooth={counts['smooth']}, featured/disk={counts['featured_or_disk']}, "
        f"excluded={morphology['excluded_examples']} "
        f"(ambiguous={counts['ambiguous_excluded']}, "
        f"invalid={counts['invalid_vote_fractions_excluded']})"
    )
    if "skipped_reason" in morphology:
        print(f"Train/test: skipped ({morphology['skipped_reason']})")
    else:
        print(f"Train/test: {morphology['train_examples']}/{morphology['test_examples']}")
        probe_metrics = morphology["logistic_probe"]
        baseline = morphology["majority_baseline"]
        print(
            f"Logistic probe: accuracy={probe_metrics['accuracy']:.6f}, "
            f"balanced accuracy={probe_metrics['balanced_accuracy']:.6f}, "
            f"F1={probe_metrics['f1_featured_or_disk']:.6f}"
        )
        if "roc_auc" in probe_metrics:
            print(f"ROC-AUC={probe_metrics['roc_auc']:.6f}")
        else:
            print(f"ROC-AUC skipped: {probe_metrics['roc_auc_skipped_reason']}")
        print(
            f"Majority baseline: accuracy={baseline['accuracy']:.6f}, "
            f"balanced accuracy={baseline['balanced_accuracy']:.6f}"
        )

    print(f"\nJSON results: {result_path}")
    print(f"Embeddings NPZ: {embedding_path}")
    for task in results["tasks"].values():
        if "plot" in task:
            print(f"Plot: {args.output_dir / task['plot']}")
    print("=== MULTITASK VALIDATION COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
