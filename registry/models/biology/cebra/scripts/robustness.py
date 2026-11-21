#!/usr/bin/env python3

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

data_dir = (ROOT / config["dataset"]["data_dir"]).resolve()
data_dir.mkdir(parents=True, exist_ok=True)
os.environ["CEBRA_DATADIR"] = str(data_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
from cebra.integrations.sklearn.utils import check_device

import cebra
import cebra.datasets


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def add_noise(neural_test, train_std, scale, rng):
    noise = rng.standard_normal(neural_test.shape).astype(np.float32)
    return neural_test + noise * (scale * train_std)


def zero_features(neural_test, fraction, rng):
    out = neural_test.copy()
    n_features = out.shape[1]
    n_drop = int(round(fraction * n_features))
    dropped = np.sort(rng.choice(n_features, size=n_drop, replace=False))
    out[:, dropped] = 0
    return out, dropped


def position_scores(decoder_model, embedding, labels_test, decoder):
    pos_col = int(decoder["position_column"])
    y_test = labels_test[:, pos_col]
    pred = np.asarray(decoder_model.predict(embedding))
    err = np.abs(pred - y_test)
    cm = float(decoder["position_to_cm_scale"])
    return {
        "r2_position": float(sklearn.metrics.r2_score(y_test, pred)),
        "median_absolute_error_position": float(np.median(err)),
        "median_absolute_error_position_cm": float(np.median(err) * cm),
    }


print(f"cebra {cebra.__version__}")
print(f"torch {torch.__version__}")

seed = int(config["seed"]["value"])
torch.manual_seed(seed)
np.random.seed(seed)

name = config["dataset"]["name"]
dataset = cebra.datasets.init(name)
neural = np.array(to_numpy(dataset.neural), dtype=np.float32, copy=True)
labels = np.array(to_numpy(dataset.continuous_index), dtype=np.float32, copy=True)
print(f"neural shape {neural.shape}")

test_ratio = float(config["split"]["test_ratio"])
split_idx = int(len(neural) * (1.0 - test_ratio))
neural_train, neural_test = neural[:split_idx], neural[split_idx:]
labels_train = labels[:split_idx]
labels_test = labels[split_idx:]
train_std = neural_train.std(axis=0).astype(np.float32)
print(f"split train {len(neural_train)} test {len(neural_test)}")

model_config = dict(config["model"])
device = check_device(model_config["device"])
print(f"fitting model for {model_config['max_iterations']} iterations on {device}")
model = cebra.CEBRA(**model_config)
model.fit(neural_train, labels_train)
emb_train = np.asarray(model.transform(neural_train))

decoder = config["decoder"]
pos_decoder = cebra.KNNDecoder(
    n_neighbors=int(decoder["n_neighbors"]),
    metric=decoder["metric"],
)
pos_decoder.fit(emb_train, labels_train[:, int(decoder["position_column"])])

rng = np.random.default_rng(seed)
conditions = [("baseline", neural_test, [])]
for scale in config["robustness"]["noise_scales"]:
    conditions.append((f"noise_{scale}", add_noise(neural_test, train_std, float(scale), rng), []))
for fraction in config["robustness"]["zero_fractions"]:
    dropped_input, dropped = zero_features(neural_test, float(fraction), rng)
    conditions.append((f"zero_{fraction}", dropped_input, dropped.tolist()))

out_dir = ROOT / "results" / "robustness"
out_dir.mkdir(parents=True, exist_ok=True)

metrics = {}
for label, neural_cond, dropped in conditions:
    emb_test = np.asarray(model.transform(neural_cond))
    scores = position_scores(pos_decoder, emb_test, labels_test, decoder)
    if dropped:
        scores["dropped_features"] = dropped
    metrics[label] = scores
    print(
        f"{label} r2_position={scores['r2_position']:.6f} "
        f"median_abs_error_cm={scores['median_absolute_error_position_cm']:.6f}"
    )

labels_order = list(metrics)
errors = [metrics[label]["median_absolute_error_position_cm"] for label in labels_order]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels_order, errors)
ax.set_ylabel("median position error (cm)")
fig.tight_layout()
fig.savefig(out_dir / "median_error.png", dpi=150)
plt.close(fig)

(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
metadata = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "python_version": platform.python_version(),
    "cebra_version": cebra.__version__,
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device": device,
    "seed": seed,
    "dataset_name": name,
    "split_index": split_idx,
    "noise_scales": config["robustness"]["noise_scales"],
    "zero_fractions": config["robustness"]["zero_fractions"],
}
(out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"wrote {out_dir / 'metrics.json'}")
print("ROBUSTNESS PASS")
