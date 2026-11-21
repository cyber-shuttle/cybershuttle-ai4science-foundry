#!/usr/bin/env python3

import json
import os
import platform
import random
import socket
import time
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

import cebra
import cebra.datasets
import sklearn
import sklearn.metrics
import torch
from cebra.integrations.sklearn.utils import check_device


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_model(cebra, model_config, neural_train, labels_train, neural_test):
    model = cebra.CEBRA(**model_config)
    t0 = time.perf_counter()
    model.fit(neural_train, labels_train)
    seconds = time.perf_counter() - t0
    emb_train = np.asarray(model.transform(neural_train))
    emb_test = np.asarray(model.transform(neural_test))
    return model, emb_train, emb_test, seconds


def decode(cebra, metrics_mod, decoder, emb_train, emb_test, labels, split_idx):
    k = int(decoder["n_neighbors"])
    metric = decoder["metric"]
    pos_col = int(decoder["position_column"])
    dir_col = int(decoder["direction_column"])
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]

    pos_decoder = cebra.KNNDecoder(n_neighbors=k, metric=metric)
    dir_decoder = cebra.KNNDecoder(n_neighbors=k, metric=metric)
    pos_decoder.fit(emb_train, y_train[:, pos_col])
    dir_decoder.fit(emb_train, y_train[:, dir_col])
    pos_pred = np.asarray(pos_decoder.predict(emb_test))
    dir_pred = np.asarray(dir_decoder.predict(emb_test))

    pred = np.stack([pos_pred, dir_pred], axis=1)
    target = y_test[:, [pos_col, dir_col]]
    err = np.abs(pos_pred - y_test[:, pos_col])
    cm = float(decoder["position_to_cm_scale"])
    scores = {
        "r2_position_and_direction": float(metrics_mod.r2_score(target, pred)),
        "r2_position": float(metrics_mod.r2_score(y_test[:, pos_col], pos_pred)),
        "median_absolute_error_position": float(np.median(err)),
        "median_absolute_error_position_cm": float(np.median(err) * cm),
    }
    arrays = {
        "embedding_train": emb_train,
        "embedding_test": emb_test,
        "position_true": y_test[:, pos_col],
        "direction_true": y_test[:, dir_col],
        "position_pred": pos_pred,
        "direction_pred": dir_pred,
    }
    return scores, arrays


def print_scores(label, scores):
    print(
        f"{label} "
        f"r2_position={scores['r2_position']:.6f} "
        f"median_abs_error={scores['median_absolute_error_position']:.6f} "
        f"median_abs_error_cm={scores['median_absolute_error_position_cm']:.6f}"
    )


print(f"cebra {cebra.__version__}")
print(f"torch {torch.__version__}")
print(f"numpy {np.__version__}")
print(f"sklearn {sklearn.__version__}")

seed = int(config["seed"]["value"])
set_seed(seed, torch)

name = config["dataset"]["name"]
print(f"loading {name} from {data_dir}")
dataset = cebra.datasets.init(name)
neural = np.array(to_numpy(dataset.neural), copy=True)
labels = np.array(to_numpy(dataset.continuous_index), copy=True)
print(f"neural shape {neural.shape}, dtype {neural.dtype}")
print(f"label shape {labels.shape}, dtype {labels.dtype}")

test_ratio = float(config["split"]["test_ratio"])
split_idx = int(len(neural) * (1.0 - test_ratio))
neural_train, neural_test = neural[:split_idx], neural[split_idx:]
labels_train = labels[:split_idx]
print(f"split train {len(neural_train)} test {len(neural_test)}")

model_config = dict(config["model"])
device = check_device(model_config["device"])
print(
    f"fitting behavior model for {model_config['max_iterations']} iterations "
    f"on {model_config['device']} (resolved {device})"
)

started = datetime.now(timezone.utc)
behavior_model, emb_train, emb_test, behavior_seconds = fit_model(
    cebra, model_config, neural_train, labels_train, neural_test
)

scores, preds = decode(
    cebra, sklearn.metrics, config["decoder"], emb_train, emb_test, labels, split_idx
)
preds["loss"] = np.asarray(behavior_model.state_dict_["loss"])
print_scores("behavior", scores)

shuffled_scores = None
shuffled_seconds = None
shuffled_loss = None
results = ROOT / "results"
results.mkdir(parents=True, exist_ok=True)

if config["shuffle_control"]["enabled"]:
    set_seed(seed, torch)
    shuffled_labels = np.random.permutation(labels_train)
    print(f"fitting shuffled model for {model_config['max_iterations']} iterations on {device}")
    shuffled_model, sh_train, sh_test, shuffled_seconds = fit_model(
        cebra, model_config, neural_train, shuffled_labels, neural_test
    )
    loss = np.asarray(shuffled_model.state_dict_["loss"])
    shuffled_loss = float(loss[-1])
    shuffled_scores, sh_preds = decode(
        cebra, sklearn.metrics, config["decoder"], sh_train, sh_test, labels, split_idx
    )
    sh_preds["loss"] = loss
    print_scores("shuffled", shuffled_scores)
    np.save(results / "embedding_posdir_shuffled.npy", np.vstack([sh_train, sh_test]))
    np.savez(results / "predictions_posdir_shuffled.npz", **sh_preds)

np.save(results / "embedding_posdir.npy", np.vstack([emb_train, emb_test]))
np.savez(results / "predictions_posdir.npz", **preds)

finished = datetime.now(timezone.utc)
behavior_loss = float(np.asarray(behavior_model.state_dict_["loss"])[-1])
metrics = {"behavior": scores, "shuffled": shuffled_scores}
metadata = {
    "started_at_utc": started.isoformat(),
    "finished_at_utc": finished.isoformat(),
    "hostname": socket.gethostname(),
    "python_version": platform.python_version(),
    "cebra_version": cebra.__version__,
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device": device,
    "seed": seed,
    "dataset_name": name,
    "neural_shape": list(neural.shape),
    "split_index": split_idx,
    "behavior_fit_seconds": behavior_seconds,
    "shuffled_fit_seconds": shuffled_seconds,
    "behavior_final_loss": behavior_loss,
    "shuffled_final_loss": shuffled_loss,
}
(results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
(results / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("wrote results/metrics.json")
print("RUN PASS")
