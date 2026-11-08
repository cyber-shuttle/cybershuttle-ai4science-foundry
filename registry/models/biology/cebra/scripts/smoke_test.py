#!/usr/bin/env python3

import os
import random
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
import torch


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


print(f"cebra {cebra.__version__}")
print(f"torch {torch.__version__}")
print(f"numpy {np.__version__}")

seed = int(config["seed"]["value"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

name = config["dataset"]["name"]
print(f"loading {name} from {data_dir}")
dataset = cebra.datasets.init(name)
neural = to_numpy(dataset.neural)
labels = to_numpy(dataset.continuous_index)
print(f"neural shape {neural.shape}, dtype {neural.dtype}")
print(f"label shape {labels.shape}, dtype {labels.dtype}")

test_ratio = float(config["split"]["test_ratio"])
split_idx = int(len(neural) * (1.0 - test_ratio))
neural_train = neural[:split_idx]
neural_test = neural[split_idx:]
print(f"split train {len(neural_train)} test {len(neural_test)}")

model_config = dict(config["model"])
n_iter = int(config["smoke"]["max_iterations"])
model_config["max_iterations"] = n_iter
print(f"fitting {n_iter} iterations on {model_config['device']}")
model = cebra.CEBRA(**model_config)
model.fit(neural_train, labels[:split_idx])

embedding = np.asarray(model.transform(neural_train))
print(f"embedding shape {embedding.shape}, dtype {embedding.dtype}")
print("SMOKE PASS")
