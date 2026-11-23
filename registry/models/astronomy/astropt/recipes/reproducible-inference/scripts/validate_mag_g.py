import json
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision import transforms

from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt


NUM_EXAMPLES = 1000
BATCH_SIZE = 32


def normalise(x):
    std, mean = torch.std_mean(x, dim=1, keepdim=True)
    return (x - mean) / (std + 1e-8)


def data_transforms():
    return transforms.Compose([
        transforms.Lambda(normalise),
    ])


def process_galaxy(idx, func):
    galaxy = func(
        torch.from_numpy(
            np.array(idx["image"]).swapaxes(0, 2)
        ).to(float)
    ).to(torch.float)

    positions = torch.arange(
        0, len(galaxy), dtype=torch.long
    )

    return {
        "images": galaxy,
        "images_positions": positions,
        "mag_g": idx["mag_g"],
    }


print("=== AstroPT mag_g Validation ===")
print("Device:", torch.cuda.get_device_name(0))
print("Examples:", NUM_EXAMPLES)

model = load_astropt(
    "Smith42/astroPT_v2.0",
    path="astropt/095M",
)

model = model.to("cuda")
model.eval()

galproc = GalaxyImageDataset(
    None,
    spiral=True,
    transform={"images": data_transforms()},
    modality_registry=model.modality_registry,
)

streaming_ds = (
    load_dataset(
        "Smith42/galaxies",
        split="test",
        revision="v2.0",
        streaming=True,
    )
    .select_columns(("image", "mag_g"))
    .filter(lambda x: x["mag_g"] is not None)
    .map(partial(process_galaxy, func=galproc.process_galaxy))
    .with_format("torch")
    .take(NUM_EXAMPLES)
)

# Materialize the requested examples before GPU inference so the
# Hugging Face streaming connection is fully consumed and closed.
ds = list(streaming_ds)
del streaming_ds

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

all_embeddings = []
all_targets = []

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.inference_mode():
    for batch in loader:
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to("cuda")

        z = model.generate_embeddings(batch)["images"]

        all_embeddings.append(z.detach().cpu().numpy())
        all_targets.append(
            batch["mag_g"].detach().cpu().numpy()
        )

torch.cuda.synchronize()
embedding_runtime = time.perf_counter() - start

X = np.concatenate(all_embeddings, axis=0)
y = np.concatenate(all_targets, axis=0)

halfway = len(X) // 2

X_train = X[:halfway]
X_test = X[halfway:]

y_train = y[:halfway]
y_test = y[halfway:]

probe = LinearRegression()
probe.fit(X_train, y_train)

pred = probe.predict(X_test)

probe_mse = mean_squared_error(y_test, pred)
probe_mae = mean_absolute_error(y_test, pred)
probe_r2 = r2_score(y_test, pred)

baseline_value = float(np.mean(y_train))
baseline_pred = np.full_like(y_test, baseline_value)

baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

results = {
    "model": "Smith42/astroPT_v2.0",
    "checkpoint": "astropt/095M",
    "dataset": "Smith42/galaxies",
    "dataset_revision": "v2.0",
    "split": "test",
    "target": "mag_g",
    "num_examples": int(len(X)),
    "train_examples": int(len(X_train)),
    "test_examples": int(len(X_test)),
    "embedding_dimension": int(X.shape[1]),
    "embedding_runtime_seconds": embedding_runtime,
    "peak_cuda_memory_gb":
        torch.cuda.max_memory_allocated() / 1024**3,
    "linear_probe": {
        "mse": float(probe_mse),
        "mae": float(probe_mae),
        "r2": float(probe_r2),
    },
    "mean_baseline": {
        "value": baseline_value,
        "mse": float(baseline_mse),
        "mae": float(baseline_mae),
        "r2": float(baseline_r2),
    },
}

with open("outputs/validation_mag_g.json", "w") as f:
    json.dump(results, f, indent=2)

np.save("outputs/validation_embeddings.npy", X)
np.save("outputs/validation_mag_g.npy", y)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=10, alpha=0.6)
low = min(y_test.min(), pred.min())
high = max(y_test.max(), pred.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("True mag_g")
plt.ylabel("Predicted mag_g")
plt.title("AstroPT 95M Linear Probe Validation")
plt.tight_layout()
plt.savefig("outputs/mag_g_probe.png", dpi=150)
plt.close()

print("Embedding shape:", X.shape)
print(f"Embedding runtime: {embedding_runtime:.2f} seconds")
print(
    f"Peak CUDA memory: "
    f"{results['peak_cuda_memory_gb']:.3f} GB"
)

print()
print("--- Linear Probe ---")
print(f"MSE: {probe_mse:.6f}")
print(f"MAE: {probe_mae:.6f}")
print(f"R2:  {probe_r2:.6f}")

print()
print("--- Mean Baseline ---")
print(f"MSE: {baseline_mse:.6f}")
print(f"MAE: {baseline_mae:.6f}")
print(f"R2:  {baseline_r2:.6f}")

print()
print(
    "Probe beats baseline MSE:",
    bool(probe_mse < baseline_mse)
)
print(
    "Probe beats baseline MAE:",
    bool(probe_mae < baseline_mae)
)

print("Saved: outputs/validation_mag_g.json")
print("Saved: outputs/mag_g_probe.png")
print("=== VALIDATION COMPLETE ===")
