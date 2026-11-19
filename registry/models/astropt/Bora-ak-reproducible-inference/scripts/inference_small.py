import json
import time
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from astropt.local_datasets import GalaxyImageDataset
from astropt.model_utils import load_astropt


NUM_EXAMPLES = 8
BATCH_SIZE = 8


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

    galaxy_positions = torch.arange(
        0, len(galaxy), dtype=torch.long
    )

    return {
        "images": galaxy,
        "images_positions": galaxy_positions,
        "mag_g": idx["mag_g"],
    }


print("=== AstroPT Reproducible Inference ===")
print("Device:", torch.cuda.get_device_name(0))

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

ds = (
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

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    num_workers=0,
)

batch = next(iter(loader))

for key, value in batch.items():
    if torch.is_tensor(value):
        batch[key] = value.to("cuda")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    embeddings = model.generate_embeddings(batch)["images"]

torch.cuda.synchronize()
runtime = time.perf_counter() - start

embeddings_np = embeddings.detach().cpu().numpy()
mag_g_np = batch["mag_g"].detach().cpu().numpy()

finite = bool(np.isfinite(embeddings_np).all())

np.save("outputs/embeddings_small.npy", embeddings_np)
np.save("outputs/mag_g_small.npy", mag_g_np)

results = {
    "model": "Smith42/astroPT_v2.0",
    "checkpoint": "astropt/095M",
    "dataset": "Smith42/galaxies",
    "dataset_revision": "v2.0",
    "split": "test",
    "num_examples": NUM_EXAMPLES,
    "embedding_shape": list(embeddings_np.shape),
    "all_embeddings_finite": finite,
    "runtime_seconds": runtime,
    "peak_cuda_memory_gb":
        torch.cuda.max_memory_allocated() / 1024**3,
}

with open("outputs/inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Examples:", NUM_EXAMPLES)
print("Embedding shape:", embeddings_np.shape)
print("All embeddings finite:", finite)
print(f"Inference runtime: {runtime:.4f} seconds")
print(
    f"Peak CUDA memory: "
    f"{results['peak_cuda_memory_gb']:.3f} GB"
)
print("Saved: outputs/embeddings_small.npy")
print("Saved: outputs/mag_g_small.npy")
print("Saved: outputs/inference_results.json")
print("=== SUCCESS ===")
