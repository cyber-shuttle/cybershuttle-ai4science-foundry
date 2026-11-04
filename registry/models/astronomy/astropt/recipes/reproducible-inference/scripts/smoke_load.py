import time
import torch
import importlib.metadata
from astropt.model_utils import load_astropt

print("=== AstroPT Assignment 2 Smoke Test ===")
print("AstroPT version:", importlib.metadata.version("astropt"))
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

start = time.perf_counter()

model = load_astropt(
    repo_id="smith42/astropt_v2.0",
    path="astropt/095M",
    weights_filename="ckpt.pt",
)

model = model.to("cuda")
model.eval()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

params = sum(p.numel() for p in model.parameters())

print("Model loaded successfully.")
print(f"Parameter count: {params:,}")
print(f"Load time: {elapsed:.2f} seconds")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"CUDA peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
