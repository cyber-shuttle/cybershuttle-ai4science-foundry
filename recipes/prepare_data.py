import os
import torch
from neuralop.datasets import load_darcy_flow_small

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("==> Loading Darcy Flow dataset via neuralop...")
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=100,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[50, 50],
    test_batch_sizes=[16, 16]
)

# Extract a test batch from resolution 16
test_loader_16 = test_loaders[16]
batch = next(iter(test_loader_16))
x = batch['x']
y = batch['y']

print(f"Extracted evaluation tensor - x shape: {list(x.shape)}, y shape: {list(y.shape)}")
torch.save({'x': x, 'y': y}, "data/darcy_test_sample.pt")
print("Saved clean evaluation batch to data/darcy_test_sample.pt")
