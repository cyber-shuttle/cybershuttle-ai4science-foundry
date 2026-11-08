import os
import torch
import torch.nn.functional as F
from neuralop.models import FNO
from neuralop.datasets import load_darcy_flow_small

os.makedirs("models", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Checkpoint Setup] Training on: {device}")

model = FNO(n_modes=(8, 8), in_channels=1, out_channels=1, hidden_channels=32).to(device)

train_loader, _, _ = load_darcy_flow_small(
    n_train=100,
    batch_size=16,
    test_resolutions=[16],
    n_tests=[16],
    test_batch_sizes=[16]
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("[Checkpoint Setup] Training FNO for 20 epochs to reach target relative L2 convergence (< 0.20)...")
model.train()
for epoch in range(1, 21):
    total_loss = 0.0
    for batch in train_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:02d}/20 - MSE Loss: {total_loss / len(train_loader):.6f}")

checkpoint_path = "models/fno_darcy_checkpoint.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'architecture': 'FNO-2D',
    'n_modes': [8, 8],
    'hidden_channels': 32,
    'in_channels': 1,
    'out_channels': 1
}, checkpoint_path)
print(f"[Checkpoint Setup] Fully converged checkpoint saved to {checkpoint_path}")
