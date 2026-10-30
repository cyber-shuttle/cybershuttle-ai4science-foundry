"""
Darcy Checkpoint Inference Script

Loads a trained FNO checkpoint produced by train_darcy.py and
runs inference on a sample from the Darcy test dataset.

This demonstrates:

    Train -> Checkpoint -> Inference

rather than:

    Random Initialization -> Inference
"""

from pathlib import Path
import sys

import torch

# Allow importing config files from NeuralOperator repo
sys.path.insert(0, "../")

from config.darcy_config import Default
from neuralop import get_model


DATA_PATH = "../neuralop/data/datasets/data/darcy_test_16.pt"
CHECKPOINT_PATH = "./checkpoints/model_state_dict.pt"


def main():

    print("=== Darcy Checkpoint Inference ===")

    print("Dataset:", DATA_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)

    # -----------------------------
    # Device selection
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Running on device:", device)

    # -----------------------------
    # Build the same model used
    # during Darcy training
    # -----------------------------
    config = Default()

    model = get_model(config)
    model = model.to(device)

    # -----------------------------
    # Load trained weights
    # -----------------------------
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("Checkpoint loaded successfully.")

    # -----------------------------
    # Load dataset
    # -----------------------------
    data = torch.load(DATA_PATH)

    x = data["x"][0].float()
    y_true = data["y"][0].float()

    x = x.unsqueeze(0).unsqueeze(0)
    y_true = y_true.unsqueeze(0).unsqueeze(0)

    x = x.to(device)
    y_true = y_true.to(device)

    print("Input shape:", x.shape)
    print("Target shape:", y_true.shape)

    # -----------------------------
    # Run inference
    # -----------------------------
    with torch.no_grad():
        pred = model(x)

    print("Output shape:", pred.shape)

    # -----------------------------
    # Sanity checks
    # -----------------------------
    assert pred.shape == y_true.shape

    if torch.isnan(pred).any():
        raise ValueError("Prediction contains NaNs")

    if torch.isinf(pred).any():
        raise ValueError("Prediction contains Infs")

    # -----------------------------
    # Simple metrics
    # -----------------------------
    mse = torch.mean((pred - y_true) ** 2).item()

    pred_np = pred.squeeze().cpu().numpy()

    print("\nPrediction statistics:")
    print("min :", pred_np.min())
    print("max :", pred_np.max())
    print("mean:", pred_np.mean())

    print("\nMSE against target:", mse)

    print("\n=== Checkpoint inference complete ===")


if __name__ == "__main__":
    main()