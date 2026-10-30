"""
Darcy Inference Script (Reproducible)

Purpose:
    Run a single forward pass of the Fourier Neural Operator (FNO)
    on the built-in Darcy dataset included in the NeuralOperator repository.

Model Source:
    NeuralOperator (MIT License)
    https://github.com/neuraloperator/neuraloperator
    Model architecture defined in neuraloperator/neuralop/models/fno.py

Dataset Source:
    neuraloperator/neuralop/data/datasets/data/darcy_test_16.pt
    Contains boolean permeability fields ("x") and pressure solutions ("y").

Environment Requirements:
    Python 3.10
    PyTorch 2.5.1
    CUDA 12.4 (optional)
    NeuralOperator installed via `pip install -e ../neuraloperator`

Expected Output:
    Shape: (1, 1, 16, 16)
    Values: Smooth, low-variance pressure field
    No NaNs or infinities

This script is used for the Reproducible Inference assignment on AI Makerspace.
"""

import torch
from neuralop.models.fno import FNO


DATA_PATH = "../neuraloperator/neuralop/data/datasets/data/darcy_test_16.pt"


def main():

    print("=== Darcy Inference Script ===")
    print("Dataset path:", DATA_PATH)

    # Load dataset
    data = torch.load(DATA_PATH)

    # Convert boolean tensors to float
    x = data["x"][0].float()      # (16, 16)
    y_true = data["y"][0].float()

    # Add batch + channel dimensions
    x = x.unsqueeze(0).unsqueeze(0)      # (1, 1, 16, 16)
    y_true = y_true.unsqueeze(0).unsqueeze(0)

    print("Input shape:", x.shape)
    print("Target shape:", y_true.shape)

    # Build FNO model
    model = FNO(
        n_modes=(12, 12),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        positional_embedding=None
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    model = model.to(device)
    x = x.to(device)

    # Run inference
    with torch.no_grad():
        pred = model(x)

    print("Output shape:", pred.shape)

    # Smoke test: shape check
    assert pred.shape == (1, 1, 16, 16), "Unexpected output shape!"

    # Numerical sanity checks
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        raise ValueError("Prediction contains NaNs or infinities!")

    # Convert to numpy for statistics
    pred_np = pred.squeeze().cpu().numpy()

    print("Prediction stats:")
    print("min:", pred_np.min())
    print("max:", pred_np.max())
    print("mean:", pred_np.mean())

    print("=== Inference complete ===")


if __name__ == "__main__":
    main()