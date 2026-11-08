import os
import sys
import time
import json
import torch
import numpy as np
from neuralop.models import FNO


def run_fno_inference():
    print("=" * 65)
    print("  AI Makerspace: Reproducible FNO-2D Inference Pipeline")
    print("=" * 65)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device Selection] Running inference on: {device}")
    if torch.cuda.is_available():
        print(f"[GPU Hardware] {torch.cuda.get_device_name(0)}")
        print(f"[CUDA Version] {torch.version.cuda}")

    data_path = "data/darcy_test_sample.pt" if os.path.exists("data/darcy_test_sample.pt") else "darcy_test_sample.pt"
    model_path = "models/fno_darcy_checkpoint.pt" if os.path.exists("models/fno_darcy_checkpoint.pt") else "fno_darcy_checkpoint.pt"

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Missing artifacts. Run prepare_data.py and export_model_checkpoint.py first.")
        sys.exit(1)

    # 1. Load Checkpoint and Metadata
    print(f"[Model Provenance] Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    n_modes = tuple(checkpoint['n_modes'])
    hidden_channels = checkpoint['hidden_channels']
    model = FNO(
        n_modes=n_modes,
        in_channels=checkpoint['in_channels'],
        out_channels=checkpoint['out_channels'],
        hidden_channels=hidden_channels
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[Model Init] FNO architecture loaded (modes={n_modes}, hidden={hidden_channels})")

    # 2. Data Loading
    print(f"[Data Provenance] Ingesting test evaluation batch from: {data_path}")
    dataset = torch.load(data_path, map_location=device, weights_only=False)
    x_test = dataset['x'].to(device)
    y_true = dataset['y'].to(device)
    num_samples = x_test.shape[0]
    print(f"[Data Dimensions] Input batch: {list(x_test.shape)}, Target batch: {list(y_true.shape)}")

    # 3. Smoke Test Check
    t0 = time.perf_counter()
    with torch.no_grad():
        smoke_sample = x_test[:1]
        smoke_pred = model(smoke_sample)
    smoke_latency_ms = (time.perf_counter() - t0) * 1000.0
    assert smoke_pred.shape == y_true[:1].shape, f"Shape mismatch: {smoke_pred.shape} vs {y_true[:1].shape}"
    print(f"[Smoke Test] PASSED | Single-sample output shape: {list(smoke_pred.shape)} | Latency: {smoke_latency_ms:.2f} ms")

    # 4. Substantive Scientific Validation (Relative L2 Error)
    t_start = time.perf_counter()
    with torch.no_grad():
        y_pred = model(x_test)
        diff = torch.norm(y_pred.reshape(num_samples, -1) - y_true.reshape(num_samples, -1), 2, dim=1)
        norm_true = torch.norm(y_true.reshape(num_samples, -1), 2, dim=1)
        rel_l2_errors = (diff / norm_true).cpu().numpy()
    batch_latency_s = time.perf_counter() - t_start
    mean_rel_l2 = float(np.mean(rel_l2_errors))
    std_rel_l2 = float(np.std(rel_l2_errors))
    mse_val = float(torch.nn.functional.mse_loss(y_pred, y_true).item())

    # Scientific threshold: error norm < 0.20 for coarse 16x16 resolution
    validation_passed = mean_rel_l2 < 0.20
    print(f"[Substantive Check] Batch Latency: {batch_latency_s:.4f} s ({num_samples} samples)")
    print(f"[Substantive Check] Mean Relative L2 Error: {mean_rel_l2:.4f} (Std: {std_rel_l2:.4f})")
    print(f"[Substantive Check] Mean Squared Error (MSE): {mse_val:.6f}")
    print(f"[Validation Status] Relative L2 < 0.20 Threshold: {'PASSED' if validation_passed else 'FAILED'}")

    # 5. Export Output Artifacts
    os.makedirs("outputs", exist_ok=True)
    results = {
        "execution_device": str(device),
        "gpu_hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "smoke_test": {
            "status": "PASSED",
            "latency_ms": round(smoke_latency_ms, 2),
            "expected_shape": list(y_true[:1].shape),
            "observed_shape": list(smoke_pred.shape)
        },
        "substantive_validation": {
            "test_samples": num_samples,
            "mean_relative_l2_error": round(mean_rel_l2, 5),
            "std_relative_l2_error": round(std_rel_l2, 5),
            "mean_squared_error": round(mse_val, 6),
            "threshold_applied": 0.20,
            "status": "PASSED" if validation_passed else "FAILED"
        }
    }
    out_file = "outputs/smoke_test_output.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Artifact Export] Written validation metrics to {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_fno_inference()
