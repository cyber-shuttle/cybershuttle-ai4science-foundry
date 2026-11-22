import json
import time
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

def load_qm9_u0(root="./data"):
    dataset = QM9(root=root)
    # PyG maps atomization energy targets [7,8,9,10] -> [12,13,14,15]
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]
    return dataset

def predict_one(model, mol, target=7, device="cuda"):
    """Stateless single-molecule prediction contract for Custos gateway dispatch."""
    t0 = time.perf_counter()
    mol = mol.to(device)
    with torch.no_grad():
        pred = model(mol.z, mol.pos, batch=None).item()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    actual = mol.y[0, target].item()
    return {
        "target": "U0 (Target 7)",
        "predicted_ev": round(pred, 6),
        "actual_ev": round(actual, 6),
        "abs_error_mev": round(abs(pred - actual) * 1000.0, 4),
        "gpu_latency_ms": round(latency_ms, 3),
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | PyTorch: {torch.__version__}")

    dataset = load_qm9_u0("./data")
    target = 7

    # 1. Checkpoint load timing
    t_load = time.perf_counter()
    model, (_, _, test_dataset) = DimeNetPlusPlus.from_qm9_pretrained(
        root="./data", dataset=dataset, target=target
    )
    model = model.to(device).eval()
    print(f"[MODEL LOAD] Pre-trained DimeNet++ (Target 7) loaded in {time.perf_counter() - t_load:.2f}s")

    # 2. Single-molecule test (mirrors Temmy's 10-molecule Custos gateway benchmark)
    results_10 = [predict_one(model, test_dataset[i], target, device) for i in range(10)]
    mae_10 = sum(r["abs_error_mev"] for r in results_10) / 10.0
    lat_10 = sum(r["gpu_latency_ms"] for r in results_10) / 10.0
    print(f"[10-MOL GATEWAY BENCHMARK] MAE: {mae_10:.4f} meV | Avg GPU Latency: {lat_10:.2f} ms/mol")
    print("Sample Gateway JSON Payload:", json.dumps(results_10[0]))

    # 3. Full 10,000-molecule QM9 test split on PACE GPU
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    total_err, total_mols = 0.0, 0
    t_full = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.z, batch.pos, batch.batch).view(-1)
            total_err += torch.abs(preds - batch.y[:, target].view(-1)).sum().item()
            total_mols += batch.num_graphs
    walltime = time.perf_counter() - t_full
    mae_ev = total_err / total_mols
    print(f"[FULL TEST SET (n={total_mols})] MAE: {mae_ev:.6f} eV ({mae_ev*1000:.4f} meV) | Total Time: {walltime:.2f}s")