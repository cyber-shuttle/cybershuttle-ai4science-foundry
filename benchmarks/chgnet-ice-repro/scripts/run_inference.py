#!/usr/bin/env python3
"""
Run CHGNet 0.3.0 inference for the smoke test or MPtrj subset. This script only runs inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from chgnet.model.model import CHGNet
from pymatgen.core import Lattice, Structure

from metrics import average_errors, component_errors

MODEL_NAME = "0.3.0"
CHECKPOINT_FILENAME = "chgnet_0.3.0_e29f68s314m37.pth.tar"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def to_numpy(item) -> np.ndarray:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    return np.asarray(item)


def serialized_prediction(prediction: dict, n_atoms: int) -> dict:
    e = float(to_numpy(prediction["e"]).reshape(-1)[0])
    f = to_numpy(prediction["f"]).reshape(n_atoms, 3).astype(float)
    s = to_numpy(prediction["s"]).reshape(3, 3).astype(float)
    m = to_numpy(prediction["m"]).reshape(-1).astype(float)
    if m.size != n_atoms:
        raise ValueError(f"Expected {n_atoms} magnetic moments, got {m.size}")
    arrays = [np.array([e]), f, s, m]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("CHGNet returned a non-finite energy/force/stress/moment")
    return {
        "energy_per_atom": e,  # eV/atom (not total eV!)
        "forces_ev_per_angstrom": f.tolist(),
        "stress_gpa": s.tolist(),
        "magmom_mu_b": m.tolist(),
    }


def predict(model: CHGNet, structure: Structure, task: str = "efsm") -> dict:
    # Do not use torch.no_grad: force and stress are energy derivatives.
    return model.predict_structure(structure, task=task)


def runtime_info(args: argparse.Namespace) -> dict:
    import chgnet

    checkpoint = Path(chgnet.__file__).parent / "pretrained" / MODEL_NAME / CHECKPOINT_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected bundled checkpoint not found: {checkpoint}")
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    return {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {name: metadata.version(name) for name in
                     ("chgnet", "torch", "pymatgen", "ase", "numpy")},
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
        "requested_device": args.device,
        "model_checkpoint_id": MODEL_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "slurm": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS",
            "SLURM_JOB_ACCOUNT", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST",
            "SLURM_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_MEM_PER_NODE",
        )},
        "command": sys.argv,
    }


def run_smoke(model: CHGNet, out_dir: Path) -> bool:
    """Run the fixed silicon smoke test."""
    # Silicon diamond conventional cell, 8 atoms.
    structure = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]
    )
    # Move one atom slightly so the finite-difference force check is nonzero.
    structure.translate_sites([0], [0.03, -0.02, 0.015], frac_coords=False)
    n = len(structure)
    p = serialized_prediction(predict(model, structure, "efsm"), n)

    # A rigid translation should not change periodic energy.
    shifted = structure.copy()
    shifted.translate_sites(list(range(n)), [0.13, -0.11, 0.07], frac_coords=False)
    shifted_energy = float(to_numpy(predict(model, shifted, "e")["e"]).reshape(-1)[0])
    translation_error = abs(p["energy_per_atom"] - shifted_energy)

    # CHGNet reports eV/atom, so multiply by N before taking dE/dx.
    h = 0.005  # Angstrom
    plus, minus = structure.copy(), structure.copy()
    plus.translate_sites([0], [h, 0, 0], frac_coords=False)
    minus.translate_sites([0], [-h, 0, 0], frac_coords=False)
    e_plus = float(to_numpy(predict(model, plus, "e")["e"]).reshape(-1)[0]) * n
    e_minus = float(to_numpy(predict(model, minus, "e")["e"]).reshape(-1)[0]) * n
    numeric_force = -(e_plus - e_minus) / (2 * h)
    model_force = float(p["forces_ev_per_angstrom"][0][0])
    force_abs_error = abs(numeric_force - model_force)
    force_tolerance = 0.15 + 0.05 * max(abs(numeric_force), abs(model_force))

    net_force_norm = float(np.linalg.norm(
        np.asarray(p["forces_ev_per_angstrom"]).sum(axis=0)
    ))
    checks = {
        "finite_shapes": {
            "expected": "finite energy, Nx3 forces, 3x3 stress, N moments",
            "observed": f"all finite and correctly shaped; N={n}",
            "pass": True,
        },
        "periodic_translation_invariance": {
            "expected": "change in energy < 0.001 eV/atom",
            "observed_energy_difference_ev_per_atom": translation_error,
            "threshold": 0.001,
            "pass": translation_error < 0.001,
        },
        "newtons_third_law_total_force": {
            "expected": "norm(sum of atom forces) < 0.1 eV/Angstrom",
            "observed_total_force_norm_ev_per_angstrom": net_force_norm,
            "threshold": 0.1,
            "pass": net_force_norm < 0.1,
        },
        "finite_difference_force": {
            "expected": "analytic force ~= -d(E_total)/dx, central difference",
            "observed_model_force_ev_per_angstrom": model_force,
            "observed_numeric_force_ev_per_angstrom": numeric_force,
            "absolute_error_ev_per_angstrom": force_abs_error,
            "step_angstrom": h,
            "tolerance_ev_per_angstrom": force_tolerance,
            "pass": force_abs_error < force_tolerance,
        },
    }
    success = all(check["pass"] for check in checks.values())
    write_json(out_dir / "smoke_predictions.json", {
        "input": "Synthetic periodic diamond silicon, cubic a=5.43 Angstrom; atom 0 displaced by [0.03,-0.02,0.015] A",
        "structure": structure.as_dict(),
        "predictions": p,
    })
    write_json(out_dir / "smoke_validation.json", {
        "status": "PASS" if success else "FAIL",
        "checks": checks,
        "limitations": "Invariance and derivative checks verify internal physical consistency, not DFT accuracy.",
    })
    print(f"SMOKE STATUS: {'PASS' if success else 'FAIL'}", flush=True)
    for name, check in checks.items():
        print(f"  {name}: {'PASS' if check['pass'] else 'FAIL'}", flush=True)
    return success


def run_benchmark(model: CHGNet, subset: Path, out_dir: Path) -> bool:
    payload = json.loads(subset.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Subset JSON must contain a nonempty 'records' list")
    results: list[dict] = []
    failures: list[dict] = []
    errors: list[dict[str, list[float]]] = []
    for i, record in enumerate(records):
        mp_id, frame_id = record["material_id"], record["frame_id"]
        try:
            structure = Structure.from_dict(record["structure"])
            p = serialized_prediction(predict(model, structure, "efsm"), len(structure))
            component = component_errors(p, record["reference"], len(structure))
            error_means = {key: sum(vals) / len(vals) for key, vals in component.items() if vals}
            errors.append(component)
            results.append({
                "material_id": mp_id,
                "frame_id": frame_id,
                "n_atoms": len(structure),
                "prediction": p,
                "mean_absolute_errors": error_means,
            })
        except Exception as exc:
            failures.append({
                "material_id": mp_id,
                "frame_id": frame_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            print(f"FAILED {mp_id}/{frame_id}: {type(exc).__name__}: {exc}", flush=True)
        print(f"Processed {i + 1}/{len(records)}", flush=True)

    summary = {
        "status": "COMPLETED" if len(results) == len(records) else "INCOMPLETE",
        "subset_path": str(subset),
        "subset_sha256": sha256_file(subset),
        "source_provenance": payload.get("provenance"),
        "requested": len(records),
        "successful": len(results),
        "failed": len(failures),
        "aggregate_mae": average_errors(errors),
        "published_chgnet_0_3_0_full_test_mae_for_context_only": {
            "energy_mev_per_atom": 29,
            "force_mev_per_angstrom": 68,
            "stress_gpa": 0.314,
            "magmom_mu_b": 0.037,
        },
        "warning": (
            "This deterministic MPtrj subset is NOT verified as the paper's held-out "
            "test set. Metric values are not a reproduction of published test MAEs."
        ),
        "failures": failures,
    }
    write_json(out_dir / "benchmark_predictions.json", results)
    write_json(out_dir / "benchmark_metrics.json", summary)
    print("AGGREGATE MAE:", json.dumps(summary["aggregate_mae"], indent=2), flush=True)
    print(f"BENCHMARK STATUS: {summary['status']} ({len(results)}/{len(records)})", flush=True)
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "benchmark"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--subset", type=Path, default=Path("inputs/mptrj_subset.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/local"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable (is a GPU allocated and CUDA torch installed?)")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK") or 4))
    info = runtime_info(args)
    t0 = time.perf_counter()
    write_json(args.out_dir / "run_metadata.json", info)
    model = CHGNet.load(model_name=MODEL_NAME, use_device=args.device, verbose=True)
    try:
        success = (run_smoke(model, args.out_dir) if args.mode == "smoke" else
                   run_benchmark(model, args.subset, args.out_dir))
    finally:
        info["wall_seconds_including_model_loading"] = time.perf_counter() - t0
        info["utc_finished"] = datetime.now(timezone.utc).isoformat()
        write_json(args.out_dir / "run_metadata.json", info)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
