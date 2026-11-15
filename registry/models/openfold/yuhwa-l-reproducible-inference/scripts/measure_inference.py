#!/usr/bin/env python3
"""Measure OpenFold inference time, GPU use, and accuracy for one protein.

Runs OpenFold's official run_pretrained_openfold.py inside this process with
the same options as scripts/run_inference.sh (model_1_ptm, seed 42, no
relaxation, default attention and chunking), and wraps its run_model() so
every forward pass is timed and its GPU memory recorded. The model is loaded
once, run --warmup times (unmeasured: chunk-size tuning and kernel setup
happen here), then --repeats more times for the measurements.

Results go to OUTPUT_DIR/<target>/:
  predictions/     OpenFold's predicted structure (PDB, pLDDT in B-factors)
  <target>_superposed_on_reference.pdb   prediction rotated onto the reference
  confidence.npz   per-residue pLDDT and the PAE matrix
  result.json      every measurement, the exact OpenFold command, and versions
and one summary row is appended to OUTPUT_DIR/measurements.csv.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import re
import runpy
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from structure_metrics import compare, write_superposed

RECIPE_DIR = Path(__file__).resolve().parent.parent
PACE_AF_DATA = Path("/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data")
CONFIG_PRESET = "model_1_ptm"
GIB = 2**30

CSV_COLUMNS = [
    "target", "length", "status", "gpu", "max_template_date", "warmup", "repeats",
    "time_s_median", "time_s_min", "time_s_max", "time_s_first",
    "peak_alloc_gib", "peak_reserved_gib", "peak_gpu_used_gib", "gpu_util_mean_pct",
    "mean_plddt", "ptm", "mean_pae", "tm_score", "lddt_ca", "rmsd_ca", "n_matched",
    "job_id", "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fasta", type=Path, required=True,
                        help="FASTA file with one sequence; its header must match a "
                             "folder in --alignment-dir.")
    parser.add_argument("--alignment-dir", type=Path, required=True,
                        help="Folder of precomputed alignments, one subfolder per target.")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Experimental structure (PDB or mmCIF) for the accuracy metrics.")
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--max-template-date", required=True,
                        help="Only use templates released on or before this date (YYYY-MM-DD). "
                             "Use a date before the reference was released.")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured passes (default: 1).")
    parser.add_argument("--repeats", type=int, default=3, help="Measured passes (default: 3).")
    parser.add_argument("--output-dir", type=Path, default=RECIPE_DIR / "results/my-measurements")
    parser.add_argument("--openfold-dir", type=Path,
                        default=Path(os.environ.get("OPENFOLD_DIR", Path.home() / "scratch/openfold")))
    parser.add_argument("--params", type=Path,
                        default=Path(os.environ.get("PARAMS", PACE_AF_DATA / "params/params_model_1_ptm.npz")))
    parser.add_argument("--template-dir", type=Path,
                        default=Path(os.environ.get("TEMPLATE_DIR", PACE_AF_DATA / "pdb_mmcif/mmcif_files")))
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the inputs and print the OpenFold command without running it.")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be at least 1 and --warmup at least 0.")
    for attr in ("fasta", "alignment_dir", "reference", "output_dir", "openfold_dir",
                 "params", "template_dir"):
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    return args


def read_fasta(path):
    """Return (tag, sequence). The tag is parsed the same way OpenFold does."""
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            records.append([re.split(r"\W| \|", line[1:])[0], ""])
        elif line and records:
            records[-1][1] += line
    if len(records) != 1:
        sys.exit(f"{path} must contain exactly one sequence (found {len(records)}).")
    return records[0]


class GpuMonitor:
    """Samples GPU memory use and utilization every 0.1 s in a background thread (NVML)."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []  # (time, memory used in bytes, utilization %)
        self.note = None
        self._handle = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            import torch
            pynvml.nvmlInit()
            self._nvml = pynvml
            # The GPU that PyTorch uses as cuda:0.
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(
                f"GPU-{torch.cuda.get_device_properties(0).uuid}")
        except Exception as e:  # monitoring is optional
            self.note = f"GPU monitor disabled: {e}"

    def start(self):
        if self._handle is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            self.samples.append((time.perf_counter(), used, util))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def window(self, start, end):
        """Peak memory used (bytes) and mean utilization (%) between two times."""
        inside = [(used, util) for t, used, util in self.samples if start <= t <= end]
        if not inside:
            return None, None
        return max(u for u, _ in inside), sum(u for _, u in inside) / len(inside)


def run_openfold(openfold_dir, argv, warmup, repeats, record):
    """Run run_pretrained_openfold.py in this process, timing each forward pass.

    Fills `record` as it goes, so partial results survive an out-of-memory error.
    """
    import torch
    import openfold.utils.script_utils as script_utils

    original_run_model = script_utils.run_model

    def timed_run_model(model, batch, tag, output_dir):
        for i in range(warmup + repeats):
            out = None  # free the previous pass's outputs first
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            out = original_run_model(model, dict(batch), tag, output_dir)
            torch.cuda.synchronize()
            end = time.perf_counter()
            record["passes"].append({
                "warmup": i < warmup,
                "start": start,
                "end": end,
                "seconds": end - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "mean_plddt": out["plddt"].float().mean().item(),
            })
        record["plddt"] = out["plddt"].float().cpu().numpy()
        record["ptm"] = out["ptm_score"].item()
        record["pae"] = out["predicted_aligned_error"].float().cpu().numpy()
        return out

    script_utils.run_model = timed_run_model
    saved_argv, saved_cwd = sys.argv, os.getcwd()
    sys.argv = argv
    # As with `python run_pretrained_openfold.py`, its folder must be importable
    # (it imports OpenFold's own scripts/ package).
    sys.path.insert(0, str(openfold_dir))
    os.chdir(openfold_dir)  # run_inference.sh runs OpenFold from its checkout too
    try:
        runpy.run_path(argv[0], run_name="__main__")
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        script_utils.run_model = original_run_model


def environment_info(openfold_dir):
    import torch
    info = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
    }
    try:
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        info["nvidia_driver"] = None
    try:
        info["openfold_commit"] = subprocess.run(
            ["git", "-C", str(openfold_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["openfold_commit"] = None
    for package in ("biopython", "numpy"):
        info[package] = importlib.metadata.version(package)
    return info


def fmt(value, digits=4):
    return "" if value is None else round(value, digits)


def append_csv(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    tag, sequence = read_fasta(args.fasta)
    run_dir = args.output_dir / tag

    required = {
        "OpenFold checkout": args.openfold_dir / "run_pretrained_openfold.py",
        "Model parameters": args.params,
        "Template mmCIF folder": args.template_dir,
        f"Alignments for {tag} (see README step 6.1)": args.alignment_dir / tag,
        "Reference structure": args.reference,
    }
    for what, path in required.items():
        if not path.exists():
            sys.exit(f"{what} not found: {path}")
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; remove it or choose another --output-dir.")

    argv = [
        str(args.openfold_dir / "run_pretrained_openfold.py"),
        str(run_dir / "input"),
        str(args.template_dir),
        "--use_precomputed_alignments", str(args.alignment_dir),
        "--output_dir", str(run_dir),
        "--config_preset", CONFIG_PRESET,
        "--jax_param_path", str(args.params),
        "--model_device", "cuda:0",
        "--data_random_seed", "42",
        "--max_template_date", args.max_template_date,
        "--skip_relaxation",
    ]

    if args.dry_run:
        print(f"Target:    {tag} ({len(sequence)} residues)")
        print(f"Reference: {args.reference} chain {args.reference_chain}")
        print(f"Output:    {run_dir}")
        print("OpenFold command:\n  python " + " ".join(argv))
        return 0

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible; run inside a GPU allocation (see slurm/measure_inference.slurm).")

    (run_dir / "input").mkdir(parents=True)
    shutil.copy(args.fasta, run_dir / "input" / f"{tag}.fasta")
    print(f"=== {tag} ({len(sequence)} residues) -> {run_dir} ===", flush=True)

    env = environment_info(args.openfold_dir)
    monitor = GpuMonitor()
    record = {"passes": []}
    status, error = "ok", None
    started = time.perf_counter()
    monitor.start()
    try:
        run_openfold(args.openfold_dir, argv, args.warmup, args.repeats, record)
    except Exception as e:
        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        status = "oom" if oom else "error"
        error = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        traceback.print_exc()
    finally:
        monitor.stop()
    wall_seconds = time.perf_counter() - started

    passes = record["passes"]
    for p in passes:
        p["gpu_used_peak_bytes"], p["gpu_util_mean_pct"] = monitor.window(p["start"], p["end"])
    if status == "ok" and len(passes) != args.warmup + args.repeats:
        status, error = "error", "OpenFold did not run the expected forward passes"
    measured = [p for p in passes if not p["warmup"]]
    times = [p["seconds"] for p in measured]

    def peak(key):
        values = [p[key] for p in measured if p.get(key) is not None]
        return max(values) / GIB if values else None

    utils = [p["gpu_util_mean_pct"] for p in measured if p["gpu_util_mean_pct"] is not None]

    accuracy = {}
    prediction = run_dir / "predictions" / f"{tag}_{CONFIG_PRESET}_unrelaxed.pdb"
    if status == "ok":
        accuracy = compare(prediction, args.reference, ref_chain=args.reference_chain)
        write_superposed(prediction, args.reference,
                         run_dir / f"{tag}_superposed_on_reference.pdb",
                         ref_chain=args.reference_chain)
        if accuracy["sequence_identity"] < 0.9:
            print(f"WARNING: prediction and reference are only "
                  f"{accuracy['sequence_identity']:.0%} identical in sequence; "
                  f"check --reference and --reference-chain.")

    mean_pae = None
    if "plddt" in record:
        mean_pae = float(record["pae"].mean())
        np.savez_compressed(run_dir / "confidence.npz", plddt=record["plddt"],
                            pae=record["pae"].astype(np.float16))

    row = {
        "target": tag,
        "length": len(sequence),
        "status": status,
        "gpu": env["gpu"],
        "max_template_date": args.max_template_date,
        "warmup": args.warmup,
        "repeats": len(times),
        "time_s_median": fmt(statistics.median(times) if times else None, 2),
        "time_s_min": fmt(min(times) if times else None, 2),
        "time_s_max": fmt(max(times) if times else None, 2),
        "time_s_first": fmt(passes[0]["seconds"] if passes else None, 2),
        "peak_alloc_gib": fmt(peak("peak_allocated_bytes"), 2),
        "peak_reserved_gib": fmt(peak("peak_reserved_bytes"), 2),
        "peak_gpu_used_gib": fmt(peak("gpu_used_peak_bytes"), 2),
        "gpu_util_mean_pct": fmt(sum(utils) / len(utils) if utils else None, 1),
        "mean_plddt": fmt(float(record["plddt"].mean()) if "plddt" in record else None, 2),
        "ptm": fmt(record.get("ptm"), 3),
        "mean_pae": fmt(mean_pae, 2),
        "tm_score": fmt(accuracy.get("tm_score"), 3),
        "lddt_ca": fmt(accuracy.get("lddt_ca"), 3),
        "rmsd_ca": fmt(accuracy.get("rmsd_ca"), 2),
        "n_matched": accuracy.get("n_matched", ""),
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "error": error or "",
    }

    result = {
        "summary": row,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "openfold_command": ["python", *argv],
        "fasta": str(args.fasta),
        "sequence": sequence,
        "reference": str(args.reference),
        "reference_chain": args.reference_chain,
        "accuracy": accuracy,
        "passes": passes,
        "wall_seconds": wall_seconds,
        "gpu_samples": len(monitor.samples),
        "gpu_monitor_note": monitor.note,
        "environment": env,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    append_csv(args.output_dir / "measurements.csv", row)

    print(f"\n=== {tag}: {status} ===")
    for key in ("time_s_median", "time_s_first", "peak_alloc_gib", "peak_gpu_used_gib",
                "gpu_util_mean_pct", "mean_plddt", "ptm", "tm_score", "lddt_ca", "rmsd_ca", "error"):
        if row[key] != "":
            print(f"{key:>18}: {row[key]}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
