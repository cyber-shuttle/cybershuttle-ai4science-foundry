#!/usr/bin/env python3
"""Snapshot package set and machine/Slurm details for a reproducible run."""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(*cmd: str) -> str:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
        return p.stdout + p.stderr
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pip_freeze.txt").write_text(run(sys.executable, "-m", "pip", "freeze"), encoding="utf-8")
    (args.out_dir / "nvidia_smi.txt").write_text(run("nvidia-smi") if shutil.which("nvidia-smi") else
                                                   "nvidia-smi not available\n", encoding="utf-8")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and shutil.which("scontrol"):
        (args.out_dir / "slurm_job_info.txt").write_text(run("scontrol", "show", "job", job_id), encoding="utf-8")
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "loaded_modules": os.getenv("LOADEDMODULES", ""),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV"),
        "slurm_environment": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    (args.out_dir / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Environment snapshot: {args.out_dir}")


if __name__ == "__main__":
    main()
