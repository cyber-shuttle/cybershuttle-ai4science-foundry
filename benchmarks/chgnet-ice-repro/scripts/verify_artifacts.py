#!/usr/bin/env python3
"""Check that a saved run has the files and PASS/COMPLETED status I expect."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_manifest(run_dir: Path) -> None:
    manifest = run_dir / "artifact_checksums.sha256"
    if not manifest.is_file():
        raise AssertionError("missing artifact_checksums.sha256")
    for lineno, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise AssertionError(f"invalid checksum line {lineno}: {raw_line!r}")
        expected, rel_path = parts
        rel_path = rel_path.lstrip("*")
        target = run_dir / rel_path
        if not target.is_file():
            raise AssertionError(f"checksum target missing: {rel_path}")
        observed = sha256_file(target)
        if observed != expected:
            raise AssertionError(f"checksum mismatch for {rel_path}: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark"], required=True)
    parser.add_argument("--expected-count", type=int, default=24,
                        help="Expected number of benchmark structures")
    args = parser.parse_args()
    files = [
        "run_metadata.json",
        "pip_freeze.txt",
        "nvidia_smi.txt",
        "environment.json",
        "slurm_job_info.txt",
        "slurm_accounting.txt",
        "artifact_checksums.sha256",
    ]
    files += (["smoke_predictions.json", "smoke_validation.json"] if args.mode == "smoke" else
              ["benchmark_predictions.json", "benchmark_metrics.json"])
    missing = [f for f in files if not (args.run / f).is_file()]
    if missing:
        print(f"FAIL: missing {missing}")
        return 1
    try:
        meta = json.loads((args.run / "run_metadata.json").read_text())
        assert meta["checkpoint_sha256"] and meta["model_checkpoint_id"] == "0.3.0"
        verify_checksum_manifest(args.run)
        if args.mode == "smoke":
            data = json.loads((args.run / "smoke_validation.json").read_text())
            assert data["status"] == "PASS" and all(c["pass"] for c in data["checks"].values())
        else:
            data = json.loads((args.run / "benchmark_metrics.json").read_text())
            assert data["status"] == "COMPLETED"
            assert data["requested"] == args.expected_count and data["successful"] == data["requested"]
            assert data["failed"] == 0 and not data.get("failures")
            assert data["subset_sha256"]
            for key in ("energy_mev_per_atom", "force_mev_per_angstrom", "stress_gpa", "magmom_mu_b"):
                metric = data["aggregate_mae"][key]
                assert metric["count"] > 0 and metric["mae"] is not None
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"FAIL: invalid metadata or validation: {e}")
        return 1
    print(f"PASS: {args.mode} artifacts verified under {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
