# CHGNet reproducible inference on AI Makerspace

**Completed H100 run:** job **6199752**, October 11, 2026, 10:55 PM EDT (recorded UTC timestamp is October 12). CHGNet ran on CUDA on an NVIDIA H100 80GB HBM3 in the ICE GPU allocation. All smoke, numerical-reference, and physical-symmetry checks passed. Raw evidence is in `evidence/6199752/`; the original September ICE CPU logs and results are in `evidence/historical/`. Confirm your course's designation of this ICE H100 allocation as AI Makerspace if needed.

The job used 1 GPU, 4 CPU cores, and 16 GB requested RAM. One baseline inference took **1.317 seconds**; loading plus inference and all symmetry checks took **1.811 seconds**, excluding imports and setup. Peak Python RSS was **1,417.6 MiB**; PyTorch peak allocated/reserved GPU memory was **176.0/180.0 MiB**. These measurements exclude the editor and non-PyTorch GPU allocations.

This workflow loads CHGNet checkpoint **0.3.0**, predicts energy, forces, stress, and magnetic moments for the official eight-atom LiMnO2 example, and validates both numerical outputs and physical symmetry. The larger held-out-data evaluation is semester work and is not necessary for this inference assignment.

Model-selection tracking issue: https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/21

## 1. Required resources

Use your course-authorized allocation. Request **one NVIDIA H100 or H200 GPU, 4 CPU cores, 16 GB RAM, and 30 minutes** for inference. H100 and H200 are the two GPU types verified for this workflow. If you select a GPU in Open OnDemand, choose either of those types; for Slurm, use your site's documented GPU selection options if a specific type is required. The default Slurm request is one GPU without a model constraint. Allow **10 GB storage** for environment/source/checkpoints, and roughly **1-5 GB transfers** for software. No full MPtrj dataset download is needed. Environment creation may need a longer initial compute session than inference.

On PACE-ICE, `ice-gpu` and `coc` worked as the partition and account for these course runs. Confirm your own allocation and any required QoS with the course staff before using those values. The documented ICE module is `anaconda3/2023.03` (Python **3.10.13**). If that module is unavailable, use an equivalent Python 3.10 installation for the interactive route; the supplied ICE batch script loads the named module. CPU validation is supported, but only the course can confirm whether CPU meets the Makerspace requirement.

## 2. Setup

In an allocated Linux compute session with Git and outbound package access, clone the submitted branch into your own scratch directory. On PACE-ICE, use `$HOME/scratch`; elsewhere, set `WORK_DIR` to a writable scratch directory provided by your site before running these commands. Keep the checkout and environment at the same path while reproducing the run.

```bash
WORK_DIR="${WORK_DIR:-$HOME/scratch}"
mkdir -p "$WORK_DIR"
git clone --branch assignment2-chgnet-pratham --single-branch \
  https://github.com/prathampatel21/cybershuttle-ai4science-foundry.git \
  "$WORK_DIR/chgnet-foundry"
cd "$WORK_DIR/chgnet-foundry/docs/tutorials/chgnet-pratham"
```

Run **this setup once** in a fresh checkout. Use the archived dependency lock for the standard GPU reproduction. If Python 3.10 is already available, skip only the module command.

```bash
module load anaconda3/2023.03
python3 --version  # must be Python 3.10.x
mkdir -p runs
set -o pipefail
TORCH_BACKEND=cu121 DEPENDENCY_LOCK=evidence/6199752/dependencies-lock.txt \
  bash scripts/setup.sh 2>&1 | tee runs/setup.log
```

Continue only if setup succeeds. The setup pins the upstream CHGNet source, installs PyTorch 2.5.1 from the official CUDA 12.1 wheel index, installs the remaining dependencies from the lock, and records artifact hashes. The lock omits PyTorch and CHGNet because the script installs them separately. CUDA wheels include their runtime; the GPU driver must support it. For a CPU-only diagnostic, use a separate checkout and change `TORCH_BACKEND=cu121` to `TORCH_BACKEND=cpu`.

## 3. Run inference

In an already allocated GPU session:

```bash
DEVICE=cuda bash scripts/run.sh
```

Or, after setup, submit a scheduled job from the tutorial directory. These are examples from PACE-ICE; substitute the values authorized for your account:

```bash
export MAKERSPACE_PARTITION='ice-gpu'
export MAKERSPACE_ACCOUNT='coc'
# Set MAKERSPACE_QOS only if required by your allocation.
bash scripts/submit.sh
```

`inference.sbatch` supplies the resource requests and explicitly requests CUDA. A missing GPU causes a failure rather than silent CPU fallback. A CPU diagnostic run can be started in a CPU compute session with `DEVICE=cpu bash scripts/run.sh`. Seed is 42; thread count is 4; inference uses the unchanged supplied CIF and explicit checkpoint 0.3.0. Outputs go to `runs/JOB_ID/` (or a UTC-stamped manual ID). No training or fine-tuning is performed.

## 4. Validate

The script exits with code 0 only if all checks pass. `results.json` contains expected/observed outputs, tolerances, maximum errors, hashes, versions, UTC time, hardware, runtime, memory, and job context.

- **Smoke test:** finite outputs with energy scalar, forces 8x3, stress 3x3, and eight magnetic moments.
- **Numerical validation:** compare all four outputs with the official `test_predict_structure` reference. Energy is approximately **-7.36769 eV/atom**; the Mn moments are approximately **3.86942 mu_B**. Relative tolerances are energy 1e-4, forces 1e-3, stress 5e-3, moments 1e-3; absolute tolerance is 1e-4 in each output's units. The comparison uses the maximum of absolute/relative tolerances, matching `pytest.approx` rather than summing them.
- **Physical validation:** rotate coordinates and lattice together by 30, 60, and 120 degrees about [1,2,3]. Energy/moments should be invariant; forces and stress should rotate appropriately. Translation and reordered-site checks test invariance and consistent site mapping. Absolute tolerances are e/m/f 1e-4 and stress 1e-3 in their respective units; they are fixed before execution.

These validate implementation consistency for one supplied input; agreement with the model's reference output is not an independent test of DFT accuracy or generalization. All five transformations passed in the H100 run. Failed checks must be investigated and preserved; do not silently widen tolerances.

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v

# Replace JOB_ID with the Run ID printed by run.sh, or the ID returned by sbatch.
RUN_ID='JOB_ID'
python - "runs/$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
status = run / "exit-status.txt"
result = run / "results.json"
if not status.is_file() or not result.is_file():
    raise SystemExit(f"No completed result in {run}; check your Run ID and job logs.")
record = json.loads(result.read_text())
if status.read_text().strip() != "0" or record.get("passed") is not True:
    raise SystemExit(f"Validation failed; inspect {result} and {run / 'execution.log'}.")
print(f"PASS: {result}")
print("Reference:", record["reference_checks"])
print("Symmetry:", record["symmetry_checks"])
PY
```

For a scheduled job, wait until it finishes before checking the result. The files in `evidence/6199752/` document the original H100 run; they do not validate your run. [Independent peer reproduction](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/pull/39#issuecomment-6407596978) confirmed both the direct and Slurm routes on an H200.

CUDA timing synchronizes the device. GPU memory reports PyTorch peaks, not all driver/editor memory. Process RSS excludes other processes. `inference_seconds` excludes model loading; `load_inference_validation_seconds` includes loading and the complete validation suite, but excludes imports and package installation.

## 5. Preserve execution evidence and submit

After the job finishes:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
python scripts/archive_evidence.py RUN_ID
```

Save the `sacct` output and scheduler stdout/stderr beside the archived run. Keep `runs/setup.log` as installation evidence. The archive includes the raw log, results, environment lock, hashes, resource description, and exit status. Keep failed run evidence too. Review logs for unnecessary private paths before publishing; do not commit credentials, environment-variable dumps, GPU weights, `.venv`, or the entire upstream checkout.

## 6. Troubleshooting and cleanup

The September ICE CPU run initially used PyPI CHGNet 0.4.0: energy/moments passed but forces/stress failed. The source revision recorded here (reported package 0.4.1) passed all four without changing tolerances. Preserve this distinction between software package version and checkpoint 0.3.0. The recovered original logs are archived in `evidence/historical/`.

If source checkout fails, verify the recorded commit against the old ICE checkout; do not replace it with floating `main`. If compiled graph conversion fails, record the build/compiler errors. If CUDA is missing, check `nvidia-smi`, the GPU request, and torch's installed wheel; do not reinstall inside a running inference job. For numerical failures compare versions, hashes, site order, and device. Contact the TA for unauthorized partition/account errors.

Cancel only your unused job with `scancel JOB_ID`, or stop its OnDemand session. Preserve the tracked evidence. You can archive local generated environments/source/runs by moving `.venv`, `artifacts`, and `runs` to a backup directory; no cleanup script deletes them automatically. Persistent project files remain after compute sessions expire.

## Sources and licenses

- Paper: Deng et al. (2023), https://doi.org/10.1038/s42256-023-00716-3
- Official code, checkpoint, example, and reference: https://github.com/CederGroupHub/chgnet/tree/b9a6c15860237297a5a5169660e83b2e61a78e76
- Artifact source/identity: `config/workflow.json`; downloaded artifact hashes: `artifacts/manifest.json`, copied into execution evidence.
- Code license: upstream `LICENSE` (BSD-style three-clause with enhancements provision). Retain it with any redistributed upstream artifacts. No separate checkpoint license was identified in its README. The tiny example is distributed with the official repository.
- Reference values in `config/reference.json` are transcribed from the official test cited there, using the September reproduction script. No independently downloaded DFT labels are used here. Validate the pinned test file against this transcription during the fresh run; its checksum is archived.
