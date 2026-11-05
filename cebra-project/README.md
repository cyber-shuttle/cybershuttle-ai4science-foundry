# CEBRA hippocampus inference

This project reproduces the CEBRA hippocampus example for a Georgia Tech AI
Makerspace assignment. The goal is to learn a three-dimensional representation of
real hippocampal neural activity using animal position and running direction as
auxiliary information, then generate embeddings from a saved model.

The workflow was successfully executed on Georgia Tech PACE. The official
Achilles hippocampus data was retrieved, CEBRA was trained, saved, reloaded and
used for inference, and numeric validation completed successfully. A
position-colored embedding plot was generated. Representative execution evidence
is included in [logs/example_successful_run.out](logs/example_successful_run.out).

## Model and scientific reference

CEBRA learns low-dimensional representations using contrastive learning with
behavioral or temporal auxiliary variables. This project uses CEBRA-Behavior.

- Version: **CEBRA 0.6.1**.
- [Official repository](https://github.com/AdaptiveMotorControlLab/CEBRA).
- [Version 0.6.1 source](https://github.com/AdaptiveMotorControlLab/CEBRA/tree/v0.6.1).
- Schneider, S., Lee, J. H., and Mathis, M. W. (2023). *Learnable latent embeddings
  for joint behavioural and neural analysis*. Nature 617, 360–368.
  [DOI: 10.1038/s41586-023-06031-6](https://www.nature.com/articles/s41586-023-06031-6).
- [Official hippocampus example](https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html).

This is a reproduction of the example workflow, rather than a reproduction of
every experiment or performance claim in the publication. The linked demo page
can change; the parameters implemented here are recorded below.

## Data provenance and retrieval

Dataset identifier: **`rat-hippocampus-single-achilles`**.

The original recordings are from CRCNS **hc-11**, contributed by Grosmark, Long
and Buzsáki (2016), *Recordings from hippocampal area CA1, PRE, during and POST
novel spatial learning*, DOI **10.6080/K0862DC5**.
See the [original dataset description](https://crcns.org/data-sets/hc/hc-11/about-hc-11)
and Grosmark and Buzsáki (2016), *Diversity in neural firing dynamics supports
both rigid and learned hippocampal sequences*, Science 351, 1440–1443.

The project uses the processed Achilles artifact distributed by CEBRA. The
[0.6.1 loader](https://github.com/AdaptiveMotorControlLab/CEBRA/blob/v0.6.1/cebra/datasets/hippocampus.py)
provides spike counts in 25 ms bins and aligned behavioral columns
`[position, right, left]`. Position is measured in meters in the demo.

On PACE, `run_inference.py` calls:

```python
cebra.datasets.init("rat-hippocampus-single-achilles", root=str(data_dir), download=True)
```

The loader retrieves the processed file automatically and reuses its cache.
The source is:

```text
https://cebra.fra1.digitaloceanspaces.com/data/rat_hippocampus/achilles.jl.gz
```

The default decompressed location is `data/rat_hippocampus/achilles.jl`.
The storage-aware commands below instead cache it under
`$CEBRA_WORKDIR/data/rat_hippocampus/achilles.jl`.
The workflow verifies the cached file against the loader's checksum and records
its actual SHA-256, source URL and original DOI in run metadata. This identifies
the processed artifact even without a separate dataset version label.
Automatic retrieval and checksum verification succeeded on PACE. No raw CRCNS
credentials were required.

## Repository files

```text
cebra-project/
├── README.md
├── requirements.txt
├── .gitignore
├── test_cebra.py
├── run_inference.py
├── jobs/
│   └── cebra_inference.slurm
├── logs/
│   ├── .gitkeep
│   └── example_successful_run.out
├── results/
│   └── .gitkeep
└── validation/
    └── validate_results.py
```

`test_cebra.py` remains the original synthetic smoke test. `run_inference.py`
performs the real training and saved-model inference stages. The validation
script checks generated arrays and saves a position-colored plot.

## Environment setup on PACE

The following versions were verified during the successful PACE execution.

| Software | Verified version |
| --- | --- |
| Python | 3.10.13 |
| CEBRA | 0.6.1 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.9 |
| PyTorch | 2.14.1 |

Run the setup commands in Bash from the repository root. When home-directory
quota is limited, choose writable scratch or project storage for the environment,
package caches and downloaded data. Replace the generic storage path below;
no particular scratch directory is assumed to exist. Keep the checkout in
scratch or project storage too if generated `results/` and `logs/` should stay
off home storage. Follow that storage location's retention policy when preserving
execution evidence.

```bash
set -e
export PYTHONNOUSERSITE=1
export CEBRA_WORKDIR=/path/to/scratch-or-project-storage/cebra
export CEBRA_ENV="$CEBRA_WORKDIR/cebra-env"
export CONDA_PKGS_DIRS="$CEBRA_WORKDIR/conda-pkgs"
export PIP_CACHE_DIR="$CEBRA_WORKDIR/pip-cache"
mkdir -p "$CEBRA_WORKDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$CEBRA_WORKDIR/data"
```

Choose **one** creation option below, using an unused `CEBRA_ENV` outside the
repository. For an existing environment, skip creation, set `CEBRA_ENV` to its
location and use the matching activation command. Keep the exported variables
in the same shell through installation and submission.

**Option A: Python virtual environment.** Select a PACE module or executable
providing Python 3.10.13. The exact site module command remains **TODO** because
it was not supplied with the execution details.

```bash
set -e
export PYTHONNOUSERSITE=1
test ! -e "$CEBRA_ENV"
python3.10 -m venv "$CEBRA_ENV"
source "$CEBRA_ENV/bin/activate"
```

**Option B: Conda with a private package cache.** This is an alternative when a
shared Conda cache is corrupted or home-directory quota causes problems, rather
than the only valid setup method. Load your site's Conda installation first.
The cache directory must exist before creating the environment. A separate
`CONDA_PKGS_DIRS` avoids reusing the corrupted shared cache; it does not repair it.
See Conda's [custom storage locations](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/custom-env-and-pkg-locations.html)
and [environment management](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

```bash
set -e
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$CEBRA_WORKDIR/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
test ! -e "$CEBRA_ENV"
cebra_conda_hook=$(conda shell.bash hook)
eval "$cebra_conda_hook"
conda create --yes --prefix "$CEBRA_ENV" python=3.10.13 pip
conda activate "$CEBRA_ENV"
```

**Verify and install after either option.** Check the active Python prefix and
pip module before installing anything. A failed creation, activation or check
stops setup. `PYTHONNOUSERSITE=1` disables user-site imports; `--no-user` prevents
pip from falling back to a user installation when the environment is unwritable.
Do not use `pip install --user` as a workaround.

```bash
set -e
export PYTHONNOUSERSITE=1
: "${CEBRA_ENV:?Set CEBRA_ENV before installing.}"
python - <<'PY'
import os
from pathlib import Path
import sys
import pip

expected = Path(os.environ["CEBRA_ENV"]).resolve()
if Path(sys.prefix).resolve() != expected or not Path(pip.__file__).resolve().is_relative_to(expected):
    raise SystemExit("Python and pip must come from CEBRA_ENV.")
if sys.version_info[:3] != (3, 10, 13):
    raise SystemExit("Use Python 3.10.13 to reproduce the recorded environment.")
print("Python executable:", sys.executable)
print("pip module:", pip.__file__)
PY
python -m pip --version
python -m pip install --no-user -r requirements.txt
python -m pip check
```

The successful run used PyTorch 2.14.1 with CUDA available, CUDA build 13.0 and an
NVIDIA A100 80GB PCIe. This confirms compatibility for the executed environment.
The exact package installation source and CUDA wheel installation command are
**TODO**; they were not included in the observed results. When recreating the
environment, follow the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for a compatible GPU build, preserve the verified version, and document the
installation source. Use the
[CEBRA installation guide](https://cebra.ai/docs/installation.html) as a reference.

PACE GPU nodes may have different GPU architectures. The installed PyTorch
build must support the allocated GPU architecture and be compatible with the
node's CUDA/driver environment. On an allocated GPU compute node, after
activation and before running the workflow, verify the device and build:

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

These are device checks, not training or inference. CUDA availability alone does
not establish support for every GPU architecture; investigate incompatible-build
warnings before proceeding. Keep the recorded package versions unless an
explicitly documented environment change is needed. The setup options and
device checks above have not been executed as part of these documentation fixes.

`requirements.txt` contains the project's required software dependencies and
pinned versions used for this workflow. Do not copy private paths, credentials
or unrelated environment contents into requirements.

The script explicitly imports CEBRA's hippocampus module before using the
registered loader. This avoids requiring extra packages for unrelated datasets
when their optional imports would otherwise interrupt registration in
[CEBRA's dataset initialization](https://github.com/AdaptiveMotorControlLab/CEBRA/blob/v0.6.1/cebra/datasets/__init__.py).
Any import warning should remain in the actual run log.

## Configuration and sample policy

The fixed model parameters in `run_inference.py` are:

| Parameter | Value |
| --- | --- |
| `model_architecture` | `offset10-model` |
| `batch_size` | `512` |
| `learning_rate` | `3e-4` |
| `output_dimension` | `3` |
| `distance` | `cosine` |
| `conditional` | `time_delta` |
| `time_offsets` | `10` |
| `temperature` | `1` |
| `max_iterations` | `5000` |

They follow the behavior example. The device is explicitly `cuda` for the
allocated GPU workflow; `verbose=True` records training progress and
`pad_before_transform=True` preserves sample count. These execution settings
are also recorded. The program requires a SLURM job ID and an available CUDA
device, and refuses to reuse a nonempty output directory.

The default seed is `42`. Python, NumPy and PyTorch RNGs are seeded, and
deterministic operations are requested with warnings rather than silently
suppressing unsupported cases. CEBRA 0.6.1 marks its estimator nondeterministic;
identical retraining results across runs or hardware are not guaranteed.
See the [versioned estimator source](https://github.com/AdaptiveMotorControlLab/CEBRA/blob/v0.6.1/cebra/integrations/sklearn/cebra.py).

Every neural sample is used for training and inference in original temporal
order. This follows the demo's full-data stage. Behavior is used during `fit`;
`transform` receives only neural activity. Saved labels are for plotting and
validation. No preprocessing, resampling, feature reordering, random split,
held-out decoder or pretrained weights are introduced.

This validation is descriptive and uses training samples. It cannot establish
out-of-sample decoding performance. A held-out evaluation would be a separate
extension requiring a documented split and leakage controls.

## Smoke test on an allocated PACE node

Run from the repository root after environment activation and allocation of a
compute node. Do not train on the Mac or a PACE login node.

The unchanged script runs with `python -u test_cebra.py`. For grading evidence,
the following command also checks its arrays, records versions and runtime, and
saves the actual output without editing the script:

```bash
mkdir -p logs
set -o pipefail
python -u - <<'PY' 2>&1 | tee "logs/smoke-${SLURM_JOB_ID:?Run on an allocated node}.log"
import json
import platform
import random
import runpy
import socket
import time
import numpy as np
import torch
import cebra

print("Host:", socket.gethostname())
print("Python:", platform.python_version())
print("CEBRA:", cebra.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
started = time.perf_counter()
status = "failed"
try:
    test = runpy.run_path("test_cebra.py")
    assert test["X"].shape == (1000, 20)
    assert test["embedding"].shape == (1000, 3)
    assert np.isfinite(test["embedding"]).all()
    print("Smoke shape and finite-value checks passed")
    status = "success"
finally:
    print(json.dumps({"status": status, "seed": 42,
                      "elapsed_seconds": time.perf_counter() - started}))
PY
```

The synthetic smoke test completed successfully with input shape `(1000, 20)`
and embedding shape `(1000, 3)`. These dimensions describe synthetic data, not
Achilles. No smoke-test runtime is reported here because none was provided.
Keep the actual smoke log as execution evidence.

## Full workflow on PACE

On an allocated GPU node, after activation, run from the repository root:

```bash
mkdir -p logs
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${CEBRA_WORKDIR:?Set CEBRA_WORKDIR to writable scratch or project storage.}"
set -o pipefail
python -u run_inference.py --seed 42 --data-dir "$CEBRA_WORKDIR/data" \
  --output-dir "results/job-${SLURM_JOB_ID:?Run on an allocated node}" \
  2>&1 | tee "logs/workflow-${SLURM_JOB_ID}.log"
python -u validation/validate_results.py \
  --results-dir "results/job-${SLURM_JOB_ID}" \
  2>&1 | tee "logs/validation-${SLURM_JOB_ID}.log"
```

Run validation only if the workflow command succeeds. Use a new output directory
for each attempt. `--seed`, `--data-dir` and `--output-dir` are configurable;
model hyperparameters remain fixed in the source.

The stages within the workflow are:

1. Record runtime environment information, retrieve the processed dataset, and
   verify its arrays and checksum.
2. **Training:** `model.fit(neural, behavior)` on all samples.
3. Save `model.pt`, release the model object, and reload that checkpoint using
   `cebra.CEBRA.load(...)`.
4. **Inference:** `model.transform(neural)` on the same ordered samples.
5. Save embeddings, aligned labels, sample indices and execution metadata.

The checkpoint is trained during this run. Its checksum, dataset checksum,
source-script checksum, configuration and seed establish its provenance.
Reloading demonstrates that inference uses the saved checkpoint. A new full
workflow command retrains; this script does not provide an inference-only mode.

## SLURM submission

The successful job executed in a GPU partition using **one GPU, 4 CPU cores,
16 GB requested memory and a 30-minute requested walltime**. The GPU directive
`#SBATCH --gres=gpu:1` was successfully used. The job completed with exit status
`0`, SLURM walltime `1:05`, and approximately `2.57 GB` observed memory usage.
These are tested requests and observations, not minimum resource requirements.

The script now includes the tested partition `coc-gpu` (`#SBATCH -p coc-gpu`)
and QoS `coc-ice` (`#SBATCH -q coc-ice`). These are the supplied course resource
settings, not personal account identifiers. Use an allocation that permits these
settings; students using another allocation should select its approved settings.
Any required module loading must be completed before setup and submission.

Export `CEBRA_ENV` and `CEBRA_WORKDIR` as shown in setup, activate the environment,
and run the device checks inside an allocation before execution. The script
supports virtual environments and Conda prefixes. For Conda, submit from the
shell where `conda activate "$CEBRA_ENV"` has initialized and exported
`CONDA_EXE`; the job uses that executable to initialize its Bash activation.
It verifies Python/pip ownership and disables user-site imports before checking
dependencies. Submit from the repository root in the same configured shell:

```bash
: "${CEBRA_ENV:?Set and activate CEBRA_ENV before submitting.}"
: "${CEBRA_WORKDIR:?Set CEBRA_WORKDIR to writable scratch or project storage.}"
mkdir -p logs results
sbatch jobs/cebra_inference.slurm
```

Data is cached under `$CEBRA_WORKDIR/data`; if `CEBRA_WORKDIR` is unset, the job
script retains its original `data/` default. Log destinations and result paths
remain relative to the repository. Setup/activation changes have only been
checked statically; the verified timings and results below are from the earlier
successful PACE execution.

SLURM must be able to find `logs/` before the script starts, because it opens its
output files first. The script uses `SLURM_SUBMIT_DIR` as the project root.
Use the real ID printed by `sbatch` for monitoring:

```bash
squeue -j <actual-job-id>
```

Logs will be `logs/cebra-<actual-job-id>.out` and `.err`; artifacts will be in
`results/job-<actual-job-id>/`. Angle-bracket values in this README are placeholders
to replace, not fabricated IDs or literal shell arguments.

## Outputs and validation

After a successful job, the run directory should contain:

| File | Contents |
| --- | --- |
| `model.pt` | Trained CEBRA checkpoint, saved before inference. |
| `embeddings.npz` | `embeddings`, `behavior`, and `sample_indices` arrays. |
| `run_metadata.json` | Actual versions, hostname, CUDA/GPU details, seed, parameters, shapes, checksums, paths, stage timings and status. |
| `embedding_by_position.png` | Three-dimensional embedding colored by animal position. |
| `validation_summary.json` | Observed numeric checks and a pending visual-review reminder. |

The successful run had neural shape `(10178, 120)`, behavior shape `(10178, 3)`
and embedding shape `(10178, 3)`, covering `10178` samples. All neural, behavior
and embedding finite-value checks passed. Checkpoint saving and reloading
succeeded, numeric validation passed, and the position-colored plot was generated.
Training took `42.7069 s`, inference took `0.2064 s`, and the end-to-end workflow
took `54.4930 s` on an NVIDIA A100 80GB PCIe.

On failure, the workflow records the stage and error when possible, prints the
traceback, and exits unsuccessfully.

Validation can be repeated on existing outputs without loading data or training:

```bash
python -u validation/validate_results.py --results-dir results/job-<actual-job-id>
```

Numeric acceptance criteria are a nonempty `(N, 3)` embedding, aligned `(N, 3)`
behavior, no NaN/Inf values, the recorded neural sample count, and preserved
sample order. Visualization should show whether position varies coherently
through the embedding, which is reasonable to investigate because position
and direction inform contrastive sampling during training.

An exact orientation or exact match to the reference figure is not required.
Shape and finite-value checks establish technical validity, not scientific
accuracy. The plot is qualitative evidence; it does not by itself prove
topology, generalization or decoding accuracy. **TODO:** review the generated
figure and record its actual visual structure. Plot generation is confirmed,
but no visual interpretation was supplied. No claim of held-out generalization,
decoding accuracy or exact figure reproduction is made.

## Verified PACE execution

| Item | Observed result |
| --- | --- |
| Python | 3.10.13 |
| CEBRA | 0.6.1 |
| PyTorch | 2.14.1 |
| CUDA available | True |
| PyTorch CUDA build | 13.0 |
| GPU | NVIDIA A100 80GB PCIe |
| Neural input shape | (10178, 120) |
| Behavioral shape | (10178, 3) |
| Embedding shape | (10178, 3) |
| Neural/behavior finite | True |
| Embedding finite | True |
| Training iterations | 5000 |
| Training time | 42.7069 s |
| Inference time | 0.2064 s |
| End-to-end workflow time | 54.4930 s |
| SLURM walltime | 1:05 |
| CPUs requested | 4 |
| Memory requested | 16 GB |
| Observed memory usage | ~2.57 GB |
| GPU request | 1 GPU |
| Checkpoint save/reload | Successful |
| Numeric validation | Passed |
| Sample count | 10178 |
| Job exit status | 0 |

The workflow retrieved the official Achilles hippocampus dataset, verified its
arrays and checksum, trained CEBRA, saved and reloaded the checkpoint, generated
embeddings with `model.transform(...)`, completed numeric validation successfully,
and generated a position-colored embedding visualization.

Validation is descriptive on the training samples and does not establish
held-out generalization or decoding performance. The observed dimensions and
finite values meet the technical acceptance criteria. Visual interpretation
remains to be documented after reviewing the plot.

The [representative execution log](logs/example_successful_run.out) records the
reported versions, GPU, shapes, full-precision workflow timings and success
status with identifying infrastructure details removed. It does not include
the scheduler walltime or memory usage, which are summarized above from the
provided execution results. The plot and JSON summaries are generated outputs
on PACE and are not currently included in the local repository tree.

## Troubleshooting

- **No matching dependency distribution:** confirm the supplied pin, Python
  version and approved package source. In particular, do not replace
  the verified `torch==2.14.1` pin; reproduce the recorded CUDA build `13.0`
  using the appropriate installation source.
- **CUDA unavailable:** check the compute allocation and whether PyTorch is a
  compatible GPU build. The real workflow fails instead of falling back to CPU.
- **SLURM job ID missing:** run on an allocated compute node rather than the Mac
  or a login node. Job-ID detection is a guard, not verification of course access.
- **Optional dataset warning:** the workflow explicitly registers hippocampus.
  If its import still fails, retain the traceback and identify the missing
  required dependency before changing the environment.
- **Download or checksum failure:** check compute-node network access and the
  official artifact URL. If downloads are restricted, retrieve the same artifact
  through an approved PACE transfer process into the expected cache. Verify its
  checksum; do not substitute data or suppress errors.
- **Checkpoint reload failure:** retain the traceback and exact runtime versions.
  CEBRA's save/load API is experimental across versions. Use the same documented
  environment for training and inference.
- **Memory or walltime exceeded:** retain scheduler evidence and adjust the
  resource request based on measurements. Do not silently reduce training
  iterations or change model parameters to make the run finish.
- **Existing output directory:** use a fresh `--output-dir`; preserve failed-run
  evidence rather than overwriting it.
- **Missing or invalid validation files:** inspect the workflow status and sample
  alignment first. Empty output directories are not successful results.

## Evidence, version control and cleanup

Keep source files, requirements and documentation in version control.
`requirements.txt` contains software dependencies only. `.gitignore` excludes
environments, downloaded data, large checkpoints and raw embedding archives;
keep those artifacts excluded unless they are explicitly needed.

Small selected logs, plots and JSON summaries may be committed as execution
evidence. Raw cluster-specific logs may remain private when a representative
example log is committed, as with `logs/example_successful_run.out`. Remove
identifying infrastructure details from shared evidence while preserving
technical results. Review evidence before sharing it.

After retaining the evidence needed for grading, optional cleanup from the
repository root can remove only generated data or selected large artifacts:

```bash
rm -ri "${CEBRA_WORKDIR:?Set CEBRA_WORKDIR before cleanup.}/data"
rm -i results/job-<actual-job-id>/model.pt results/job-<actual-job-id>/embeddings.npz
```

For runs that used the original repository-local data default, use `rm -ri data/`
instead. These cleanup commands prompt before removal. Keep
metadata, validation summaries, plots and selected actual logs. Retain a copy of
the checkpoint elsewhere if checkpoint-only inference is needed;
retraining from configuration is not guaranteed to recreate identical weights.
No virtual environment should be committed or deleted as part of this cleanup.
