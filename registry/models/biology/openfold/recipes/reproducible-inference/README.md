# OpenFold Reproducible Inference: 6KWC

This recipe reproduces OpenFold structure inference for the 191-residue 6KWC xylanase sequence on **Georgia Tech PACE/ICE**, using the official OpenFold code, the `model_1_ptm` parameters on the PACE shared filesystem, and the precomputed alignments included here.

| Item | Value |
|---|---|
| Target | 6KWC (`6KWC_1`, 191 residues) |
| OpenFold commit | `be2ec1841f16c966c65ae0e7599ebbadc725757d` |
| Model / preset | `model_1_ptm` |
| Data random seed | 42 |
| Device | `cuda:0`, NVIDIA A100-PCIE-40GB (`coc-gpu`) |
| PyTorch | 2.5.1, CUDA 12.4 |
| Alignments | Precomputed (included) |
| Relaxation | Skipped |

Result: the prediction matches all 191 input residues, has a mean pLDDT of **97.72**, and a C-alpha RMSD of **0.961 Angstrom** over all 191 residues after optimal superposition on experimental 6KWC chain A.

All commands below assume a PACE/ICE shell. First get this repository into scratch (login node); `$RECIPE` is this recipe's directory:

```bash
cd ~/scratch
git clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry
export RECIPE="$PWD/registry/models/biology/openfold/recipes/reproducible-inference"
```

Until pull request #41 is merged, the recipe is only on the contributor's branch. Clone it with `git clone -b openfold-reproducible-inference https://github.com/Yuhwa-L/cybershuttle-ai4science-foundry.git` instead, then run the `cd` and `export` lines above.

In a new shell (for example, inside a later allocation), set `RECIPE` again before running the commands below.

## 1. Setup

Clone on the login node (1.1), then build the environment on a compute node (1.2–1.3). This is done once: the environment stays in scratch, and later allocations only activate it (section 3).

### 1.1 Clone OpenFold at the pinned commit (login node)

```bash
cd ~/scratch
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
git checkout be2ec1841f16c966c65ae0e7599ebbadc725757d
```

`~/scratch/openfold` is the default `OPENFOLD_DIR` used by `scripts/run_inference.sh`.

### 1.2 Create the Conda environment under scratch (compute node)

Get an interactive allocation first. Login nodes limit memory per user, and `mamba env create` there was killed with exit code 137 (out of memory) while solving the environment; the allocation's 32 GB is enough. Stay in this allocation through section 1.3, and set `RECIPE` again inside it.

```bash
salloc --account=coc --partition=coc-gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00
```

The environment is several GB. The PACE home quota is small, and creating it in `$HOME` failed for us with `OSError: [Errno 122] Disk quota exceeded`, so keep the Conda package cache, environments, and pip cache in scratch:

```bash
module load miniforge
export CONDA_PKGS_DIRS="$HOME/scratch/.conda/pkgs"
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

mamba env create -y -f "$RECIPE/environment.yml"
```

`flash-attn` is deliberately not in `environment.yml`: its build needs the already-installed PyTorch, and with it in the pip list `mamba env create` failed in our clean test. It is installed in the next step.

This creates `openfold-env` in `~/scratch/.conda/envs`. `CONDA_ENVS_PATH` must be set in every new shell before `conda activate openfold-env` (the SLURM script sets it for you).

### 1.3 Install OpenFold into the environment

```bash
conda activate openfold-env
pip install flash-attn==2.8.3 --no-build-isolation
unset GCC_ROOT
cd ~/scratch/openfold
scripts/install_third_party_dependencies.sh
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

`unset GCC_ROOT` is required on PACE: the default `gcc` module sets it, and with it set the Conda compiler fails with `cannot execute 'cc1plus'` while building OpenFold's CUDA attention kernel. The install script also prints a harmless `fatal: destination path 'cutlass' already exists` if it is run twice.

Check the installation:

```bash
python -c "import torch, flash_attn, openfold, cuda, tensorrt, polygraphy; print(torch.__version__, torch.version.cuda)"
python ~/scratch/openfold/run_pretrained_openfold.py --help
```

`tensorrt`, `polygraphy`, and `cuda-python` are needed only because the OpenFold entry point imports them at this commit. TensorRT acceleration was not enabled.

Exact package versions from the original run are in `requirements-lock.txt` and `logs/environment.log`.

## 2. Data used (nothing to download)

PACE/ICE provides the AlphaFold 2.3.2 data on its shared filesystem, and `scripts/run_inference.sh` uses it by default:

| Data | Path |
|---|---|
| Parameters | `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/params/params_model_1_ptm.npz` |
| Templates | `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/pdb_mmcif/mmcif_files` |

Parameter SHA256: `5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84`

The input FASTA is `inputs/6KWC.fasta`. The precomputed alignments are in `alignments/6KWC_1/` (`bfd_uniref_hits.a3m`, `hhsearch_output.hhr`, `mgnify_hits.sto`, `uniref90_hits.sto`). They were taken from the 6KWC monomer example in the VizFold Foundation repository at commit `745229dcb0b7eb5ed06ebf6678e99eded418b543`.

Because the alignments are included, **the MSA/database search does not need to be rerun**. This recipe reproduces only the OpenFold inference stage.

## 3. Run inference (GPU node)

Inference needs a GPU, so do not run it on a login node.

### Option A: batch job

```bash
cd "$RECIPE"
sbatch slurm/openfold_inference.slurm
```

Submit from the recipe directory. The script requests one A100 on `coc-gpu` (account `coc`, 8 CPUs, 32 GB, 2 hours), activates `openfold-env` from `~/scratch/.conda/envs`, prints `nvidia-smi`, and calls `scripts/run_inference.sh`. Edit the `#SBATCH` lines if your account or partition differs.

### Option B: interactive allocation

```bash
salloc --account=coc --partition=coc-gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00

module load miniforge
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
conda activate openfold-env
cd "$RECIPE"
./scripts/run_inference.sh
```

`run_inference.sh` checks that the OpenFold checkout, input, alignments, parameters, templates, and a CUDA GPU are available before starting. It runs `model_1_ptm` on `cuda:0` with `--data_random_seed 42`, precomputed alignments, and `--skip_relaxation`.

Output goes to `results/run/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb`. Set `OUTPUT_DIR` to write somewhere else; `OPENFOLD_DIR`, `PARAMS`, `TEMPLATE_DIR`, and `ALIGNMENTS_DIR` can be overridden the same way.

## 4. Validate

```bash
cd "$RECIPE"
python scripts/validate_structure.py --prediction results/run/predictions/6KWC_1_model_1_ptm_unrelaxed.pdb
```

Without `--prediction`, the script validates the included reference prediction. It checks the predicted sequence against the FASTA, reports pLDDT (stored in the B-factor column), and superimposes the prediction on experimental 6KWC chain A (`reference/6KWC.pdb`). Expected output:

```text
Prediction residues: 191
Prediction sequence exact match: True
Mean pLDDT: 97.72
Median pLDDT: 98.57
Min pLDDT: 58.91
Max pLDDT: 98.92
Experimental sequence exact match: True
Aligned C-alpha residues: 191
C-alpha RMSD after optimal superposition: 0.961 Angstrom
```

The reference prediction is `results/6KWC_1_model_1_ptm_unrelaxed.pdb` (SHA256 `eca9dedb426047a781658bf8679e5f99d26fcb007e8f4c5a7929bfd15cd48b69`).

## 5. Visualizations (optional)

The images in `visualizations/` were made with PyMOL, which is not part of `environment.yml`:

```bash
cd "$RECIPE"
pymol -cq scripts/visualize_structure.pml
```

- `6KWC_1_openfold_plddt.png`: prediction colored by pLDDT.
- `6KWC_1_openfold_vs_experimental.png`: prediction (blue) on experimental 6KWC chain A (gray).

## 6. Proteins of different lengths

`length-series/` adds five proteins from 159 to 1,125 residues, to measure how inference time, GPU memory, and accuracy change with protein size. They are listed in `length-series/targets.tsv`:

| PDB chain | Residues | Released | Resolution | Protein |
|---|---|---|---|---|
| 9OUF_A | 159 | 2025-07-30 | 1.39 Å | ADP-ribosyl hydrolase |
| 9Q5U_A | 325 | 2025-09-24 | 2.50 Å | Carbohydrate-binding module of an RTX protein |
| 9NXH_A | 500 | 2025-08-20 | 1.75 Å | Alpha-L-arabinofuranosidase |
| 9LB7_A | 777 | 2025-07-30 | 2.35 Å | Trehalose-6-phosphate phosphorylase |
| 8P5S_A | 1125 | 2023-08-16 | 2.46 Å | 2-oxoglutarate dehydrogenase |

They were chosen from the PDB copy on the PACE shared filesystem (snapshot of 2025-09-30), one near each of 150, 300, 500, 800, and 1,100 residues, with these criteria:

- one protein chain per entry, no unknown residues;
- X-ray structure at 2.5 Å or better, with at least 90% of residues resolved;
- released in 2023 or later, and no 20-residue stretch shared with any PDB entry from before 2018, so the protein is new to OpenFold, whose parameters were trained on structures released before May 2018.

The inputs, experimental structures, and alignments for these proteins are already in the repository, so only step 6.1 is needed. Steps 6.2 and 6.3 show how they were made and can be used to regenerate them.

### 6.1 Unpack the alignments

```bash
cd "$RECIPE/length-series/alignments"
for f in *.tar.gz; do tar -xzf "$f"; done
```

Each protein gets `length-series/alignments/<ID>_A/` with the same four files as `alignments/6KWC_1/`. The alignments are stored compressed (35 MB in total) because they are large as plain text: 288 MB in total, and 138 MB for 8P5S alone, more than the 100 MB per-file limit on GitHub.

### 6.2 Regenerate inputs (optional, login node)

```bash
cd "$RECIPE"
python scripts/prepare_length_series.py
```

This writes `length-series/inputs/<ID>_A.fasta` (the chain's full sequence) and copies the experimental structure to `length-series/reference/<ID>.cif`, both from the PACE PDB copy. Nothing is downloaded.

### 6.3 Recompute alignments (optional, CPU nodes, about 6 hours)

```bash
cd "$RECIPE"
sbatch slurm/compute_alignments.slurm
```

This runs OpenFold's `scripts/precompute_alignments.py` once per protein, as a 5-task job array on `coc-cpu` (8 CPUs and 64 GB each), against the AlphaFold databases on PACE: UniRef90 and MGnify (JackHMMER), BFD and UniRef30 (HHblits), and PDB70 (HHsearch, template hits). It writes the same `length-series/alignments/<ID>_A/` folders as step 6.1. A protein whose folder already exists is skipped, so delete a folder to recompute it. To pack the results again:

```bash
cd "$RECIPE/length-series/alignments"
for d in */; do d=${d%/}; tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-07 00:00Z' -cf - "$d" | gzip -n > "$d.tar.gz"; done
```

The committed alignments come from job 6852413 (2026-11-06, logs in `logs/length-series-alignments/`), which ran all five tasks in parallel and finished `COMPLETED` in under 6 hours:

| Protein | Wall time | Peak memory | Alignments | Compressed |
|---|---|---|---|---|
| 8P5S_A | 5 h 49 min | 38 GB | 138 MB | 15 MB |
| 9LB7_A | 1 h 49 min | 41 GB | 91 MB | 13 MB |
| 9NXH_A | 2 h 43 min | 41 GB | 56 MB | 6.3 MB |
| 9OUF_A | 2 h 42 min | 37 GB | 0.8 MB | 0.2 MB |
| 9Q5U_A | 2 h 40 min | 38 GB | 2.0 MB | 0.3 MB |

Wall time depends on the shared file system as much as on length. Peak memory stays near 40 GB for every protein, so keep `--mem=64G`. 9OUF and 9Q5U find few related sequences (small folders), which makes them useful low-MSA test cases. A rerun can find slightly different hits if the PACE databases are updated.

## What was tested

Everything below was run on PACE/ICE on 2026-10-11 with one NVIDIA A100-PCIE-40GB (`coc-gpu`).

1. **Original run** (existing environment): inference exit status 0 (`logs/inference.log`), validation in `logs/validation.log`.
2. **Clean-environment rerun** (`openfold-repro-test`, built under `~/scratch/.conda/envs`): inference was run through `slurm/openfold_inference.slurm` with a separate `OUTPUT_DIR`. OpenFold exited with status 0 (`logs/clean-reproduction.log`, `logs/clean-reproduction-validation.log`).
3. **Guide walkthrough**: sections 1.2, 1.3, 3 (Option B commands), and 4 were then run exactly as written, in a shell without any personal Conda initialization, creating `openfold-env` with a single `mamba env create`. The import check printed `2.5.1 12.4`, `run_pretrained_openfold.py --help` worked, and inference exited with status 0 (`logs/guide-walkthrough.log`).

4. **Batch job**: `sbatch slurm/openfold_inference.slurm` was submitted from the recipe directory as in section 3, Option A. Job 6200577 finished `COMPLETED` with exit code 0:0 in 4 min 18 s wall clock, using about 2.8 GB of memory (`logs/sbatch-6200577.out`, `logs/sbatch-6200577.err`).

All four runs produced a **byte-identical** prediction (SHA256 above) and the same validation numbers.

Notes:

- The environment resolved to the latest compatible package versions on that date, which may differ from `requirements-lock.txt`.
- All runs used the same GPU model. Other GPUs or package versions may give small numerical differences; use the validation numbers, not the checksum, to judge a reproduction.

## Troubleshooting

These are the failures we hit on PACE/ICE while building this recipe, and how each was resolved.

| Symptom | Cause | Fix |
|---|---|---|
| `mamba env create` killed with exit code 137 | The login node's per-user memory limit is too low for the environment solver | Run section 1.2 inside a `salloc` allocation with `--mem=32G` |
| `OSError: [Errno 122] Disk quota exceeded` during `conda env create` | Conda package cache and environment were written to `$HOME` | Set `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH`, `PIP_CACHE_DIR`, and `TMPDIR` to scratch (section 1.2) |
| `Pip failed` at `flash-attn` ("Getting requirements to build wheel ... error") during `mamba env create` | `flash-attn` needs PyTorch at build time, which pip's isolated build does not see | Install it afterwards with `pip install flash-attn==2.8.3 --no-build-isolation` (section 1.3) |
| `fatal error: cannot execute 'cc1plus'` and `nvcc fatal: Failed to preprocess host compiler properties` in `install_third_party_dependencies.sh` | The default PACE `gcc` module sets `GCC_ROOT`, which misdirects the Conda compiler | `unset GCC_ROOT` before running the script (section 1.3) |
| `ModuleNotFoundError: No module named 'attn_core_inplace_cuda'` | The previous step failed, so OpenFold's CUDA kernel was not built | Fix the compiler error above and rerun the install script |
| `EnvironmentNameNotFound: openfold-env` | `CONDA_ENVS_PATH` is not set in the current shell | `export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"` |
| `jackhmmer: error while loading shared libraries: libopenblas.so.0` in the alignment job | An environment created before `libopenblas` was added to `environment.yml` | On a compute node: `mamba install -n openfold-env --freeze-installed -c conda-forge libopenblas` |
| `run_inference.sh` stops with `ERROR: No CUDA GPU visible` | Running on a login node | Run inside a GPU allocation (section 3) |

Warnings about `libaio`, `torch.cuda.amp` deprecation, and the Triton cache directory appear in successful runs and can be ignored.

## Runtime and resources

On one A100-PCIE-40GB with 8 CPUs and 32 GB of memory, model inference for this 191-residue target took about 8 to 17 seconds, and the whole OpenFold command of the original run took 1 min 41 s of wall-clock time. That run peaked at about 2.1 GB of host memory (`logs/inference.log`). Creating the environment took about 6 minutes with a warm package cache.

## Cleanup

```bash
rm -rf "$RECIPE/results/run"                          # your own prediction output
conda env remove -n openfold-env                      # the environment (needs CONDA_ENVS_PATH set)
rm -rf ~/scratch/.conda/pkgs ~/scratch/.cache/pip     # package caches
rm -rf ~/scratch/openfold                             # the OpenFold checkout
```

## Files

| Path | Contents |
|---|---|
| `environment.yml` | Conda environment specification |
| `requirements-lock.txt` | Package versions of the original run |
| `scripts/run_inference.sh` | Inference command with preflight checks |
| `scripts/validate_structure.py` | Sequence, pLDDT, and RMSD validation |
| `slurm/openfold_inference.slurm` | PACE/ICE batch script |
| `scripts/prepare_length_series.py` | Writes length-series inputs and reference structures |
| `slurm/compute_alignments.slurm` | Computes length-series alignments on CPU nodes |
| `length-series/` | Five proteins of different lengths: list, inputs, compressed alignments, experimental structures |
| `logs/length-series-alignments/` | Logs of the alignment job (6852413) |
| `logs/` | Environment, GPU, inference, and validation logs |
| `results/`, `reference/`, `visualizations/` | Reference prediction, experimental structure, images |
