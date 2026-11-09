
# DNABERT-2: Reproducible Inference on AI Makerspace

## Overview

This project reproduces pretrained DNABERT-2 inference on Georgia Tech's AI Makerspace through the PACE ICE cluster using an NVIDIA A100 GPU. The workflow follows the official DNABERT-2 Quick Start example to generate a 768-dimensional embedding from a DNA sequence using mean pooling.

The project includes a reproducible Python environment, pinned model revision, Slurm job scripts, inference outputs, validation checks, execution logs, and troubleshooting documentation.

This covers pretrained inference.

## Model and Source Information

- Model: DNABERT-2 (117M parameters)
- Hugging Face: https://huggingface.co/zhihan1996/DNABERT-2-117M
- Model revision: `7bce263b15377fc15361f52cfab88f8b586abda0`
- Official repository: https://github.com/MAGICS-LAB/DNABERT_2
- Architecture: Transformer-based DNA foundation model using BPE tokenization and ALiBi positional bias
- Model weights: Downloaded automatically from Hugging Face using `from_pretrained()`
- Custom model code: Loaded using `trust_remote_code=True`
- Output: 768-dimensional mean-pooled sequence embedding

The checkpoint revision is pinned in both Python scripts so that the same model weights and custom code are retrieved. The model files are cached in `.hf-cache/` and do not need to be committed to GitHub.

Because `trust_remote_code=True` runs external Python code, the model source should be reviewed before execution.

## Inputs

The reference sequence is taken from the official DNABERT-2 README, Section 4 (Quick Start).

It is stored in `inputs/reference.txt` to document the input used for inference. The inference script uses the same sequence directly in Python.

Validation uses:
- The original reference sequence.
- An identical copy to test consistency.
- A modified sequence with one nucleotide substitution.

The SHA-256 checksum of the reference input file is recorded in `evidence/input-sha256.txt`.

To check the input file:

```bash
cat inputs/reference.txt
sha256sum -c evidence/input-sha256.txt
```

No external dataset is required for this workflow.

## Hardware and Compute Environment

Inference and validation were executed on Georgia Tech's PACE ICE cluster through Slurm.

| Resource | Configuration |
|---|---|
| Slurm account | `coc` |
| Partition | `coc-gpu` |
| GPU | NVIDIA A100 80GB PCIe |
| GPUs requested | 1 |
| CPUs requested | 2 |
| Memory requested | 16 GB |
| Time limit | 30 minutes |
| Environment | Conda on scratch storage |

Slurm job details are recorded in `evidence/slurm-jobs.txt`, with execution logs in `logs/`.

The job scripts request an A100 GPU using `--gres=gpu:a100:1`. Other users may need to substitute their own Slurm account, partition, or GPU allocation.

## Environment Setup

The original DNABERT-2 requirements are preserved in `requirements-upstream.txt`.

The working software versions are recorded in `requirements.txt` and `evidence/versions.txt`. Conda package details are also recorded in `evidence/conda-explicit.txt`.
The dependencies in `requirements.txt` use installable package versions rather than environment-specific local file paths, allowing others to recreate the environment.

Key versions:
- Python 3.10.22
- PyTorch 1.13.1+cu117
- Transformers 4.29.2
- NumPy 1.24.4
- setuptools 69.5.1

### Create the environment

Connect to PACE ICE, and work from scratch storage from the beginning to avoid exceeding the home directory disk quota. Clone the DNABERT-2 contribution branch into scratch, then enter the project directory.

For example:

```bash
cd ~/scratch
git clone --branch add-dnabert2-inference --single-branch https://github.com/athellis/cybershuttle-ai4science-foundry.git
cd cybershuttle-ai4science-foundry/registry/models/biology/dnabert2
```


Create the Conda environment:

```bash
mamba create -p "$PWD/.conda-env" python=3.10 pip -y
conda activate "$PWD/.conda-env"
```

Set temporary and cache locations on scratch:

```bash
mkdir -p .tmp logs outputs
export TMPDIR="$PWD/.tmp"
export HF_HOME="$PWD/.hf-cache"
export PIP_CACHE_DIR="$PWD/.pip-cache"
```

Install the tested dependencies:

```bash
python -m pip install --no-cache-dir -r requirements.txt
python -m pip check
```

Confirm the key versions:

```bash
python -c "import torch, numpy, transformers, setuptools; print(torch.__version__, numpy.__version__, transformers.__version__, setuptools.__version__)"
```

The `.conda-env`, `.hf-cache`, `.pip-cache`, and `.tmp` directories are excluded from GitHub.

The pinned dependency list records the environment used for the successful runs. Installation on a fresh environment should be checked for compatibility before comparing results.

## Running Inference

Submit the inference job from the project root:

```bash
sbatch jobs/inference.sbatch
```

Check job status:

```bash
squeue -u $USER
```

The job:
- Loads the pinned pretrained DNABERT-2 checkpoint.
- Tokenizes the reference DNA sequence.
- Computes hidden states using the model.
- Applies mean pooling to produce a 768-dimensional embedding.
- Saves the embedding and inference metadata.

Expected output artifacts:
- `outputs/embedding.pt`: Complete embedding tensor.
- `outputs/inference_results.json`: Model revision, device, input sequence, embedding shape, and numerical sample.

A successful run prints `INFERENCE SUCCESSFUL`.

## Validation

Submit the validation job:

```bash
sbatch jobs/validation.sbatch
```

The validation script checks:
1. Embeddings have the expected 768 dimensions.
2. All values are finite.
3. Identical DNA inputs produce identical embeddings.
4. A single-nucleotide substitution produces a measurable change.

### Observed Validation Results

| Metric | Result |
|---|---|
| Embedding dimension | 768 |
| Finite values | True |
| Duplicate maximum absolute difference | 0.0 |
| Duplicate validation | Passed |
| Modified sequence L2 difference | 0.6468818784 |
| Modified sequence cosine similarity | 0.9727207422 |

The inference smoke test produced the expected 768-dimensional embedding with no NaN or infinite values.

The duplicate test produced a maximum absolute difference of 0.0, confirming that identical inputs generated identical embeddings within the same run.

The modified sequence produced an L2 difference of 0.6469 and cosine similarity of 0.9727. This shows that changing one nucleotide affected the embedding while the two representations remained highly similar.

These results demonstrate technical consistency and sensitivity to input changes. They do not establish biological prediction accuracy because the workflow generates sequence embeddings rather than task-specific predictions.

A successful validation job prints `VALIDATION SUCCESSFUL`.

## Execution Evidence

Final successful jobs:

| Job ID | Task | Status | Runtime |
|---|---|---|---|
| 6200298 | Inference | COMPLETED | 6 seconds |
| 6200300 | Validation | COMPLETED | 6 seconds |

Both jobs used the `coc-gpu` partition with an NVIDIA A100 GPU.

Execution logs:
- `logs/dnabert2-inference-6200298.out`
- `logs/dnabert2-inference-6200298.err`
- `logs/dnabert2-validation-6200300.out`
- `logs/dnabert2-validation-6200300.err`

Resource allocation details are available in `evidence/slurm-jobs.txt`.

Earlier unsuccessful attempts are documented in the troubleshooting section. The original failed inference job was `6197615`, followed by successful inference job `6197706`.

## Troubleshooting and Compatibility

### Home Directory Disk Quota

The initial installation of the upstream requirements failed while building `antlr4-python3-runtime` because the PACE home directory reached its 30 GB storage quota.

Error: `[Errno 122] Disk quota exceeded`

I checked storage usage with:

```bash
quota -s
du -sh ~/.cache/pip
du -sh ~/.conda/envs/dnabert2-env
```

The pip cache was using approximately 2.9 GB. I cleared it using:

```bash
python -m pip cache purge
```

I then recreated the environment on the larger scratch filesystem:

```bash
mamba create -p "$PWD/.conda-env" python=3.10 pip -y
conda activate "$PWD/.conda-env"

mkdir -p .tmp
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.pip-cache"

python -m pip install --no-cache-dir -r requirements-upstream.txt
```

The installation completed successfully after moving the environment to scratch. Additional dependency fixes are documented below.

### NumPy Incompatibility

The upstream requirements initially installed NumPy 2.2.6, which was incompatible with PyTorch 1.13.1.

Observed error: `RuntimeError: Numpy is not available`

I resolved this by installing NumPy 1.24.4:

```bash
python -m pip install --no-cache-dir "numpy==1.24.4"
```

I verified the fix using:

```bash
python -c "import torch; print(torch.tensor([1.0, 2.0]).numpy())"
```

The test returned `[1. 2.]`. The working NumPy version is included in `requirements.txt`.

### Missing pkg_resources

The first Slurm inference job (`6197615`) failed during model loading because Accelerate 0.20.3 attempted to import `pkg_resources`.

Observed error: `ModuleNotFoundError: No module named 'pkg_resources'`

I resolved this by installing setuptools 69.5.1:

```bash
python -m pip install --no-cache-dir "setuptools==69.5.1"
```

I verified the import using:

```bash
python -c "import pkg_resources; print('pkg_resources works')"
```

After this fix, inference job `6197706` successfully generated the expected embedding.

### Triton Not Available

DNABERT-2 reported that Triton could not be imported and automatically used its PyTorch attention fallback.

Triton acceleration was not required for this workflow. Both inference and validation succeeded using the fallback implementation.

### Model Weight Warnings

The checkpoint produced warnings about unused prediction-head weights and newly initialized pooler weights.

The workflow uses mean-pooled hidden states rather than the pooler output or a classification head, so these warnings did not prevent inference.

## Reproducing and Verifying Outputs

After running both Slurm jobs, inspect the outputs:

```bash
cat outputs/inference_results.json
cat outputs/validation.json
```

Verify the following:
- Model revision matches `7bce263b15377fc15361f52cfab88f8b586abda0`.
- Both scripts used CUDA.
- Embedding dimension is 768.
- All embedding values are finite.
- Duplicate validation passed.
- Both jobs completed successfully.

Inspect the logs using the job IDs returned by Slurm:

```bash
cat logs/dnabert2-inference-<JOB_ID>.out
cat logs/dnabert2-validation-<JOB_ID>.out
```

The logs should contain `INFERENCE SUCCESSFUL` and `VALIDATION SUCCESSFUL`, respectively.

A separate inference run can also be performed using the same job script to compare regenerated embeddings against the original output.

Minor numerical differences may occur with changes in hardware or software versions.

## Cleanup

After reproducing the workflow, generated outputs and logs can be removed if they are no longer needed:

```bash
rm -rf outputs/ logs/
```

To remove the Conda environment and cached model files:

```bash
conda deactivate
rm -rf .conda-env/ .hf-cache/ .pip-cache/ .tmp/
```

Run these commands only from the project root. Preserve execution evidence before cleanup.

## References

- DNABERT-2 official implementation: https://github.com/MAGICS-LAB/DNABERT_2
- DNABERT-2 pretrained checkpoint: https://huggingface.co/zhihan1996/DNABERT-2-117M
- Zhou et al., *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome* (2023). https://arxiv.org/abs/2306.15006
