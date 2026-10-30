# Darcy Inference — Reproducible SLURM Workflow

This project demonstrates a fully reproducible inference workflow for the Fourier Neural Operator (FNO) on the Darcy flow dataset using the NeuralOperator library.

The repository includes:

- a standalone inference script (`darcy_inference.py`)
- a SLURM batch script (`run_darcy_inference.slurm`)
- environment specification (`environment.yml`)
- execution logs
- validation, troubleshooting, and provenance documentation

The goal is to provide a clean, minimal, and fully reproducible example of running scientific ML inference on PACE ICE.

---

## Upstream Dependencies

This workflow depends on the official NeuralOperator repository:

- NeuralOperator: https://github.com/neuraloperator/neuraloperator
- FNO paper: https://arxiv.org/abs/2010.08895

The Darcy dataset used by this workflow is distributed with NeuralOperator and is loaded directly from the cloned repository.

---

## Expected Directory Structure

Clone NeuralOperator and place this recipe directory alongside it:

```text
project-root/
├── neuraloperator/
│   └── ...
└── darcy-fno/
    ├── README.md
    ├── environment.yml
    ├── darcy_inference.py
    ├── run_darcy_inference.slurm
    ├── validation.md
    ├── troubleshooting.md
    ├── sources.md
    ├── ...
    └── logs/
```

The inference script expects the Darcy dataset at:

```text
../neuraloperator/neuralop/data/datasets/data/darcy_test_16.pt
```

relative to the directory containing `darcy_inference.py`.

---

## Environment Setup

### 1. Clone NeuralOperator

```bash
git clone https://github.com/neuraloperator/neuraloperator.git
```

### 2. Load Anaconda

```bash
module load anaconda3
```

### 3. Create and activate the environment

From the recipe directory:

```bash
conda env create -f environment.yml
conda activate neuralop
```

### 4. Install NeuralOperator

```bash
cd ../neuraloperator
pip install -e .
cd ../darcy-fno
```

---

## Local Testing

Run the inference workflow locally:

```bash
python darcy_inference.py
```

Expected output:

```text
Input shape: torch.Size([1, 1, 16, 16])
Target shape: torch.Size([1, 1, 16, 16])
Output shape: torch.Size([1, 1, 16, 16])
```

The exact numerical values may vary depending on initialization, but the output should:

- have shape `(1, 1, 16, 16)`
- contain finite values
- contain no NaNs or infinities

---

## Running on SLURM (CPU)

Submit the reproducible inference job:

```bash
sbatch run_darcy_inference.slurm
```

The job executes `darcy_inference.py` and writes output to a SLURM log file.

When the job finishes, SLURM will produce:

```text
darcy_inference_<jobid>.out
```

Move the file into the logs directory:

```bash
mv darcy_inference_<jobid>.out logs/
```

The inference script records:

- dataset path
- input and target shapes
- output shape
- numerical statistics (min, max, mean)
- execution device
- node information
- SLURM job metadata

---

## Repository Structure

```text
darcy-fno/
├── README.md
├── environment.yml
├── darcy_inference.py
├── run_darcy_inference.slurm
├── validation.md
├── troubleshooting.md
├── sources.md
├── ...
└── logs/
    └── darcy_inference_<jobid>.out
```

---

## Validation

Validation criteria and expected behavior are documented in:

```text
validation.md
```

The workflow verifies:

- successful dataset loading
- correct tensor dimensions
- successful model execution
- finite numerical outputs
- successful execution on a PACE compute node

---

## Troubleshooting

Known issues and their resolutions are documented in:

```text
troubleshooting.md
```

including:

- conda activation warnings
- SLURM account and partition issues
- environment setup problems

---

## Provenance

Model, dataset, environment, and execution provenance are documented in:

```text
sources.md
```

---

## Reproducibility Summary

This repository provides:

- a reproducible NeuralOperator inference workflow
- a pinned conda environment
- a validated SLURM execution path
- execution logs from PACE ICE
- troubleshooting guidance
- complete provenance documentation

Together, these components allow another student to clone NeuralOperator, reproduce the environment, execute the workflow, and validate the results independently.