# Checkpoint-Based Inference — Reproducible Trained FNO Workflow

This project demonstrates a fully reproducible inference workflow using
a trained Fourier Neural Operator (FNO) checkpoint produced through the
Darcy training workflow documented in `TRAINING.md`.

Unlike the workflow described in `INFERENCE.md`, this workflow uses
trained model weights obtained after 300 epochs of supervised learning.

The workflow:

- loads a trained NeuralOperator checkpoint
- reconstructs the trained FNO architecture
- loads Darcy test data
- performs inference using trained weights
- verifies checkpoint correctness
- computes prediction quality metrics

The goal is to demonstrate how trained NeuralOperator models can be
saved, reused, and executed without repeating the training process.

---

## Relationship to Other Workflows

This repository contains three related workflows:

### Random Model Inference

Documented in:

```text
INFERENCE.md
```

Workflow:

```text
Dataset
    ↓
Randomly Initialized FNO
    ↓
Inference
```

---

### Training and Checkpoint Creation

Documented in:

```text
TRAINING.md
```

Workflow:

```text
Dataset
    ↓
Training
    ↓
Checkpoint Creation
```

---

### Checkpoint-Based Inference

Documented here.

Workflow:

```text
Checkpoint
    ↓
Load Trained Weights
    ↓
Inference
```

---

## Upstream Dependencies

This workflow depends on the official NeuralOperator repository:

- NeuralOperator: https://github.com/neuraloperator/neuraloperator
- FNO paper: https://arxiv.org/abs/2010.08895

The trained weights were produced using the workflow described in
`TRAINING.md`.

---

## Expected Directory Structure

Clone NeuralOperator and place this recipe directory alongside it:

```text
project-root/
├── neuraloperator/
│   └── ...
└── darcy-fno/
    ├── CHECKPOINT_INFERENCE.md
    ├── darcy_checkpoint_inference.py
    ├── checkpoints/
    │   ├── model_state_dict.pt
    │   ├── optimizer.pt
    │   ├── scheduler.pt
    │   ├── manifest.pt
    │   └── model_metadata.pkl
    └── ...
```

The checkpoint inference script expects trained model weights to exist in:

```text
checkpoints/model_state_dict.pt
```

relative to the workflow directory.

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

### 3. Create and Activate Environment

```bash
conda env create -f environment.yml
conda activate neuralop
```

### 4. Install NeuralOperator

```bash
cd neuraloperator
pip install -e .
```

---

## Verify Checkpoint Availability

Before running inference, confirm that checkpoint files exist:

```bash
ls checkpoints
```

Expected output:

```text
manifest.pt
model_metadata.pkl
model_state_dict.pt
optimizer.pt
scheduler.pt
```

If these files do not exist, complete the training workflow described in
`TRAINING.md` first.

---

## Preparing the Inference Script

This repository provides:

```text
darcy_checkpoint_inference.py
```

which loads a trained FNO checkpoint and performs inference using the
trained model weights.

Because the script relies on NeuralOperator configuration files and
imports located inside the NeuralOperator repository, it should be copied
into the NeuralOperator scripts directory before execution.

Move into the workflow directory:

```bash
cd darcy-fno
```

Copy the script into the NeuralOperator scripts directory:

```bash
cp darcy_checkpoint_inference.py \
   ../neuraloperator/scripts/
```

Change into the scripts directory:

```bash
cd ../neuraloperator/scripts
```

The inference workflow should be executed from this location.

---

## Running Checkpoint-Based Inference

From the workflow directory:

```bash
python darcy_checkpoint_inference.py
```

The script will:

1. Load the trained checkpoint.
2. Reconstruct the trained FNO architecture.
3. Load a Darcy test sample.
4. Perform inference using trained weights.
5. Compute prediction statistics.
6. Compute mean squared error (MSE) against the target solution.

---

## Expected Output

Successful execution should report:

```text
Checkpoint loaded successfully.

Input shape:
torch.Size([1, 1, 16, 16])

Target shape:
torch.Size([1, 1, 16, 16])

Output shape:
torch.Size([1, 1, 16, 16])
```

The exact prediction values may vary slightly depending on software and
hardware environment, but output dimensions should remain unchanged.

---

## Reference Inference Results

Checkpoint-based inference produced:

```text
Prediction statistics

min  = -0.0226
max  = 0.9674
mean = 0.3862
```

Prediction quality:

```text
MSE = 0.00232
```

These values were obtained using the checkpoint produced by the
training workflow documented in `TRAINING.md`.

---

## Why Checkpoint-Based Inference Matters

Training the FNO model required:

```text
300 epochs
1000 training samples
671,113 learnable parameters
```

Checkpointing allows these trained weights to be reused without repeating
the full training workflow.

This changes the workflow from:

```text
Dataset
    ↓
Training
    ↓
Inference
```

to:

```text
Checkpoint
    ↓
Inference
```

which is considerably faster and more practical for downstream users.

---

## Produced Artifacts

```text
checkpoints/
├── model_state_dict.pt
├── optimizer.pt
├── scheduler.pt
├── manifest.pt
└── model_metadata.pkl
```

Required inference script:

```text
darcy_checkpoint_inference.py
```

---

## Validation

This workflow verifies:

- successful checkpoint loading
- successful reconstruction of the training architecture
- successful inference execution
- matching tensor dimensions
- finite numerical outputs
- successful computation of prediction error

---

## Reproducibility Summary

This workflow reproduces:

- checkpoint loading
- trained-model inference
- prediction quality validation
- NeuralOperator model reuse without retraining

Together with `INFERENCE.md` and `TRAINING.md`, this document completes a
full NeuralOperator lifecycle:

```text
Random Inference
    ↓
Training
    ↓
Checkpoint Creation
    ↓
Checkpoint-Based Inference
```