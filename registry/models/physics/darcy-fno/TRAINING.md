# Darcy Training and Checkpoint Creation — Reproducible GPU Workflow

This project demonstrates a fully reproducible training workflow for the
Fourier Neural Operator (FNO) on the Darcy flow dataset using the
NeuralOperator library.

The workflow:

- trains the FNO_Small2d model
- evaluates model performance during training
- saves model checkpoints
- records training logs
- documents hardware compatibility findings

The goal is to provide a reproducible example of scientific machine learning
training on PACE ICE GPUs.

---

## Upstream Dependencies

This workflow depends on the official NeuralOperator repository:

- NeuralOperator: https://github.com/neuraloperator/neuraloperator
- FNO paper: https://arxiv.org/abs/2010.08895

Training uses the built-in Darcy Flow dataset shipped with
NeuralOperator.

---

## Expected Directory Structure

Clone NeuralOperator and place this recipe directory alongside it:

```text
project-root/
├── neuraloperator/
│   └── ...
└── darcy-fno/
    ├── TRAINING.md
    ├── train_darcy_checkpoint.py
    ├── environment.yml
    ├── checkpoints/
    ├── ...
    └── training_logs/
```

The modified training script should be copied into:

```text
neuraloperator/scripts/
```

before execution.

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

### 3. Create and activate environment

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

## GPU Requirements

Training requires a GPU. 

Tested successfully on:

```text
NVIDIA A100
```

Known incompatible hardware:

```text
NVIDIA RTX PRO 6000 Blackwell
```

The current PyTorch environment does not support
compute capability:

```text
sm_120
```

and produces:

```text
RuntimeError:
CUDA error: no kernel image is available
for execution on the device
```

---

## Requesting an A100 Allocation

From a login node:

```bash
salloc \
    --partition=ice-gpu \
    --gres=gpu:a100:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:30:00
```
Note: If A100 is not available, H100 may work as well, but that has not been fully tested yet.

Verify allocation:

```bash
hostname
nvidia-smi
```

Expected GPU:

```text
A100
```
---

## Training Scripts

Two training scripts are available.

### Standard NeuralOperator Training

The official NeuralOperator training script is:

```text
scripts/train_darcy.py
```

This script reproduces the standard Darcy training workflow but does **not**
save checkpoints by default.

To run the standard workflow:

```bash
cd scripts

python train_darcy.py \
    2>&1 | tee \
    ../../darcy-fno/training_logs/darcy_training_full.log
```

This workflow is useful for users who only wish to reproduce training and
observe the resulting metrics.

Successful execution will produce:

```text
training_logs/
└── darcy_training_full.log
```

The log contains:

- model configuration
- optimizer configuration
- scheduler configuration
- epoch-by-epoch training metrics
- evaluation metrics
- final training results

### Checkpoint-Enabled Training

This repository provides:

```text
train_darcy_checkpoint.py
```

which is a modified version of `train_darcy.py` that enables checkpoint
generation through the NeuralOperator Trainer API.


Move into your darcy-fno folder and copy the provided script into the NeuralOperator scripts directory:

```bash
cd ../darcy-fno
cp train_darcy_checkpoint.py \
   ../neuraloperator/scripts/
```

Change into the scripts directory:

```bash
cd ../neuraloperator/scripts
```

Run training and save a complete training log:

```bash
python train_darcy_checkpoint.py \
    2>&1 | tee \
    ../../darcy-fno/training_logs/darcy_checkpoint_training_full.log
```

During training, NeuralOperator should report:

```text
[Rank 0]: saved training state to ./checkpoints
```

indicating that checkpoint generation is functioning correctly.

Successful execution will produce:

```text
darcy-fno/checkpoints/
├── model_state_dict.pt
├── model_metadata.pkl
├── optimizer.pt
├── scheduler.pt
└── manifest.pt
```

### Verify Checkpoint Creation

Confirm that checkpoint files were created:

```bash
ls ../../darcy-fno/checkpoints
```

Expected output:

```text
manifest.pt
model_metadata.pkl
model_state_dict.pt
optimizer.pt
scheduler.pt
```

---

### Reference Training Log

This repository includes a successful checkpoint-enabled training run:

```text
darcy_checkpoint_training_full.log
```

and a successful standard run:

```
darcy_training_full.log
```

Users may compare their output against this log to verify:

- configuration settings
- model architecture
- optimizer configuration
- training progress
- evaluation metrics
- checkpoint creation

Minor numerical differences between runs are expected, but the overall
behavior and metric trends should be similar.

Checkpointing is functioning correctly when the training log contains:

```text
[Rank 0]: saved training state to ./checkpoints
```

---

### Expected Runtime

On an NVIDIA A100 GPU, the full 300-epoch workflow completed successfully
and produced the expected checkpoint files.

Actual runtime may vary depending on hardware allocation and cluster load. It took me about 15 minutes give or take.


## Expected Configuration

Model:

```text
FNO_Small2d
```

Parameters:

```text
671,113
```

Dataset:

```text
Darcy Flow Small
```

Training samples:

```text
1000
```

Testing samples:

```text
100 @ 16x16
50 @ 32x32
```

Epochs:

```text
300
```

---

## Generated Checkpoints

Successful execution produces:

```text
checkpoints/
├── model_state_dict.pt
├── model_metadata.pkl
├── optimizer.pt
├── scheduler.pt
└── manifest.pt
```

These files contain:

- trained model weights
- optimizer state
- scheduler state
- checkpoint metadata

---

## Final Training Results

Final epoch:

```text
299
```

Training metrics:

```text
avg_loss = 0.0230
train_err = 0.1837
```

Evaluation metrics:

```text
16_h1 = 0.2062
16_l2 = 0.1199

32_h1 = 0.5497
32_l2 = 0.1669
```

---

## Produced Artifacts

training_logs/
├── darcy_checkpoint_training_full.log
├── darcy_training_full.log
└── training_history.txt

checkpoints/
├── model_state_dict.pt
├── optimizer.pt
├── scheduler.pt
├── manifest.pt
└── model_metadata.pkl

---

## Reproducibility Summary

This workflow reproduces:

- NeuralOperator Darcy training
- GPU execution on PACE ICE
- checkpoint generation
- evaluation metrics
- training log capture

The resulting checkpoint can be loaded by
`darcy_checkpoint_inference.py` and used for inference without
repeating training. See `CHECKPOINT_INFERENCE.md`.