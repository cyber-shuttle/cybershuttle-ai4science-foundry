# NeuralOperator Reproduction Workflows

This project documents a series of reproducible NeuralOperator workflows
performed on Georgia Tech PACE infrastructure.

Available workflows:

1. Random Model Inference
2. Darcy Training and Checkpoint Creation
3. Checkpoint-Based Inference

## Documents

- INFERENCE.md
- TRAINING.md
- CHECKPOINT_INFERENCE.md

## Environment Setup

See environment.yml.

## Hardware Notes

Training is known to work on A100 GPUs and H100s probably work too.

Blackwell GPUs may fail due to unsupported CUDA architecture in the
current PyTorch environment.

See troubleshooting.md.