# Reproducible FNO-2D Inference on Georgia Tech PACE ICE Cluster

This recipe provides a self-contained, reproducible inference and validation pipeline for a 2D Fourier Neural Operator (FNO-2D) surrogate model evaluated on steady-state Darcy Flow pressure field benchmarks on Georgia Tech's AI Makerspace / PACE ICE cluster.

---

## 1. System & Hardware Provenance
- **Compute Cluster:** Georgia Tech PACE ICE (`login-ice.pace.gatech.edu`)
- **Validated GPU Hardware:** NVIDIA A40 (48 GB VRAM) / NVIDIA A100 (`coc-gpu` / `ice-gpu` partitions), also supports CPU fallback
- **Software Stack:** Python 3.10, PyTorch (>=2.2.0), `neuraloperator==0.3.0`, `torch-harmonics==0.9.2`

---

## 2. Bundled Model & Data Artifacts
All required evaluation tensors and model weights are bundled directly in this directory for turnkey execution without external database downloads:
- **Model Checkpoint (`fno_darcy_checkpoint.pt`):** Pretrained FNO-2D state dictionary (`n_modes=(8, 8)`, `hidden_channels=32`, `in_channels=1`, `out_channels=1`).
- **Evaluation Dataset (`darcy_test_sample.pt`):** 16 evaluation instances at `16x16` spatial resolution from the standard Darcy Flow benchmark (`x`: permeability input field `[16, 1, 16, 16]`, `y`: reference PDE solver pressure field `[16, 1, 16, 16]`).

---

## 3. Environment Setup & Reproduction (Using Scratch Storage)
To avoid PACE home directory quota limits, clone the repository and build the Conda environment inside your `~/scratch` directory:

```bash
# 1. Navigate to scratch space and clone the repository
mkdir -p ~/scratch && cd ~/scratch
git clone [https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git)
cd cybershuttle-ai4science-foundry
git checkout fno-inference-workflow
cd recipes

# 2. Load Anaconda and create the environment in scratch storage
module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda env create -f environment.yml -p ~/scratch/conda-envs/fno-inference
conda activate ~/scratch/conda-envs/fno-inference
