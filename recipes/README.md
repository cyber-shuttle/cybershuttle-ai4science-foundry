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
git clone [https://github.com/VisheshJ2007/cybershuttle-ai4science-foundry.git](https://github.com/VisheshJ2007/cybershuttle-ai4science-foundry.git)
cd cybershuttle-ai4science-foundry
git checkout fno-inference-workflow
cd recipes

# 2. Load Anaconda and create the environment in scratch storage
module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda env create -f environment.yml -p ~/scratch/conda-envs/fno-inference
conda activate ~/scratch/conda-envs/fno-inference
```

*(If you prefer a standard named environment and have home quota available, run `conda env create -f environment.yml && conda activate fno-inference`).*

---

## 4. Execution Workflow

### Option A: GPU Batch Submission via Slurm (Recommended)
From inside the `recipes/` directory:
```bash
sbatch submit_inference.sbatch
```
Monitor job progress with `squeue -u $USER`. Once complete, inspect `slurm_fno_inference_<JOB_ID>.out` and `outputs/smoke_test_output.json`.

### Option B: Interactive / Standalone Execution (GPU or CPU)
From inside the `recipes/` directory with the Conda environment activated:
```bash
python run_inference.py
```

---

## 5. Validation Gates & Expected Outputs
`run_inference.py` automatically runs two verification checks and writes results to `outputs/smoke_test_output.json`:

1. **Smoke Test (Single Forward Pass):**
   - Output Tensor Dimension: `[1, 1, 16, 16]` (Matches ground truth)
   - Single-sample Latency: `190.39 ms`
   - Status: `PASSED`
2. **Substantive Scientific Validation:**
   - Metric: Relative L2 Error Norm (`||u_pred - u_true||_2 / ||u_true||_2`)
   - Expected Mean Relative L2 Error: `0.0777` (Std: `0.0183`, threshold `< 0.20`)
   - Expected Mean Squared Error (MSE): `0.001452`
   - Status: `PASSED`
3. **Hardware Acceleration Comparison:**
   - CPU Batch Latency (16 samples): `~3.50 s`
   - NVIDIA A40 GPU Batch Latency (16 samples): `0.0682 s` (>50x speedup)
