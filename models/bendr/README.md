BENDR Inference Workflow

This repository provides a reproducible inference workflow for BENDR, utilizing self-supervised transformer representations for EEG data, adapted for execution on the GT PACE ICE HPC cluster.

--

1. Environment & Setup
The workflow runs within an isolated Conda environment () configured with Python 3.10 and PyTorch with CUDA support.

2. Setup Instructions
1. Clone the repository and navigate to the model directory:
   ```bash
   git clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git
   cd cybershuttle-ai4science-foundry/models/bendr
   ```
2. Create and activate the environment:
   ```bash
   conda create -n bendr-env python=3.10 -y
   conda activate bendr-env
   ```
3. Install required dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install moabb pyyaml-include mne
   ```

---

2. Model, Data & Artifact Traceability
- Source Model Repository:[SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR)
- Pre-trained Weights / Checkpoints: Lightweight initialization / smoke test weights compatible with DN3 framework specifications.
- Dataset / Inputs: Synthesized/dummy electroencephalography (EEG) tensor inputs configured to match standard BENDR input dimensions (`[batch_size, channels, time_steps]`).

---

3. Inference Workflow & Job Scripts
- Slurm Batch Job (`job_infer.slurm`):Configured for the `coc-gpu` partition. Submit with `sbatch job_infer.slurm`.
- Inference Script (`infer_smoke_test.py`): Executes forward pass with dummy EEG tensors.

---

4. Execution Evidence & Logs
Sample execution logs are stored under `models/bendr/logs/` confirming successful tensor generation (`torch.Size([2, 512])`).

---

5. Validation of Results
- Observed Output Tensor Shape:`torch.Size([2, 512])`
- Conclusion:** Confirms Python 3.10 syntax compatibility and CUDA device execution.
