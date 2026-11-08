<<<<<<< HEAD
```markdown
# Reproducible Inference: DimeNet++ on QM9

**Student:** Adesh Srivastava  
**Target Platform:** Georgia Tech AI Makerspace (PACE ICE Cluster)  
**Hardware Target:** NVIDIA CUDA-enabled GPUs (A40 / L40S)  

---

This repository contains a fully reproducible inference workflow for evaluating the **DimeNet++** Graph Neural Network on the **QM9** benchmark dataset, executed on the Georgia Tech PACE ICE cluster.

---

## 1. Environment & Setup

The software environment is managed via Conda and pip to ensure PyTorch and PyTorch Geometric correctly interface with the cluster's CUDA 11.8 drivers.

### Step-by-Step Instructions:

1. **Clone this repository and navigate into the recipe directory:**
   ```bash
   git clone [https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git](https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry.git)
   cd cybershuttle-ai4science-foundry/registry/models/chemistry/dimenetpp/recipes/reproducible-inference

```

2. **Configure Scratch Storage (PACE ICE Quota Protection):**
To prevent exceeding the strict 5 GB `$HOME` quota during package extraction, redirect Conda packages, environments, and pip caches to scratch space:
```bash
export CONDA_PKGS_DIRS="$HOME/scratch/.conda/pkgs"
export CONDA_ENVS_PATH="$HOME/scratch/.conda/envs"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TMPDIR="$HOME/scratch/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$TMPDIR"

```


3. **Standard Setup (Conda):**
```bash
conda env create -f environment.yml
conda activate dimenet-env

```


4. **Recommended PACE ICE Fallback Setup (Pre-compiled Pip Wheels):**
*Note: If Conda package extraction or SAT solving hangs due to network filesystem I/O limits on login nodes, build the environment rapidly using pre-compiled wheels:*
=======
# Reproducible Inference: DimeNet++ on QM9
**Student:** Adesh Srivastava

This repository contains a fully reproducible inference workflow for evaluating the **DimeNet++** Graph Neural Network on the **QM9** benchmark dataset, executed on the Georgia Tech AI Makerspace (PACE ICE) cluster.

## 1. Environment & Setup
The software environment is managed via Conda and `pip` to ensure PyTorch and PyTorch Geometric correctly interface with the cluster's CUDA 11.8 drivers.

**Setup Instructions:**
1. Clone this repository and navigate into it.
2. Create the environment from the provided YAML file:
   `conda env create -f environment.yml`
3. Activate the environment:
   `conda activate dimenet-env`

*(Note for PACE ICE: If Conda package extraction hangs due to network filesystem I/O limits, you can alternatively build the environment rapidly using pre-compiled `pip` wheels)*:
>>>>>>> d0755fe426f748c99f026566dc0e95dd0afea13d
```bash
module load anaconda3
conda create -n dimenet-env python=3.10 -y
conda activate dimenet-env

# Install PyTorch with CUDA 11.8 using --extra-index-url to preserve PyPI resolution
pip install torch==2.1.2 torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install core dependencies
pip install torch_geometric==2.4.0 tqdm scikit-learn pandas "numpy<2"
<<<<<<< HEAD

# Install PyG accelerated binary extensions
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f [https://data.pyg.org/whl/torch-2.1.2+cu118.html](https://data.pyg.org/whl/torch-2.1.2+cu118.html)

```



---

## 2. Artifact Traceability (Model, Data, & Provenance)

* **Model Architecture:** DimeNet++ (Directional Message Passing Neural Network). Implemented natively via `torch_geometric.nn.models.DimeNetPlusPlus`.
* **Dataset:** QM9 Benchmark. Contains $\sim 134\text{k}$ small organic molecules. Automatically downloaded via `torch_geometric.datasets.QM9` from the MoleculeNet DeepChem mirror.
* **Primary Paper:** Gasteiger et al., *Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules* (NeurIPS 2020). [Source Repository](https://github.com/gasteigerjo/dimenet).
* **Weights & Checkpoint Limitation:** Currently, `evaluate.py` initializes the `DimeNetPlusPlus` architecture with untrained (randomly initialized) weights to validate the end-to-end forward-pass geometry, 3D spatial graph construction (`radius_graph`), and GPU tensor execution across the cluster without dependency on external S3 buckets. Because the model is evaluated without a trained checkpoint, the resulting Mean Absolute Error (MAE) on Target 7 ($U_0$) is expectedly high ($12197.0133$). A subsequent release will provide pre-trained checkpoint weights.

---

## 3. Inference Workflow & Job Scripts

The inference pipeline is contained in `evaluate.py`. It constructs 3D spatial graphs (atomic numbers and coordinates) and passes them through DimeNet++ interaction blocks to predict physical molecular properties.

### Step 3.1: Pre-Downloading the Dataset (PACE ICE Firewall Bypass)

Compute nodes on PACE ICE may restrict outbound internet access. Before submitting the batch job, execute the smoke test once on the login node (or an interactive compute node) to cache the $\sim 100\text{ MB}$ QM9 dataset locally in `./data`:

```bash
python evaluate.py --smoke-test

```

### Step 3.2: Execution Options

#### Option A: Batch Submission via SLURM (Recommended)

Submit the job script directly to the cluster scheduler:

```bash
sbatch run_inference.slurm

```

The script requests 1 standard GPU (`--gres=gpu:1`), 16GB of RAM, and 4 CPU cores for 30 minutes, fitting well within PACE ICE allocation limits.

> **Cluster Allocation & Portability Note:**
> By default, `run_inference.slurm` specifies `#SBATCH --account=coc` and `#SBATCH --partition=coc-gpu`. If you belong to a different department, course allocation, or research group on PACE ICE, please update these two lines at the top of `run_inference.slurm` prior to submission. Alternatively, pass overrides directly on the command line:
> ```bash
> sbatch --account=<your_account> --partition=<your_partition> run_inference.slurm
> 
> ```
> 
> 
> Specifying an explicit account and partition prevents silent job rejection or accidental submission to unaccelerated queues on heterogeneous clusters.

#### Option B: Interactive Execution via `salloc`

To run directly within an interactive session:

```bash
salloc --account=coc --partition=coc-gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00
conda activate dimenet-env
python evaluate.py --target 7

```

---

## 4. Validation of Results

This workflow includes two levels of validation, directly embedded in the execution logs:

1. **Smoke Test:** Executes `python evaluate.py --smoke-test`. This runs a single batch of 32 molecules (724 atoms) to verify that the environment is functional, the dataset loads properly, and the output tensor shapes (`[32, 1]`) align without CUDA memory faults.
2. **Substantive Validation:** Executes a full evaluation over the 10,000-molecule test split targeting index 7 (Internal Energy $U_0$ at 0K). It calculates the Mean Absolute Error (MAE) across all 313 batches, verifying directional message-passing execution across the dataset. As noted in Section 2, an MAE of approximately **$12197.0133$** is expected due to the unparameterized baseline weights.

---

## 5. Execution Evidence

Reference execution evidence is provided in the repository as `execution_6229368.log`. This log contains:

* Node allocation details (`atl1-1-01-005-9-0`) and GPU hardware (`NVIDIA A40`).
* Dataset caching and processing milestones.
* Output tensor shape verification from the smoke test.
* The benchmark MAE metric ($12197.0133$ achieved in 53 seconds walltime) from the substantive validation run.

```

```
=======
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

## 2. Artifact Traceability (Model, Data, & Provenance)
- **Model Architecture:** DimeNet++ (Directional Message Passing Neural Network). Implemented natively via `torch_geometric.nn.models.DimeNetPlusPlus`.
- **Dataset:** QM9 Benchmark. Contains ~134k small organic molecules. Automatically downloaded via `torch_geometric.datasets.QM9` from the MoleculeNet DeepChem mirror.
- **Primary Paper:** Gasteiger et al., *Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules* (NeurIPS 2020). Source repository: `https://github.com/gasteigerjo/dimenet`.
- **Weights & Checkpoint Limitation:** Currently, `evaluate.py` initializes the `DimeNetPlusPlus` architecture with **untrained (randomly initialized) weights** to validate the end-to-end forward-pass geometry, 3D spatial graph construction (`radius_graph`), and GPU tensor execution across the cluster. Because the model is evaluated without a trained checkpoint, the resulting Mean Absolute Error (MAE) on Target 7 ($U_0$) is expectedly high (`12197.0133`). I will release an update to this repository with instructions and support for running this workflow with loaded pre-trained weights.

## 3. Inference Workflow & Job Scripts
The inference pipeline is contained in `evaluate.py`. It constructs the 3D spatial graphs (atomic numbers and coordinates) and passes them through the DimeNet++ interaction blocks to predict physical properties.

**Pre-Downloading the Dataset (PACE ICE Firewall Bypass):**
Compute nodes on PACE ICE may restrict outbound internet access. Before submitting the batch job, run the smoke test once on the login node (or an interactive node) to cache the ~100 MB QM9 dataset locally in `./data`:
`python evaluate.py --smoke-test`

**To run on AI Makerspace (SLURM):**
Submit the job script directly to the scheduler:
`sbatch run_inference.slurm`

The script requests 1 standard GPU (`--gres=gpu:1`), 16GB of RAM, and 4 CPU cores for 30 minutes, fitting well within the 32 GPU-hour PACE ICE allocation limits.

## 4. Validation of Results
This workflow includes two levels of validation, directly embedded in the execution logs:
1. **Smoke Test:** Executes `python evaluate.py --smoke-test`. This runs a single batch of 32 molecules (724 atoms) to verify that the environment is functional, the dataset loads properly, and the output tensor shapes (`[32, 1]`) align without CUDA memory crashes.
2. **Substantive Validation:** Executes a full evaluation over the 10,000-molecule test split targeting index 7 (Internal Energy $U_0$ at 0K). It calculates the Mean Absolute Error (MAE) across all 313 batches, verifying the directional message-passing execution across the dataset. As noted in Section 2, an MAE around `12197.0133` is expected because the model is currently loaded with untrained weights.

## 5. Execution Evidence
Execution evidence is provided in the repository as `execution_6229368.log`. This file contains the node (`atl1-1-01-005-9-0`) and GPU (`NVIDIA A40`) allocation data, dataset processing milestones, expected output tensor shapes from the smoke test, and the final MAE metric (`12197.0133` in 53 seconds walltime) from the substantive validation run.
>>>>>>> d0755fe426f748c99f026566dc0e95dd0afea13d
