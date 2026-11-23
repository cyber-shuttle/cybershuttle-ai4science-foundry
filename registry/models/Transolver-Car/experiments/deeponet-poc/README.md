# Transolver × DeepONet: Accuracy vs. Cost Experiments on ShapeNet-Car

## Description
This repository is a research proof of concept on combining Transover and DeepONet.
It asks how much of Transolver's accuracy on ShapeNet-Car a much lighter DeepONet can retain, and what it saves in
inference time, memory, and input requirements. Variants tested: a plain DeepONet, DeepONets distilled from
Transolver's predictions, and DeepONets whose branch input is a frozen Transolver embedding. My work was done in
`Car-Design-ShapeNetCar/hybrid_deeponet.py`, `jobs/`, `logs/`, and this README. The rest of the code is my slimmed-down copy of
[thuml/Transolver](https://github.com/thuml/Transolver) from [Magical452/Transolver-Car](https://github.com/Magical452/Transolver-Car).
This is exploratory code, not merge-ready.

## Requirements
- Cluster: Georgia Tech PACE ICE (AI Makerspace), Red Hat Enterprise Linux 9.8
- GPU: 1x NVIDIA A100 80GB, partition ice-gpu, account coc, QOS coc-ice
	- Note: This is just for training. For inference and preprocessing, any CUDA GPU with at least 16 GB of memory should work.
- Job resources: 8 CPUs, 32 GB memory
- Module: anaconda3
- Python: 3.8.20
- PyTorch: 2.4.1 (CUDA 12.1 build, installed from pip)
- numpy: 1.24.4 (installed with the constraint <2)
- scipy: 1.10.1 (comes with scikit-learn)
- scikit-learn: 1.3.2
- torch-cluster: 1.6.3 (+pt24cu121, from the PyG wheel index for torch-2.4.0+cu121)
- torch_geometric: 2.6.1
- vtk: 9.5.2
- timm: 1.0.30 (also installs torchvision 0.19.1)
- einops: 0.8.1
- Full pinned list: requirements-frozen.txt
- Disk: about 2 GB for the zip, 5.8 GB unpacked, 14 GB preprocessed

## Installation
### Initializing Model
1. If your computer's IP address is broken (like mine), use a Global Protect VPN from vpn.gatech.edu.
2. Sign in to your login node with `ssh <gt username>@login-ice.pace.gatech.edu`, input your password, and then run `cd scratch`
3. Clone the repo using `git clone https://github.com/Magical452/Transolver_x_DeepONet.git` and then run `cd Transolver`
4. Load anaconda and git into your login node with `module load anaconda3` and `module load git2.45.2` respectively
5. Create and activate the environemtn with these commands:
	- `conda create --prefix $HOME/scratch/envs/transolver python=3.8 -y`
	- `conda activate $HOME/scratch/envs/transolver`
6. Install packages in this order: (Note: the order is important due to torch-cluster needing to match the installed PyTorch and CUDA version)
	- python -m pip install --no-cache-dir torch==2.4.1
	- python -m pip install --no-cache-dir "numpy<2"
	- python -m pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
	- python -m pip install --no-cache-dir torch_geometric==2.6.1
	- python -m pip install --no-cache-dir vtk timm einops scikit-learn
7. Create a directory named "data" in the repo root directory, go into that directory, and run 
`wget -c  http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip`
8. Run `mkdir -p preprocessed_data`, then `cd preprocessed_data`.
9. Run `cd ..`, then `unzip -q mlcfd_data.zip`, then `cd mlcfd_data`, then `cd training_data`.
10. Run `for f in param*.tar.gz; do tar -xzf "$f"; done` to unzip all the tar.gz files (This will take a few minutes, don't interrupt it.)
11. Check: `ls data/mlcfd_data/training_data` should list `param0` through `param8`.
Note: DO NOT REMOVE OR RENAME these folders. The jobs scripts look for them.
### Preprocessing (takes about 1.5 hours)
1. Return to the repo root directory
2. Run `mkdir -p logs`
3. Run `cd jobs` then `nano test.sbatch`, edit `--data_dir` to be the absolute path of your training data folder, and edit `--save_dir` to be the absolute path of 
your preprocessed_data folder 
4. Run `cd ..`, then `sbatch jobs/test.sbatch`
	- It creates a job that runs `main.py` for 1 epoch with `--preprocessed 0` on an A100 for 2 hours. This is our smoke test. 

## Usage
### Checks
1. Confirm the two checkpoints exist at the paths above.
2. Edit the paths in `jobs/poc.sbatch`: `--data_dir`, `--save_dir`, and any exports.
### Operation
1. Submit from the repo root with `bash jobs/submit_poc.sh`, and note that the GPU constraint is already included.
2. Check the queue with `squeue -u <username>`, and follow the log with `tail -f logs/poc-suite-<jobid>.out`.
3. Look for `cuda available: True` near the top of the log. If it says `False`, cancel and resubmit with `sbatch --gres=gpu:1 -C 'cuda12|cuda13' -J poc-suite -o logs/poc-suite-%j.out jobs/poc.sbatch --suite --epochs 40 --drag --cpu_timeC`
4. Check that the job finished (`sacct` command) and that the log ends with `SUITE DONE. failed runs: none`.
5. Read the `FINAL {...}` line for each run at the end of its section; per-run JSON files are in `results_poc/`.

Notes:
- The job runs all 7 variants in priority order and saves each result as it finishes, so partial results survive a timeout.
- Each DeepONet or hybrid run trains in about 2 minutes. Data loading (about 3 minutes) and Transolver teacher/feature
  setup (about 50 s per run) take most of the wall time.

### Variants

| Run | Branch input | Training target |
|---|---|---|
| `deeponet` | subsampled, x-sorted surface point cloud (2048 points, 6144-d) | ground truth |
| `deeponet --distill --distill_alpha 0` | same | Transolver predictions |
| `deeponet --distill --distill_alpha 0.5` | same | 0.5 × ground truth + 0.5 × Transolver |
| `transolver` | n/a (evaluates the trained checkpoint, no training) | n/a |
| `hybrid --embed slice` (200-epoch encoder) | frozen Transolver slice tokens (8192-d) | ground truth |
| `hybrid --embed meanmax` (200-epoch encoder) | frozen mean+max pooled features (512-d) | ground truth |
| `hybrid --embed slice` (100-epoch encoder) | frozen slice tokens, earlier checkpoint | ground truth |

Distillation target: `α · y_true + (1 − α) · y_teacher`, so α = 0 is pure distillation. All DeepONet variants share the
same trunk (Fourier-feature MLP on the 7-d node input of position, SDF and normal) and output 3 velocity components plus
pressure. Loss is MSE on velocity plus `reg × MSE` on surface pressure (`reg = 1.0`). All trained runs used 40 epochs,
learning rate 1e-3, batch size 4, 8192 volume query points per car per step, latent 128, hidden 256, seed 0.

## Validation
### Results (100 held-out test cars, fold 0, single seed, 1× V100)

| Model | Pressure err | Speed err | Drag err | Spearman ρ (drag) | GPU ms/car | CPU ms/car | Extra GPU mem | Trainable params |
|---|---|---|---|---|---|---|---|---|
| Transolver (200 ep) | 0.0773 | 0.0207 | 0.0135 | 0.992 | 58.4 | 665 | 223 MB | 3.86M |
| DeepONet (plain) | 0.307 | 0.101 | 0.215 | 0.274 | 3.6 | 34.5 | 128 MB | 3.72M |
| DeepONet, distilled α=0 | 0.304 | 0.100 | 0.211 | 0.281 | 3.0 | 33.8 | 128 MB | 3.72M |
| DeepONet, distilled α=0.5 | 0.308 | 0.102 | 0.216 | 0.244 | 3.0 | 33.7 | 128 MB | 3.72M |
| Hybrid, slice (enc 200) | 0.137 | 0.067 | 0.027 | 0.961 | 61.7 | 656 | 223 MB | 4.77M (+3.86M frozen) |
| Hybrid, mean/max (enc 200) | 0.144 | 0.066 | 0.031 | 0.955 | 57.8 | 646 | 285 MB | 0.82M (+3.86M frozen) |
| Hybrid, slice (enc 100) | 0.139 | 0.067 | 0.033 | 0.953 | 59.4 | 640 | 223 MB | 4.77M (+3.86M frozen) |

Errors are relative L2 (pressure, velocity) and relative drag-coefficient error. Spearman ρ compares predicted and true
drag across the 100 cars. CPU timing used 10 cars. Final-epoch metrics are reported with no checkpoint selection.

Evidence: `logs/poc-suite-7246417.out` (job 7246417: V100, COMPLETED, `SUITE DONE. failed runs: none`).
The Transolver row reproduces my earlier inference results exactly (same checkpoint).

### Findings
- **Efficiency vs. accuracy.** The plain DeepONet is about 16x (distilled: about 20x) faster per car on GPU and about 19x faster on CPU, with about 43% less
  extra inference memory. Its errors are about 4x (pressure), 5x (speed) and 16x (drag) higher, and drag ranking falls from ρ = 0.992 to about 0.27.
- **No parameter savings.** 3.72M vs 3.86M parameters is only about 4% fewer, so the savings come from the architecture and input, not model size.
- **Distillation did not help.** The three DeepONet variants differ by about 1% in pressure error, which a single seed cannot resolve. The student also
  stays about 0.30 away from the teacher's own predictions. Hypothesis (untested): the bottleneck is the DeepONet's input representation or
  capacity, not the targets.
- **Overfitting on pressure.** In the plain DeepONet, validation pressure error rises (0.249 at epoch 10 to 0.307 at epoch 40) while training loss falls
  (0.568 to 0.211). Velocity error improves slightly over the same epochs.
- **Transolver's features carry most of the useful information.** A DeepONet head on frozen Transolver features reaches ρ ≈ 0.96 and pressure error ≈ 0.14,
  but still trails Transolver itself and costs the same at inference (it needs the full Transolver forward pass), so it is a diagnostic, not an efficiency gain.
- **Pooled features are nearly as good as slice tokens** (0.144 vs 0.137 pressure) with about 6x fewer trainable parameters, though inference memory is higher.
- **Encoder checkpoint age made little difference** (enc 100 vs enc 200: 0.139 vs 0.137 pressure).

### Limitations
- Single seed and a single split, so small differences between variants are not conclusive.
- [State whether Transolver's per-car time includes input feature computation. The one-time 1.5 h preprocessing is not included in any timing.]
- The DeepONet branch encodes the car from a surface point cloud, but the trunk's query features (SDF, normals) still come from the dataset.
- A smaller DeepONet (`--n_sensors 256`) was planned but not run.
- Next steps: early stopping or regularization, sensor-count and width sweeps, an order-invariant branch encoder, multiple seeds.

## Troubleshooting Log
- **Jobs 7246226 and 7246261 were allocated AMD GPUs, so PyTorch could not use a GPU.** Both logs show `gres/gpu:mi210`
  (an AMD MI210), and both jobs ended within a couple of minutes (2 min 17 s and 1 min 4 s). A plain GPU request on `ice-gpu` can land on an AMD
  node, and the CUDA build of PyTorch cannot use it. Fix: added `-C 'cuda12|cuda13'` to `sbatch`, as in PACE's documentation, which allocated a
  Tesla V100 (job 7246417) that completed all 7 runs. Always check for `cuda available: True` near the top of the log right after a job starts.
  Evidence: `logs/poc-suite-7246226.out`, `logs/poc-suite-7246261.out`, and the successful run in `logs/poc-suite-7246417.out`.

## Ming's Notes
- For execution evidence, see the `logs/` folder: job 7246417 holds the full results, and 7246226 and 7246261 document the GPU allocation failure.
- The Transolver weights are my own (100 and 200 epochs); no pretrained weights were available from the authors.

## License
MIT License.
This repository reuses code from thuml/Transolver, which is released under the license in the `LICENSE` file at the top level of this repo.
My additions (`hybrid_deeponet.py`, job scripts, logs, and documentation) are provided under the same terms.

## Credits
- Transolver paper: https://arxiv.org/abs/2402.02366
- DeepONet paper: https://doi.org/10.1038/s42256-021-00302-5
- Dataset: http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip (Umetani and Bickel, SIGGRAPH 2018)
- Source repos: https://github.com/Magical452/Transolver-Car
- Used Claude for README writing and troubleshooting.
- Citations:
```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
```
@article{lu2021learning,
  title   = {Learning nonlinear operators via {DeepONet} based on the universal approximation theorem of operators},
  author  = {Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  number  = {3},
  pages   = {218--229},
  year    = {2021}
}
```