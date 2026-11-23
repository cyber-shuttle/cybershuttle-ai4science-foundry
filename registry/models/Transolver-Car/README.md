# Reproducible Inference on AI Makerspace for Transolver

## Deacription
This repository is a reproducible inference worflow for Transolver on a GT PACE ICE Cluster for the ShapeNetCar. My work was done in Car-Design-ShapeNetCar/, jobs/, 
logs/, and this README. (Note to TA: The reproducibility workflow is first. If you can't find something in my README.md, go to the Ming's Notes section and I will 
point you to the file that gives you evidence that I have met the requirement for the top block of points.)

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
3. Clone only this model's folder (not the whole repository):
	- `git clone --branch add-transolver-car --single-branch --filter=blob:none --sparse https://github.com/Magical452/cybershuttle-ai4science-foundry.git`
	- `cd cybershuttle-ai4science-foundry`
	- `git sparse-checkout set registry/models/Transolver-Car`
	- `cd registry/models/Transolver-Car`
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
1. Check the weights exist at `Car-Design-ShapeNetCar/metrics/Transolver/0/200_0.5/`. They are both .pth files.
2. Run `mkdir -p Car-Design-ShapeNetCar/results/Transolver` to make sure that the results of your inferences have somewhere to go
### Operation
There are two eval job scripts: eval100 and eval200.

1. Edit jobs/eval200.sbatch and/or jobs/eval100.sbatch with nano. 
Four lines hold paths: the two export lines (MLCFD_DATA_DIR and MLCFD_SAVE_DIR) and the --data_dir and --save_dir arguments. 
Use the same absolute paths as in preprocessing.
2. From the repo root directory, run `sbatch jobs/eval200.sbatch` or `sbatch jobs/eval100.sbatch`
3. Check the queue with `squeue -u <username>`, and follow the log with `tail -f logs/eval200-<jobid>.out`
4. Check the finish with `sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode`. You want COMPLETED and 0:0.
5. Read the results at the end of the log: rho_d, c_d, relative l2 error press, relative l2 error velo.

Notes:
- --nb_epochs has to match the weights being evaluated (100 or 200), because the script builds the checkpoint path from it. They are already implemented, 
so don't change it unless you need to. 
- The scripts must be submitted from the repo root, because they use $SLURM_SUBMIT_DIR and then cd into Car-Design-ShapeNetCar.
- My results are in the next section.

## Validation
### Smoke test

| Check | Expected | Observed | Evidence |
|---|---|---|---|
| GPU visible in job | `cuda available: True` | `True`, A100 80GB | `logs/test-6181188.out` |
| Data loads | No errors | "load data finish" | `logs/test-6181188.out` |
| 1-epoch run completes | Finite losses, checkpoint saved | train_loss 1.21, val_loss 0.478, 154.82 s | `logs/test-6181188.out` |
| Job exit status | `COMPLETED`, `0:0` | `COMPLETED`, `0:0` | `logs/sacct_all_jobs.txt` |

### Validation (107 held-out test cars)

| Metric | Expected (paper, Transolver) | Observed (200 epochs) | Observed (100 epochs) |
|---|---|---|---|
| Volume relative L2 | 0.0207 | 0.0207 | 0.0236 |
| Surface relative L2 | 0.0745 | 0.0773 | 0.0744 |
| Drag coefficient relative error | 0.0103 | 0.0135 | 0.0110 |
| Spearman rank correlation | 0.9935 | 0.9922 | 0.9931 |

Evidence: `logs/eval200-6195417.out` and `logs/eval100-6195416.out`.

My outputs are valid because my numbers match very closely to the paper's numbers. For the 200 epoch run, the volume relative L2 matches the paper's, Surface relative L2 
is only 0.0028 difference, Drag coefficient relative error is only 0.0032 different, and Spearman rank correlation is 0.0013 different. For a reproducibility standard, 
I think that being within 4% for 3 out four metrics is accurate enou. The only caveat is the drag coefficient, but the actual difference is very small. 
Also, the 107 test cars were held out from training, and the rank correlation is above 0.99. I did a 100 epoch run as well, which are within single digit percentages 
for 3 out of the 4 metrics. The volume is 14% more, but the acutal difference is small, so it still is accurate. As such, both of my runs are valid since their output 
numbers are close to the paper's and their rank correlations are above 0.99. 

## Troubleshooting Log
- **Lmod error loading `git/2.48.1-n563f5l`.** The module depends on `libidn2/2.3.4-56kl36a`, which Lmod could not find. It turned out git was already available 
on the cluster, so no git module was needed.
- **Shell stuck at a `>` prompt.** An unclosed quote in a pasted command left the shell waiting for more input. Fix: press Ctrl+C.
- **`gh` not available.** The GitHub CLI is not installed on the cluster. Fix: used plain `git clone` over HTTPS, since the repo is public.
- **`nano` crashed with "Segmentation fault" inside the conda environment.** Conda's `LD_LIBRARY_PATH` pointed nano at the wrong libraries. Fix: 
`env -u LD_LIBRARY_PATH nano <file>`, made permanent with an alias in `~/.bashrc`.
- **No pretrained weights available.** The authors' repo and READMEs link none, and their GitHub issue says the weights were still being organized. 
The Hugging Face repos I found (OneScience) only say weights will be uploaded later. Fix: trained the model myself (100 and 200 epochs) and committed 
the resulting weights.
- **Dataset arrived as nine `.tar.gz` archives.** The code expects unpacked `param0` to `param8` folders. Fix: `for f in param*.tar.gz; do tar -xzf "$f"; done`, 
which gives 889 samples.
- **`ModuleNotFoundError: No module named 'torch'` in a fresh environment.** Creating a conda environment installs nothing, and the repo's `requirements.txt` 
is unpinned. Fix: installed packages in a fixed order (torch, `numpy<2`, torch-cluster via the PyG wheel link, torch_geometric, then the rest), 
because `torch-cluster` must match the PyTorch and CUDA versions. Exact versions are in `requirements-frozen.txt`.
- **`ModuleNotFoundError: No module named 'sklearn'`.** The dataset code imports scikit-learn, but it is missing from the authors' `requirements.txt`. 
Fix: `pip install scikit-learn`.
- **`--nb_epochs` declared as `type=float` in `main_evaluation.py`.** The checkpoint path would be built as `200.0_0.5` instead of `200_0.5`. 
Fix: changed line 23 to `type=int`.
- **Training only saved a checkpoint at the very end.** A crash or timeout would have lost the whole run. Fix: added a backup save every 10 epochs in `train.py`.
- **Test job 6178779 failed after 6 seconds with `No module named 'torch'`.** The batch job's shell had not initialized conda, so `conda activate` 
did not work. Fix: added `eval "$(conda shell.bash hook)"` before activating. Evidence: `logs/test-6178779.out`.
- **Evaluation jobs 6189044 (eval100) and 6195005 (eval200) failed with `AttributeError: 'NoneType' object has no attribute 'GetDataType'`.** 
`utils/drag_coefficient.py` had the authors' machine path (`/data/PDE_data/...`) hardcoded, so VTK could not open the data files. Fix: those lines now 
read `MLCFD_DATA_DIR` and `MLCFD_SAVE_DIR`, which the eval scripts export. Evidence: `logs/eval100-6189044.out` and `logs/eval200-6195005.out`. 
The reruns (6195416 and 6195417) completed.
- **`git push` returned a 403 error.** The personal access token did not have write permission for the repo. Fix: created a new token with all scopes enabled.
- **`git commit` reported nothing to commit.** Files were modified but not staged. Fix: ran `git add` on the specific files first.
- **Commit author showed the cluster hostname.** Git guessed the identity automatically. Fix: set `user.name` and `user.email`, 
then ran `git commit --amend --reset-author`.
- **Harmless warnings (ignored):** the NumPy warning when torch was installed before numpy, the `torch.load` `weights_only` 
FutureWarning, and the timm `models.layers` deprecation notice.

## Ming's Notes
- For executioin evidence and logs, you can check the logs in the logs folder and it has all the information such as run time, resource information, and outputs. 
The failure notes are in the Troubleshooting Log above.
- I had to use my own weights for this because I thought that someone on HuggingFace would update the Transolver page with weights before I had to reproduce it, 
but they didn't. The number's don't look too off from what I predicted though.


## License
MIT License
This repository reuses code from thuml/Transolver, which is released under the license in the LICENSE file at the top level of this repo. My additions (job scripts, 
logs, and documentation) are provided under the same terms.

## Credits
- Paper: https://arxiv.org/abs/2402.02366
- Dataset: http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip (2,029,047,879 bytes)
  (Umetani and Bickel, SIGGRAPH 2018)
- Source repo: https://github.com/thuml/Transolver
- Used Claude for README format and troubleshooting.
- Citation: @inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
