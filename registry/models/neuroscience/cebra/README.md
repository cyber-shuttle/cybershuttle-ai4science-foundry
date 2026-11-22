# CEBRA_Reproduce_Workflow
Reproduces the first experiment of the CEBRA demo **["Encoding of space, hippocampus (CA1)"](https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html)** (notebook cells 1–30) as a batch job on Georgia Tech's PACE ICE cluster (AI Makerspace).
---
Run everything on the **login node**. This is because we are running a job that is not interactive. Running **sbatch** inside a compute node will crash and not work

## Complete Workflow

### Step 1: Enter ICE-Cluster
Replace <username> with your username
```bash
ssh <username>@login-ice.pace.gatech.edu
```

### Step 2: Getting the workflow
Go into your scratch folder and clone the repo in order to obtain the code / workflow. Then enter its directory.
```bash
cd ~/scratch
git clone https://github.com/ZacharyPrager/CEBRA_Reproduce_Workflow.git reproduce_workflow
cd reproduce_workflow
```
### Step 3: Setting up your environment
Once finished, you will see `Setup Complete`. 
```bash
bash setup_env.sh
```
### Step 4: Running the job
Once the sbatch script is finished you will see `Submitted batch job <JOB-ID>`.
```bash
sbatch run_cebra.sbatch
```
### Step 5: Watching the job
Run this code to "watch" as the job completes. You will see the state as it changes and a short summary 
```bash
python3 watch_job.py <JOB-ID>
```
### Step 6: Job validation
You should see `SUMMARY: 7/7 CHECKS PASSED`
```bash
cat outputs/run_<JOB-ID>/validation.txt
```
### Step 7: Saving the job's resource stats 
Run after the job finishes. Saves run time, memory use and exit state to `outputs/run_<JOB-ID>/slurm_accounting.txt`.
```bash
bash collect_evidence.sh <JOB-ID>
```
## Visualization Outputs
Inside `outputs/run_<JOB-ID>/figures/` you will find all the interactive simulations and data visualized. The interactive ones are with the `.html` extension. You can view these using Georgia Tech's ICE Cluster OnDemand platform. The link is found here (https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133).

## Environment & Setup

### `setup_env.sh` does the following things:
1. Creates the directories `data/`,`env/`,`logs/`,`outputs/`. 
2. Loads the `conda` module and creates the `cebra` environment
3. Installs the `CEBRA`,`matplotlib`,`PyTorch` dependencies
4. Records the version dependencies in `env/pip_freeze.txt`,`env/environment_export.yml` and `env/versions.txt`.
5. Downloads the dataset and records its SHA-256 hash in `env/data_manifest.txt`.

### Packages Breakdown
1. Python version 3.11 installed with `setup_env.sh`
2. CEBRA version 0.6.1 (with datasets and integrations) installed with `requirements.txt`
3. PyTorch 2.14.0, CUDA 13.0 build (`+cu130`), installed with `setup_env.sh`. This build only runs on newer NVIDIA GPUs (A100, H100, L40S), so `run_cebra.sbatch` requests an H100.
4. matplotlib version >= 3.8 installed with `requirements.txt`

**See `env/pip_freeze.txt` to view every package's exact version**

## Model, Data & Artifact Traceability

### Model
CEBRA-Behavior, same settings as notebook cell 12 (`offset10-model`, 3-D output, 5,000 iterations, seed 0).
CEBRA has no pretrained weights for this demo, so the job trains the model, saves it, reloads it, and runs inference from the saved copy.
CEBRA documentation can be found [Here](https://cebra.ai/docs/)

### Data
`rat-hippocampus-single-achilles` from CEBRA's built-in datasets
Saved to `data/rat_hippocampus/achilles.jl`; its SHA-256 hash is in `env/data_manifest.txt`.
The first 80% is used for training and the last 20% is held out for testing.

### Artifact Traceability
Checkpoints are saved in `outputs/run_<JOB-ID>/checkpoints/`:
   1. `cebra_posdir3_split.pt`: the real model
   2. `cebra_posdir3_shuffled_split.pt`: the shuffled control
Each checkpoint's SHA-256 hash is recorded in `run_info.json`, and `validate.py` checks that the saved model matches it.
Each job log also prints the git commit of the code that ran (`git_commit=`).

### Code Version
CEBRA code: version 0.6.1 ([PyPI](https://pypi.org/project/cebra/0.6.1/)) · [GitHub](https://github.com/AdaptiveMotorControlLab/CEBRA) · [Website](https://cebra.ai/)

### Sources
1. CEBRA paper: Schneider, Lee & Mathis, *Nature* (2023). [Here](https://www.nature.com/articles/s41586-023-06031-6)
2. CEBRA code: [GitHub](https://github.com/AdaptiveMotorControlLab/CEBRA) [Website](https://cebra.ai/)
3. Original notebook demo: [Encoding of space, hippocampus (CA1)](https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html)

## Inference Workflow & Job Scripts
### Complete script breakdown
1. `setup_env.sh` - Creates the `cebra` environment. 
2. `requirements.txt` - Python packages
3. `download_data.py` - Downloads the dataset from CEBRA and prints SHA-256 hash
4. `run_cebra.sbatch` - The exact Slurm job. Requests resources, run the model and validate 
5. `common.sh` - Activates environment, sets job settings, records job data, and track GPU usage
6. `demo_hippocampus.py` - The entire demo. Creates interactive plots and uses CEBRA's demo (cells 1-30)
7. `validate.py` - 7 checks each showing the expected vs actual results from the model
8. `watch_job.py` - CLI UI/UX element which tells you when the job is finished and quick summary 
9. `collect_evidence.sh` - saves `sacct` stats after the job finishes

### Resource Requests
1. Number of compute nodes: 1
2. Time requested: 1 hour
3. CPU cores: 2  cores
4. GPUs: 1 h100 GPU
5. Memory: 16G

### Inputs 
No user-provided input is needed. The only input is the rat hippocampus dataset (`data/rat_hippocampus/achilles.jl`), which `setup_env.sh` downloads for you 

### Outputs 
All outputs are found: `outputs/run_<JOB-ID>/`
1. `figures/01_data_raster_and_position.png`: Neural activity and rat position
2. `figures/02_behavior_train / 03_behavior_validation (.png + .html)`: Real model's embedding on training data
3. `figures/04_shuffle_train / 05_shuffle_validation (.png + .html)`: Shuffled-label control
4. `checkpoints/*.pt, cebra_posdir.pt`: saved models
5. `embedding_train.npy, embedding_test.npy`: The inference output
6. `pred_test.npy, labels_test.npy`: Decoded vs. true position on held-out data
7. `metrics.json`: Goodness of fit and decoding results
8. `run_info.json`: Versions, GPU, job ID, timings, checkpoint hashes
9. `validation.txt`: The 7 check report
10. `gpu_<JOB-ID>.csv`: GPU usage over time
11. `slurm_accounting.txt`: Runtime and memory stats

## Execution Evidence & Logs
### Run overview
All evidence of the reproduced run is found in `outputs/run_6189853/`
1. Slurm job ID: 6189853
2. GPU : h100
3. Time: 01:08:711
4. Final: COMPLETED exit code 0.00
5. Validation 7/7 checks successful and passed

### Evidence files breakdown
1. `cebra-hippocampus-6189853.out`: Job info and metrics
2. `cebra-hippocampus-6189853.err`: Training progress and warnings imported from CEBRA's demo page
3. `gpu_6189853.csv`: GPU usage and memory usage tracked
4. `slurm_accounting.txt`: `sacct` the runtime, memory and exit code
5. `validation.txt`: This is the output of `validate.py`, shows the expected vs observed results
6. `run_info.json`: Additional run information like software versions, GPU, model settings, etc
7. `metrics.json`: Goodness of fit and decoding errors
8. `checkpoints\`: Holds the saved trained models. You can load models again without having to train them
**Figures are the visual output (html files are for interactive plots and png's are pictures of the data)**

### Failures 
This project had a lot of debugging and troubleshooting, and I spent a long time fixing these errors. 
1. Moving from jupyter notebooks to python scripts. I always have issues running jupyter notebooks, so I had to pivot to a python script which functions the same way. Also, this meant creating functions that save the outputs and data like the visualization charts.
2. Originally, the `run_cebra.sbatch` script didn't specify a GPU. This caused the model to run on the CPU which led to 10+ minute runtimes. I changed the script to include a specific GPU which reduced runtime to about 1 minute. Using the logs I was able to spot this issue with the `CUDA=FALSE` flag.
3. In order to see if the job was in queue or running, I had to constantly spam `squeue`. Therefore, I made a python script which checks this for me.

## Validation of Results

### Validation methods
`validate.py` functions as the main validator of results and runs at the end of every job. In this script there are 7 tests which do the following: 
1. Checks that all the specified output files exist. If any don't exist then this test will fail
2. Using embeddings (the models output which are 3D points), the check can see if the shape matches the expected shape and if the values are NaN. The expected sizes come from `run_info.json`.
3. Finds the SHA-246 hash of the model and compares it to the `demo_hippocampus.py` hash in the `run_info.json`. SHA-256 is good for this task as if a small piece of information changes, there will be a drastic change. Thus, using this we can quickly see if the model in results is the same model.
4. Effectivity running embeddings again and tracking their difference. If the difference is too large (> .0001) then that indicates an error. Thus, we can compare the embeddings directly
5. Uses these embeddings to predict the rats position on the data. The position must have an error less than 0.2m
6. This shuffles the behavior labels. If the shuffled model was similar to the normal model, the model would not be working (it would be finding fake patterns). The real model must do much better than the shuffled one. The higher number the better for testing.
7. InfoNCE is CEBRA's training los function. This is a good way of validating the data as it comes directly from CEBRA. This value should be 5.53 for the real model and 6.16 for the shuffled.

### Smoke test
Our smoke test will use the specifier `ITERS` in `run_cebra.sbatch` to run less iterations of training (from 5000 to 200).
```bash
sbatch --export=ALL,ITERS=200 run_cebra.sbatch
```
Job ID: 6198990
```text
[PASS] output files present
        expected: all of cebra_posdir.pt, embedding_train.npy, embedding_test.npy, pred_test.npy, labels_test.npy, metrics.json, run_info.json
        observed: missing: none
[PASS] embedding shapes and values
        expected: train (8142, 3), test (2036, 3), all finite
        observed: train (8142, 3), test (2036, 3), finite=True
[PASS] checkpoint SHA-256 matches run_info.json
        expected: 1ee5004427a803b6...
        observed: 1ee5004427a803b6...
[PASS] re-running inference from checkpoint reproduces saved embedding
        expected: max |diff| < 1e-4
        observed: max |diff| = 0.00e+00
[PASS] position decoding median error
        expected: < 0.2 m
        observed: 0.1103 m (R2=0.735)
[PASS] real model beats shuffled-label control
        expected: real err < 0.5 x control err, and real R2 > control R2
        observed: real err 0.1103 m vs control 0.4383 m; R2 0.735 vs 0.003
[PASS] InfoNCE goodness of fit: real < shuffled
        expected: real < shuffled (notebook: ~5.53 vs ~6.16)
        observed: real 5.640757657051086 vs shuffled 6.1955155210495
```
Based on the `validation.txt` all tests were passed and the actual values are shown above.

## Documentation Notes and Reproducibility

### Cleanup
```bash
scancel <JOB-ID>
conda env remove -n cebra
rm -rf data/ outputs/ logs/
```
### Sources and AI usage
1. Used the Georgia Tech PACE ICE cluster documentation for the available GPUs, Slurm settings, and OnDemand.
2. Used the [Slurm documentation](https://slurm.schedmd.com/quickstart.html) to learn how `sbatch`, `squeue`, and `sacct` work.
3. Used the [CEBRA documentation](https://cebra.ai/docs/) and the [hippocampus demo notebook](https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html) as the base for `demo_hippocampus.py` and the expected results.
4. AI wrote much of the code for the scripts and helped debug errors. However, I made many of the design decisions and still worked through / troubleshot the scripts myself. I'm pointing this out as the syllabus allows AI usage when properly cited. Notable uses in this assignment are: 
     1. AI helped rewrite the `demo_hippocampus` and the functions that save the interactive graphs
     2. b. AI helped me design the 7 step validation test
     3.  AI helped me structure how each script calls the others. For example, `run_cebra.sbatch` loads `common.sh` (environment + GPU logging), runs `demo_hippocampus.py`, then runs `validate.py` on its outputs.
