# HyenaDNA small-32k

This workflow runs HyenaDNA small-32k, a pretrained DNA model, on 8 short regions of the human genome. It checks that the model's next-base guesses beat shuffled DNA and random guessing.

A successful run ends with three `PASS` lines and takes about 2-3 minutes on a V100. The first run also downloads the model.

## What you need

* **A PACE ICE account** with access to a GPU partition. You give Slurm your account and partition when you submit the job (see [Run](#run)).
* **An NVIDIA GPU.** The job requests one with `-C nvidia-gpu`, because some GPU partitions (including `coc-gpu`) also have AMD nodes that this code cannot use.
* **Internet access** on the cluster. The first run downloads the model from Hugging Face.
* **A scratch directory.** On PACE ICE this is `~/scratch`. On another system, set `SCRATCH_DIR` to your scratch path.

## Setup

Do this once, on a login node.

1. Clone into scratch:

        cd "${SCRATCH_DIR:-$HOME/scratch}"
        git clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry
        cd cybershuttle-ai4science-foundry

2. Get the PR branch. Once PR #34 is merged, skip this step:

        git fetch origin pull/34/head:hyenadna-small-32k
        git checkout hyenadna-small-32k

3. Build the environment:

        cd registry/models/biology/hyenadna-small-32k
        bash setup_env.sh

**What `setup_env.sh` does**
* Loads the `miniforge` module (tested with `miniforge/24.3.0-0`).
* Builds a Python 3.10 env at `$SCRATCH_DIR/.conda/envs/hyenadna_env`.
* Installs the packages pinned in `requirements.txt`: torch 2.5.1+cu121, transformers 5.17.0, numpy 2.2.6, tokenizers 0.23.2, safetensors 0.8.0, and huggingface_hub 1.33.0.
* Puts the conda and pip caches on scratch too.
* Stops with a message if the scratch directory does not exist. `SCRATCH_DIR` defaults to `$HOME/scratch`.

To put the env somewhere else, set `SCRATCH_DIR` and `ENV_PREFIX` when you run `setup_env.sh`, then `export ENV_PREFIX=<path>` before `sbatch`.

## Run

Run these from the `registry/models/biology/hyenadna-small-32k` folder.

1. Set your Slurm account and a GPU partition you can use. To find them, run:

        sacctmgr show assoc user=$USER format=account
        sinfo -o "%P %G" | grep -i gpu | awk '{print $1}' | sort -u

   * The first command lists your account.
   * The second lists every GPU partition on the cluster, possibly including some your account is not allowed to use.
   * Your partition is usually the one that starts with your account name (account `coc` goes with partition `coc-gpu`).
   * To check a pair without running anything, use `sbatch -A <account> -p <partition> --test-only run_inference.sbatch`.
   * The author used account `coc` with partition `coc-gpu`, and this workflow was tested on `coc-gpu`.

   Then set them:

        export SBATCH_ACCOUNT=<account>
        export SBATCH_PARTITION=<partition>

2. Submit the job:

        mkdir -p logs
        sbatch run_inference.sbatch

   Instead of the `export` lines in step 1, you can pass them directly: `sbatch -A <account> -p <partition> run_inference.sbatch`.

3. Check on it. `sbatch` prints a job number and the job has finished when it no longer shows up here:

        squeue -u $USER

4. Read the end of the log, replacing JOBNUMBER with your job number:

        tail -n 20 logs/hyenadna-infer-JOBNUMBER.out

**What the job does:** it requests one NVIDIA GPU, 4 CPUs, 32 GB, and 15 minutes. It prints the node and `nvidia-smi`, runs `infer.py` (a quick test on `ACTGACTGACTGACTG`), then runs `validate.py`. The model downloads to `$HF_HOME` the first time (default `$SCRATCH_DIR/.hf_cache`).

If your cluster has no `nvidia-gpu` node feature, either replace it with your cluster's name for NVIDIA nodes (list features with `sinfo -o "%N %f"`) or, if every GPU there is NVIDIA, delete the `#SBATCH -C nvidia-gpu` line.

## Results

The `.out` file in `logs/` should end with three `PASS` lines. Each run writes to its own folder, `outputs/run_JOBNUMBER/`, with per-region numbers in `validation_results.json`. The files directly in `outputs/` are the author's reference results, for comparison.

What you should see:

* **Quick test:** logits come out as `(1, 17, 16)`, and A, C, G, T are token IDs 7, 8, 9, 10.
* **Real DNA:** 1.0944 nats per base on average, better than the shuffled copy in all 8 regions.
* **Shuffled copies:** 1.3909.
* **Random guessing, ln(4):** 1.3863.

**How to read this**
* The loss is how wrong the model's next-base guess is, counting only A, C, G, and T.
* Each region also gets a shuffled copy: same letters, random order (`random.Random(0)`).
* Real DNA must beat random guessing, beat shuffled on average, and beat shuffled in most regions.
* Shuffled copies score about the same as random guessing, so the model is using the order of the bases, not just letter counts.
* The check matters because a broken model load can still give the right output shape.

V100 and L40S gave the same token IDs and logits within 1.05e-5 (see `logs/v100_vs_l40s_comparison.txt`). The log files in `logs/` come from the author's runs, so they show the author's paths.

## Troubleshoot

* **No log appears:** `logs/` didn't exist yet. Slurm opens the output files before the script starts, so run `mkdir -p logs` first, from this folder.
* **"scratch directory does not exist":** set `SCRATCH_DIR`, for example `SCRATCH_DIR=/path/to/scratch bash setup_env.sh`.
* **Invalid account or no partition:** `SBATCH_ACCOUNT` and `SBATCH_PARTITION` are not set. Set them (or pass `-A` and `-p`) using the two commands in step 1 of [Run](#run).
* **"no Python environment at ...":** run `bash setup_env.sh` first, or `export ENV_PREFIX=<path>` if you built the env elsewhere.
* **`nvidia-smi: command not found`:** the job landed on an AMD GPU node. Keep the `#SBATCH -C nvidia-gpu` line in `run_inference.sbatch`.
* **`Disk quota exceeded`:** probably `~/.conda/pkgs` filling your 30 GB home. `conda clean --all` barely helps because the files are hardlinked into environments. Delete `~/.conda/pkgs`, keep caches on scratch like `setup_env.sh` does, and check with `quota -s`.
* **Job ran in the wrong environment:** you probably built it at a custom `ENV_PREFIX` and didn't export it before `sbatch`. The `Env:` line at the top of the `.out` file shows which one it used.

## Cleanup

    scancel JOBNUMBER
    SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
    rm -rf "$SCRATCH_DIR/cybershuttle-ai4science-foundry" \
           "$SCRATCH_DIR/.conda/envs/hyenadna_env" "$SCRATCH_DIR/.hf_cache" \
           "$SCRATCH_DIR/.conda_pkgs" "$SCRATCH_DIR/.pip_cache"

Change the paths if you cloned somewhere else or set `ENV_PREFIX` or `HF_HOME`.

## Sources

* Model: `LongSafari/hyenadna-small-32k-seqlen-hf` at commit `8fe770c78eb13fe33bf81501612faeddf4d6f331`. The weights are BSD-3-Clause, and loading needs `trust_remote_code=True`: https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf
* Paper: https://arxiv.org/abs/2306.15794
* Code (Apache-2.0, not used directly): https://github.com/HazyResearch/hyena-dna
* Data: 8 regions of 1,000 bp from hg38, pulled on 2026-10-08 from https://api.genome.ucsc.edu/getData/sequence with `fetch_hg38_regions.py`

The 8 regions were picked arbitrarily. The coordinates are in `fetch_hg38_regions.py` and the FASTA is committed in `data/`, so you don't need to rerun the fetch script.