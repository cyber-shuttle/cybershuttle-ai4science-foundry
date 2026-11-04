# HyenaDNA small-32k

This workflow runs HyenaDNA small-32k, a pretrained model that reads DNA, on 8 short stretches of the human genome and checks that its next-base guesses are better than shuffled DNA and plain random guessing. You do not need to know how the model works. A successful run ends with three `PASS` lines, and the job takes about 2 minutes on a V100 (the first run also downloads the model).

## What you need

* A PACE ICE account with access to a GPU partition. The defaults below use Slurm account `coc` and partition `coc-gpu`; change them if yours differ (see Run).
* An NVIDIA GPU. The job script asks for one with `-C nvidia-gpu`, because `coc-gpu` also has AMD nodes that this code cannot use.
* Internet access on the cluster. The first run downloads the model from Hugging Face.
* A scratch directory. On PACE ICE this is `~/scratch`. On another system, set `SCRATCH_DIR` to your scratch path (see Setup).

## Setup

Clone into scratch so the repo, logs, and outputs stay off your 30 GB home quota. On a login node:

    cd "${SCRATCH_DIR:-$HOME/scratch}"
    git clone https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry
    cd cybershuttle-ai4science-foundry
    git fetch origin pull/34/head:hyenadna-small-32k
    git checkout hyenadna-small-32k
    cd registry/models/biology/hyenadna-small-32k
    bash setup_env.sh

Once PR #34 is merged, the files are on `main`, so skip the `git fetch` and `git checkout` lines.

`setup_env.sh` loads the `miniforge` module (tested with `miniforge/24.3.0-0`) and builds a Python 3.10 env with torch 2.5.1+cu121, transformers 5.17.0, numpy 2.2.6, tokenizers 0.23.2, safetensors 0.8.0, and huggingface_hub 1.33.0. Everything needed is pinned in `requirements.txt`. The env lives in `$SCRATCH_DIR/.conda/envs/hyenadna_env` and `$SCRATCH_DIR` defaults to `$HOME/scratch`. The conda and pip caches go to scratch too, since the home quota is only 30 GB. The script stops with a message if the scratch directory does not exist.

If you want the env somewhere else, set `SCRATCH_DIR` and `ENV_PREFIX` when you run `setup_env.sh`, then `export ENV_PREFIX=<path>` before `sbatch`.

## Run

Run these from the `registry/models/biology/hyenadna-small-32k` folder:

    mkdir -p logs
    sbatch run_inference.sbatch
    squeue -u $USER

`sbatch` prints a job number. When the job no longer shows up in `squeue`, it has finished. Read the end of its log, using your job number:

    tail -n 20 logs/hyenadna-infer-JOBNUMBER.out

The job asks for account `coc`, partition `coc-gpu`, one NVIDIA GPU, 4 CPUs, 32 GB, and 15 minutes. It prints the node and `nvidia-smi`, runs `infer.py` (a quick test on `ACTGACTGACTGACTG`), then runs `validate.py`. If your account or partition is different, edit the `#SBATCH -A` and `-p` lines or run `sbatch -A <account> -p <partition> run_inference.sbatch`. If your cluster has no `nvidia-gpu` node feature, delete the `#SBATCH -C nvidia-gpu` line. The model downloads to `$HF_HOME` the first time (default `$SCRATCH_DIR/.hf_cache`).

## Results

The `.out` file in `logs/` should end with three `PASS` lines. Each run writes its results to its own folder, `outputs/run_JOBNUMBER/`, with the numbers for each region in `validation_results.json`. The files directly in `outputs/` are the author's reference results, so you can compare yours against them.

* quick test: logits come out as `(1, 17, 16)`, and A, C, G, T are token IDs 7, 8, 9, 10
* real DNA: 1.0944 nats per base on average, better than the shuffled copy in all 8 regions
* shuffled copies: 1.3908
* random guessing, ln(4): 1.3863

The loss is how wrong the model's guess for the next base is, only counting A, C, G, and T. Each region also gets a shuffled copy with the same letters in a random order (`random.Random(0)`). Real DNA has to beat random guessing (ln(4)), beat shuffled on average, and beat shuffled in most regions. The shuffled copies score about the same as random guessing, so the model is using the order of the bases, not just how many of each letter there are. The check exists because a broken model load can still give the right output shape, and the quick test wouldn't catch that.

V100 and L40S gave the same token IDs and logits within 1.05e-5 (see `logs/v100_vs_l40s_comparison.txt`). The log files in `logs/` were produced by the author's runs, so they show the author's paths.

## Problems

If `sbatch` doesn't make a log, `logs/` didn't exist yet. Slurm opens the output files before the script starts, so run `mkdir -p logs` first, from this folder.

If `setup_env.sh` says the scratch directory does not exist, set `SCRATCH_DIR` to your scratch path, for example `SCRATCH_DIR=/path/to/scratch bash setup_env.sh`.

If the job stops right away with "no Python environment at ...", run `bash setup_env.sh` first, or `export ENV_PREFIX=<path>` if you built the env somewhere else.

If the job fails with `nvidia-smi: command not found`, it landed on an AMD GPU node. Keep the `#SBATCH -C nvidia-gpu` line in `run_inference.sbatch`.

If you hit `Disk quota exceeded`, it's probably `~/.conda/pkgs` eating your 30 GB home. `conda clean --all` barely helps because the files are hardlinked into environments. Delete `~/.conda/pkgs`, keep the caches on scratch like `setup_env.sh` does, and check with `quota -s`.

If the job ran in the wrong environment, you probably built it at a custom `ENV_PREFIX` and didn't export it before `sbatch`. The `Env:` line at the top of the `.out` file shows which one it used.

## Cleanup

    scancel JOBNUMBER
    SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
    rm -rf "$SCRATCH_DIR/cybershuttle-ai4science-foundry" \
           "$SCRATCH_DIR/.conda/envs/hyenadna_env" "$SCRATCH_DIR/.hf_cache" \
           "$SCRATCH_DIR/.conda_pkgs" "$SCRATCH_DIR/.pip_cache"

Change the paths if you cloned somewhere else or set 'ENV_PREFIX' or 'HF_HOME'.

## Sources

* Model: `LongSafari/hyenadna-small-32k-seqlen-hf` at commit `8fe770c78eb13fe33bf81501612faeddf4d6f331`. The weights are BSD-3-Clause, and loading needs `trust_remote_code=True`: https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf
* Paper: https://arxiv.org/abs/2306.15794
* Code (Apache-2.0, not used directly): https://github.com/HazyResearch/hyena-dna
* Data: 8 regions of 1,000 bp from hg38, pulled on 2026-10-08 from https://api.genome.ucsc.edu/getData/sequence with `fetch_hg38_regions.py`

The 8 regions were picked arbitrarily and none were dropped. The coordinates are in `fetch_hg38_regions.py` and the FASTA is committed in `data/`, so you don't need to rerun the fetch script.