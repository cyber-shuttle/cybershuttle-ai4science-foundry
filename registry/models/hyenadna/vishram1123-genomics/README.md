# HyenaDNA — reproducible inference and training on PACE ICE

This README covers three things for [HyenaDNA](https://github.com/HazyResearch/hyena-dna)
on Georgia Tech's PACE ICE cluster (`ice-gpu` partition / AI Makerspace):
loading a pretrained checkpoint to extract DNA sequence embeddings
([Inference](#inference)), training a small model from scratch
([Training](#training)), and a cross-model research proof of concept
([Research proof of concept: HyenaDNA consuming DNABERT-2 embeddings](#research-proof-of-concept-hyenadna-consuming-dnabert-2-embeddings)).
Everything here was run and verified on that cluster, not assumed from the
upstream docs — see [Troubleshooting](#troubleshooting) for the gaps found
along the way.

Model metadata: [`../model.yaml`](../model.yaml) · Recipes:
[`../../../../recipes/hyenadna-inference-embeddings.yaml`](../../../../recipes/hyenadna-inference-embeddings.yaml),
[`../../../../recipes/hyenadna-training-quickstart.yaml`](../../../../recipes/hyenadna-training-quickstart.yaml),
[`../../../../recipes/hyenadna-dnabert2-bridge.yaml`](../../../../recipes/hyenadna-dnabert2-bridge.yaml) ·
Resource profile: [`../../../../resource-profiles/pace-ice/ice-gpu.yaml`](../../../../resource-profiles/pace-ice/ice-gpu.yaml)

## Inference

### Setup

```bash
ssh login-ice.pace.gatech.edu
srun --account=<your-account> --partition=ice-gpu --gres=gpu:1 \
     --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

source scripts/hyenadna_setup_env.sh
```

This clones `HazyResearch/hyena-dna` to `~/scratch/hyena-dna`, creates a
conda env at `~/scratch/.conda/envs/hyena-dna` (Python 3.8), and installs a
PyTorch build matched to whatever GPU Slurm actually allocated — see
[Hardware and provenance](#hardware-and-provenance) for why this matters.
Run it inside a GPU allocation, not on the login node: picking the right
PyTorch build depends on `nvidia-smi` seeing a real GPU.

It must be `source`d, not run as `bash scripts/hyenadna_setup_env.sh`: the
`conda activate` and cache-redirect exports it performs only take effect in
the shell that runs them, and are lost immediately if it executes in its own
subshell.

By default this skips `flash-attn` — it's a slow from-source build, and
standalone inference doesn't need it (see [Training](#training) for the one
code path that does). Set `HYENADNA_INSTALL_FLASH_ATTN=1` before sourcing
the script if you want it anyway.

### Execution

```bash
# smoke test + validation check (downloads the tiny-1k checkpoint on first run)
python scripts/hyenadna_validate.py --model hyenadna-tiny-1k-seqlen

# embed a sequence directly
python scripts/hyenadna_inference.py --model hyenadna-tiny-1k-seqlen \
    --sequence ACTGACTGACTGACTG

# or a FASTA file, saving the result
python scripts/hyenadna_inference.py --model hyenadna-tiny-1k-seqlen \
    --fasta data/example.fasta --output results/embeddings.pt
```

As a batch job:

```bash
sbatch scripts/hyenadna_inference.sbatch data/example.fasta
```

`scripts/hyenadna_inference.sbatch` builds the environment if it isn't
already there, runs the validation check, then the inference call — it is
the canonical, idempotent version of the steps above.

### Verification

`scripts/hyenadna_validate.py` is the validation check, run against the
`hyenadna-tiny-1k-seqlen` checkpoint on an ICE V100:

```
[smoke test] shape=(1, 202, 128) nan=False inf=False
[determinism] L2 distance across repeated runs of the same sequence: 0.00e+00
[validation] L2 distance between two unrelated sequences: 56.35 (embedding norm scale: 138.15)

VALIDATION PASSED
```

Four properties are checked, each catching a different failure mode a
shape-only smoke test would miss:

1. **Shape and finiteness** — output tensor shape matches `config["d_model"]`
   and the input length; no NaN/Inf (catches a broken forward pass).
2. **Determinism** — the same sequence embedded twice gives bit-identical
   output in eval mode (catches, e.g., an accidentally-active dropout layer).
3. **Input sensitivity** — two unrelated random sequences embed to
   measurably different vectors, by raw L2 distance on the un-pooled
   per-position embeddings (catches a checkpoint that loaded with the wrong
   keys, or silently fell back to random init, and so produces
   shape-correct but content-blind output).

4. **Pretraining objective** — the model predicts the next nucleotide, so a
   real human mRNA (NCBI `NM_000518.5`, HBB, at `data/example.fasta`) should get
   a higher mean next-nucleotide log-likelihood than a shuffle of the same
   nucleotides. The logits come from the tied embedding matrix, as in upstream
   training. A missing gap points at a loading or tokenization bug. On an
   H100 with the tiny-1k checkpoint, perplexity was 3.825 for the real
   sequence against 4.212 for the shuffle.

Cosine similarity on mean-pooled embeddings was tried first for check 3 and
rejected: it stayed above 0.99 even between two unrelated random sequences
for this checkpoint, so it couldn't actually discriminate anything. Raw
per-position L2 distance could.

## Training

Trains a small (2-layer, `d_model=128`) HyenaDNA model from scratch on one
[GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
classification task — the upstream repo's own documented "quick entry
point" — rather than full hg38 pretraining. That's a deliberate scope
choice, not a shortcut forced by a limitation: see
[`recipes/hyenadna-training-quickstart.yaml`](../../../../recipes/hyenadna-training-quickstart.yaml)'s
notes for why the quick-start exercises the same training code path as
full pretraining, just at a scale that fits a single GPU in minutes.

> **Status:** verified. `scripts/hyenadna_train.sbatch` has been re-run
> end to end as committed via a real `sbatch` submission on PACE ICE (see
> [Verification](#verification-1) below), including
> `scripts/hyenadna_train_validate.py` running automatically at the end of
> that same job and printing `VALIDATION PASSED`.

### Setup

Same as [Inference setup](#setup), except flash-attn is required, not
optional: `src/models/sequence/dna_embedding.py` — the module the
Lightning training path always loads — hard-imports `GenerationMixin` from
`flash_attn`, with no pure-PyTorch fallback.

```bash
HYENADNA_INSTALL_FLASH_ATTN=1 source scripts/hyenadna_setup_env.sh
```

If an inference-only environment already exists (so the setup script would
be skipped), `scripts/hyenadna_train.sbatch` checks for `flash_attn`
separately and installs it on top of the existing env rather than failing.

### Execution

```bash
python -m train wandb=null experiment=hg38/genomic_benchmark_scratch \
    model.fused_dropout_add_ln=false model.fused_mlp=false \
    hydra.run.dir=~/scratch/hyena-dna/hyena_train_runs/my_run \
    2>&1 | tee ~/scratch/hyena-dna/hyena_train_runs/my_run/train.log
```

Run from inside the `hyena-dna` checkout (`~/scratch/hyena-dna`) — Hydra
resolves `experiment=...` relative to `configs/experiment/`. `wandb=null`
avoids needing a wandb account; the tradeoff is that Lightning then
persists no metrics file at all, so the log is the only record of how
training went (see [Verification](#verification-1)).

As a batch job:

```bash
sbatch scripts/hyenadna_train.sbatch
# or override which GenomicBenchmarks task / output dir:
sbatch scripts/hyenadna_train.sbatch hg38/genomic_benchmark_scratch ~/scratch/hyena-dna/hyena_train_runs/my_run
```

`scripts/hyenadna_train.sbatch` builds the environment if needed (with
flash-attn), pins `hydra.run.dir` to a predictable path instead of Hydra's
default timestamp, captures the log, and runs the validator automatically.

### Verification {#verification-1}

Re-run end to end as committed via `sbatch scripts/hyenadna_train.sbatch`
on PACE ICE (Tesla V100), with `hyenadna_train_validate.py` running
automatically at the end of the same job:

```
Epoch 99: 100%|██████████| 10/10 [00:02<00:00,  3.55it/s, loss=0.0353, val/accuracy=0.711, val/loss=1.760, train/accuracy=0.992, train/loss=0.0323]
`Trainer.fit` stopped: `max_epochs=100` reached.
[smoke test] checkpoint: .../checkpoints/last.ckpt (62 tensors, epoch=99, global_step=800)
[validation] loss: first=0.7070 last=0.0323 (n=2180 logged steps)
[validation] final val/accuracy: 0.7110

VALIDATION PASSED
```

100 epochs completed in ~6 minutes; `train/loss` fell from 0.707 to 0.032
and `train/accuracy` reached 0.992, with `val/accuracy` at 0.711 — all
within the validator's checked ranges.

Getting this clean run surfaced three real bugs in the committed scripts
(each hit live, each now fixed and covered in
[Troubleshooting](#troubleshooting)): an Lmod module-swap crash under
`set -u`, a `matplotlib`/home-quota crash, and a SLURM-spool-directory path
bug in the training script's sibling-script lookup. An earlier manual run
(before these scripts existed) hit 100 epochs on an H200 in under 2
minutes with `train/accuracy` 0.994 — consistent with this run, modulo GPU
and minor run-to-run variance.
`scripts/hyenadna_train_validate.py` formalizes two checks against a
captured log and checkpoint directory:

1. **Checkpoint integrity** — a `.ckpt` file exists under the run's
   `checkpoints/` directory and its state dict is non-empty with no
   NaN/Inf values (catches a run that crashed before checkpointing, or
   wrote a corrupt one).
2. **Learning actually happened** — the last logged training loss is
   lower than the first, and the final `val/accuracy` is a finite value in
   `[0, 1]` (catches a run that executes every step without the model
   learning anything).

Loss/accuracy are parsed from the captured stdout log via regex rather
than a metrics file, because `wandb=null` means Lightning persists none —
see [Execution](#execution-1).

## Hardware and provenance

- **Verified hardware:** PACE ICE `ice-gpu` partition. Inference confirmed
  on a Tesla V100-16GB (compute capability 7.0); from-scratch training
  confirmed on both a V100 (the packaged `sbatch` job, see
  [Verification](#verification-1)) and an H100/H200-class GPU (compute
  capability 9.0, an earlier manual run). The setup script handles both
  compute-capability ranges — see
  [GPU architecture mismatch](#troubleshooting) below.
- **Checkpoint:** `LongSafari/hyenadna-tiny-1k-seqlen` from HuggingFace
  (BSD-3-Clause), downloaded via `git clone` (git-lfs) in
  `scripts/hyenadna_inference.py`.
- **Training data:** GenomicBenchmarks' `dummy_mouse_enhancers_ensembl`
  task (binary classification, 1210 sequences), auto-downloaded by the
  upstream repo's own dataset code — no manual download step.
- **Seeds:** the inference validation check uses fixed seeds (1, 2) for its
  two comparison sequences; inference on real input has no stochastic step
  (`model.eval()`, `torch.inference_mode()`). Training uses the experiment
  config's fixed seed (`train.seed: 2222` in
  `configs/experiment/hg38/genomic_benchmark_scratch.yaml`).
- **Software:** Python 3.8; PyTorch 1.13.0+cu117 or 2.1.2+cu118 depending on
  allocated GPU (see `scripts/hyenadna_setup_env.sh`); other pinned versions
  come from the upstream repo's `requirements.txt`.

## Troubleshooting

Real gaps found while getting this to run, each worked around in the
scripts rather than just noted here:

**GPU architecture mismatch.** HyenaDNA's documented environment pins
`torch==1.13.0+cu117`, whose compiled kernels only cover compute capability
up to 8.6 (Ampere). PACE ICE's `ice-gpu` partition can also allocate H100 or
H200 nodes (compute capability 9.0), and on one of those, that pinned build
imports fine but hangs/spins at near-100% CPU on the first real GPU op,
trying to JIT a kernel for an architecture it doesn't ship. Fix:
`scripts/hyenadna_setup_env.sh` checks `nvidia-smi --query-gpu=compute_cap`
and installs `torch==2.1.2+cu118` instead when the GPU is Hopper or newer.

**`pip`'s resolver silently downgrades torch.** `requirements.txt` pulls in
`torchtext==0.14.0`, which hard-pins `torch==1.13.0`. Installing
`requirements.txt` after a GPU-enabled torch is already present downgrades
it back to a CPU-only-equivalent build — `pip` prints a dependency-conflict
warning but does not fail or stop. Fix: the setup script reinstalls the
correct GPU build of torch *after* `requirements.txt`, not before.

**Home directory quota breaks installs and downloads.** PACE ICE's home
quota (~30GB) is consumed by `pip`'s wheel cache, HuggingFace's hub cache,
and `git`'s global config file (written by `git lfs install` on every
checkpoint clone) well before scratch storage fills up, and each one fails
differently: `pip install` errors with `Disk quota exceeded`; `git lfs`
prints a non-fatal warning and continues (but the clone can still fail
depending on what else is pending); HuggingFace prints a cache-write
warning. Fix: `PIP_CACHE_DIR`, `TMPDIR`, `HF_HOME`, and `GIT_CONFIG_GLOBAL`
are all redirected to a scratch path before any install or download step,
in both sbatch jobs.

**Importing `huggingface.py` runs an inference example as a side effect.**
The upstream repo's documented pattern, `from huggingface import
HyenaDNAPreTrainedModel`, triggers a full hardcoded inference run at import
time — the module calls `inference_single()` unconditionally, with no `if
__name__ == "__main__":` guard. `scripts/hyenadna_inference.py` reimplements
the small amount of checkpoint-loading logic directly against
`standalone_hyenadna.py` instead of importing the upstream module. While
reimplementing it, a second bug surfaced: the upstream checkpoint-loading
code reads weights from `ckpt['state_dict']` — the raw `torch.load()` result
is a full PyTorch Lightning checkpoint dict (`epoch`, `global_step`,
`state_dict`, `optimizer_states`, ...), not a bare state dict — easy to miss
on a first reimplementation (it was missed here on the first attempt, and
caught by a `KeyError` rather than a silent wrong-weights load).

**`flash-attn` is a hard dependency for training, with no fallback.**
Unlike the standalone inference path, `src/models/sequence/dna_embedding.py`
— loaded by every Lightning training run — unconditionally imports from
`flash_attn`. There's no config flag to avoid it; it must be installed
(`HYENADNA_INSTALL_FLASH_ATTN=1`, a from-source compile) before training
will even import successfully.

**Two fused-kernel config flags default to unbuildable values.** Even with
flash-attn installed, the `genomic_benchmark_scratch` experiment config
defaults `model.fused_dropout_add_ln` and `model.fused_mlp` to settings
that require `dropout_add_layer_norm`, an optional fused CUDA kernel from
flash-attn's `csrc/layer_norm` subdirectory that this setup does not build
(it's a speed optimization, not a correctness requirement). Left at their
defaults, `train.py` fails immediately with `dropout_add_layer_norm is not
installed`, before a single batch runs. Fix: pass
`model.fused_dropout_add_ln=false model.fused_mlp=false` on the command
line, as `scripts/hyenadna_train.sbatch` does.

**`wandb=null` means no metrics file, not just no wandb upload.** Avoiding
a wandb account requirement by passing `wandb=null` also means Lightning
is configured with no logger at all for this pipeline — not "logs locally
instead of to wandb." The only record of training progress is whatever
reaches stdout, which is why `scripts/hyenadna_train.sbatch` captures it to
a log file and `scripts/hyenadna_train_validate.py` parses that file rather
than a structured metrics format.

**`module load miniforge` can crash a fresh job under `set -u`.** A fresh
ICE job or shell starts with `anaconda3` loaded by default. Loading
`miniforge` on top of that triggers Lmod's automatic swap-out of
`anaconda3`, whose deactivation hook references an unset shell variable —
fatal under `set -u`, and with no error message of its own: the job just
exits in seconds, before reaching anything else in the script. Fix:
`module purge` before `module load miniforge`, in
`scripts/hyenadna_setup_env.sh` and both sbatch scripts.

**`matplotlib`'s font cache hits the home quota.** `pytorch_lightning`
pulls in `torchmetrics`, which imports `matplotlib`, which writes a font
cache to `$HOME` on first import — the same quota that `pip`'s wheel cache
and HuggingFace's hub cache already needed redirecting off of (see above).
Training fails with `OSError: [Errno 122] Disk quota exceeded` partway
through importing `pytorch_lightning`, before any training code runs.
Fix: `scripts/hyenadna_train.sbatch` also redirects `MPLCONFIGDIR` (and
`HF_HOME`, for the GenomicBenchmarks dataset download) to scratch.

**A separate conda env's own `conda create` can hit the `$HOME` quota,
even with every cache already redirected to scratch.** Setting up the
`dna-bridge` env (for the DNABERT-2/HyenaDNA bridge, below) failed with the
same `OSError: [Errno 122] Disk quota exceeded` as the matplotlib/pip/HF
cache bugs above, despite `HF_HOME`, `PIP_CACHE_DIR`, and `TMPDIR` all
already pointing at scratch — because conda's own package cache
(`CONDA_PKGS_DIRS`) defaults to `~/.conda/pkgs` regardless of those, and
leftover partial downloads from earlier attempts had already filled it to
7.4GB inside a 30GB home quota. Fix: `export CONDA_PKGS_DIRS=~/scratch/.conda/pkgs`
before any `conda create`, and periodically check `du -sh ~/.conda` — this
cache does not clean itself up.

**A conda env is not isolated from `~/.local`'s user-site packages.**
Running the bridge script against a freshly-created, minimal `dna-bridge`
env (deliberately without `triton` installed) still pulled in a `triton`
whose API (`tl.dot(..., trans_b=...)`) didn't match DNABERT-2's bundled
Triton kernel, because Python by default also searches
`~/.local/lib/python3.X/site-packages` — a leftover from an unrelated,
non-conda `pip install --user` done at some earlier point — ahead of
nothing distinguishing it from the conda env's own packages. The same
leak then silently satisfied `pip install`'s dependency checks for
`huggingface_hub` and `regex` (pip saw them "already installed" in
user-site and skipped them), so each got discovered missing one at a time,
in the wrong order, once the leak was plugged. Fix:
`export PYTHONNOUSERSITE=1` before running anything in a new env meant to
be reproducible in isolation, and install every dependency explicitly
rather than relying on "it's probably already there."

**Don't assume a submodule exists without checking the source.** The
bridge script's first version added `backbone.embeddings.position_embeddings(position_ids)`
on top of the projected DNABERT-2 features, by analogy to a typical
transformer. HyenaDNA has no such submodule — `standalone_hyenadna.py`'s
`GPT2Embeddings.__init__` only creates `self.position_embeddings` when
`max_position_embeddings > 0`, and HyenaDNA's config sets it to `0` (Hyena
gets position information implicitly from its convolution filters).
`AttributeError: 'GPT2Embeddings' object has no attribute
'position_embeddings'` caught this immediately on the first live run.

**`${BASH_SOURCE[0]}` doesn't point where you'd expect under `sbatch`.**
SLURM copies a submitted batch script into a per-job spool directory on
the compute node and runs it from there, so deriving a script's own
directory from `${BASH_SOURCE[0]}` resolves to that spool directory, not
to wherever the script actually lives — breaking any lookup of sibling
files (here, `hyenadna_train_validate.py`). Fix: both sbatch scripts use
`$SLURM_SUBMIT_DIR` (the directory `sbatch` was invoked from) instead,
falling back to the `${BASH_SOURCE[0]}` form only for direct, non-`sbatch`
invocation.

## Research proof of concept: HyenaDNA consuming DNABERT-2 embeddings

Assignment 5's research proof of concept. The pattern asked for is the one
the professor described in class: one model trained or run on a *different*
model's output representations (e.g. BENDR trained on CEBRA's latents),
rather than its own native input encoding. Applied here to two DNA
foundation models already in this Foundry: [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M)
(117M params, BPE tokenization + ALiBi, verified reproducible on PACE ICE in
PR #43 of this course's Foundry) and HyenaDNA.

**Why DNABERT-2 and not Evo 2** (the model originally planned on the Future
Direction slide): Evo 2's smallest released checkpoint (1B params,
StripedHyena2) needs `transformer-engine` and GPU/kernel support that
could not be reliably stood up in the time this assignment allows. DNABERT-2
is a verified, already-reproducible Foundry recipe that runs on a single
GPU in under a minute, which made it possible to actually execute this live
end to end instead of only describing it on paper. The extraction →
resample → project → feed-through pattern implemented here is the same
pattern an Evo 2 version would need — only the source checkpoint id and
hidden size would change (see Next Steps).

### What it does

`scripts/dnabert2_hyenadna_bridge.py`:

1. Runs DNABERT-2 on the input sequence and keeps its **per-token hidden
   states** (`[1, L_bpe, 768]`), not just the mean-pooled embedding PR #43
   validated.
2. **Resamples** that token sequence to HyenaDNA's per-nucleotide sequence
   length via linear interpolation (`resample_to_length`). This is the core
   technical mismatch between the two models: HyenaDNA operates at
   single-nucleotide resolution (one embedding per A/C/G/T/N), while
   DNABERT-2's byte-pair tokenizer produces a *variable* number of
   nucleotides per token. Resampling treats the token embedding sequence as
   a continuous signal and is a coarse approximation — it assumes tokens
   are evenly spaced across the sequence, which they are not. Exact
   alignment would need the tokenizer's character-to-token offsets, which
   this tokenizer (loaded via `trust_remote_code=True`, so it is not
   guaranteed to implement the fast-tokenizer `offset_mapping` API) does
   not expose.
3. **Projects** the resampled 768-dim DNABERT-2 features down to HyenaDNA's
   128-dim embedding space with a trainable `nn.Linear` (randomly
   initialized here; a real downstream task would train it jointly with a
   task head — see Next Steps).
4. Feeds the result through HyenaDNA's own Hyena-operator layers
   (`backbone.layers`, `drop_f`, `ln_f`), **bypassing HyenaDNA's native
   `nn.Embedding(vocab_size=12, ...)` lookup entirely.** No position
   embedding needs adding back: HyenaDNA is built with
   `max_position_embeddings=0` — it gets positional information implicitly
   from the Hyena convolution filters, not a learned position table, so
   `standalone_hyenadna.py`'s `GPT2Embeddings` never even instantiates a
   `position_embeddings` submodule for this model. (Confirmed live: this
   was the first thing tried, against the actual attribute, and it does
   not exist — see Troubleshooting.)

### Running it

```bash
# one-time: a separate conda env from hyena-dna's, see Troubleshooting for why
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p ~/scratch/.conda/envs/dna-bridge python=3.10
conda activate ~/scratch/.conda/envs/dna-bridge
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.29.2 tokenizers==0.13.3 huggingface_hub==0.16.4 \
    einops==0.6.1 numpy==1.24.4 omegaconf scikit-learn scipy regex

export PYTHONNOUSERSITE=1   # see Troubleshooting -- do not skip this
export HYENADNA_REPO=~/scratch/hyena-dna
export HF_HOME=~/scratch/.cache/huggingface
python scripts/dnabert2_hyenadna_bridge.py --fasta data/example.fasta \
    --output results/bridge-embedding.pt
```

### Execution evidence (real, live PACE-ICE run)

Job output: [`logs/dnabert2-hyenadna-bridge-7248423.out`](logs/dnabert2-hyenadna-bridge-7248423.out).
Saved tensors (bridged + native embeddings for comparison):
[`results/bridge-embedding.pt`](results/bridge-embedding.pt).

```
input length: 628 nt
[bridge] DNABERT-2 tokens: 127  ->  resampled to HyenaDNA length: 630
[bridge] output shape: (1, 630, 128)  nan=False  inf=False
[determinism] L2 distance across repeated runs: 0.00e+00
[sensitivity] L2 distance, real vs shuffled sequence: 152.23
[comparison] L2 distance, DNABERT-2-bridged vs HyenaDNA-native embedding
  (same sequence): 163.57  (native norm scale: 243.00, bridged norm scale: 247.68)
BRIDGE VALIDATION PASSED
```

Reading this: the bridged pathway is deterministic (identical repeated
runs), responds to the actual sequence content (a shuffled sequence
produces a very different output — L2 152.23 vs. 0 for determinism), and
produces finite, correctly-shaped output. The comparison line shows the
bridged (DNABERT-2-conditioned) embedding and HyenaDNA's own native
embedding of the *same* sequence are at a comparable norm scale
(247.68 vs. 243.00) but represent genuinely different points in embedding
space (L2 distance 163.57, on the same order as the sensitivity check) —
i.e. HyenaDNA's layers do something non-degenerate with a foreign model's
representations; they don't just pass them through or collapse them.

### Observations

- **Getting the environment right took far more iteration than the model
  code itself.** Six consecutive job failures before the first clean run,
  none of them in the actual bridging logic (see Troubleshooting) — every
  one was a reproducibility/shared-infrastructure problem: a stray
  `~/.local` user-site install leaking into a supposedly clean conda
  environment, a conda package-cache default pointed at a nearly-full
  `$HOME` quota, and one real design bug (assuming a position-embedding
  submodule existed without checking). This is itself direct, lived
  evidence for the Future Direction slide's "make inference/training
  GPU-architecture-agnostic, not special-cased" and general
  shared-infrastructure goals — two independently-reproducible single-model
  recipes (HyenaDNA's and DNABERT-2's) did not combine cleanly on the first
  several attempts even though each works perfectly alone.
- **The resampling step is a real approximation, not a detail.** Treating
  127 BPE tokens as evenly spaced across 628 nucleotides is almost
  certainly wrong at the level of individual tokens (DNABERT-2's BPE
  vocabulary has variable-length tokens), so any signal this pathway
  carries is at the resolution of "roughly this region of the sequence,"
  not single-nucleotide precision. HyenaDNA's own single-nucleotide
  resolution is arguably its main selling point relative to other genomic
  models, so this bridge sacrifices some of exactly what makes HyenaDNA
  useful as the receiving model.
- **No labeled downstream task was run.** This proof of concept validates
  that the pathway is well-behaved (deterministic, content-sensitive,
  finite, non-degenerate) — it does not show whether DNABERT-2-conditioned
  embeddings are *better or worse* than HyenaDNA's native ones for an
  actual classification task. That requires training the projection layer
  (and likely a task head) against labeled data, which this proof of
  concept's time budget did not include.

### Next steps

- Train the projection layer (and a small classification head) against a
  GenomicBenchmarks task — reusing the infrastructure already verified in
  [Training](#training) — and compare accuracy against the from-scratch
  baseline already recorded there (val/accuracy 0.711), to get an actual
  performance signal rather than only a well-behavedness check.
- Replace the linear-interpolation resampling with exact character-to-token
  alignment (would need a tokenizer that exposes `offset_mapping`, or
  manually re-deriving it from DNABERT-2's BPE merge rules).
- The same script structure generalizes directly to Evo 2 once its
  infrastructure is tractable within a course assignment's time budget:
  swap `DNABERT2_MODEL_ID`/`DNABERT2_HIDDEN` for Evo 2's checkpoint id and
  hidden size, and everything from `resample_to_length` onward is
  unchanged.

## Cleanup

```bash
rm -rf ~/scratch/hyena-dna/checkpoints          # downloaded weights (well under 1GB for tiny/small)
rm -rf ~/scratch/hyena-dna/hyena_train_runs     # training run outputs (checkpoints + logs)
rm -rf ~/scratch/hyena-dna/data                 # auto-downloaded GenomicBenchmarks data
rm -rf ~/scratch/.conda/envs/hyena-dna           # conda environment
rm -rf ~/scratch/.cache/huggingface ~/scratch/.cache/pip  # redirected caches
rm -rf ~/scratch/.conda/envs/dna-bridge         # bridge experiment's separate env
rm -rf ~/scratch/.conda/pkgs                    # conda's package cache (see Troubleshooting)
```

The cloned repo at `~/scratch/hyena-dna` can stay — re-running
`scripts/hyenadna_setup_env.sh` is idempotent and skips the clone if it's
already there.
