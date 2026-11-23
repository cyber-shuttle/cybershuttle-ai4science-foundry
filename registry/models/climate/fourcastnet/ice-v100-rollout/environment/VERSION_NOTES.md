# Why these versions differ from the original Docker image

The official FourCastNet repo ships a Dockerfile built on `nvcr.io/nvidia/pytorch:21.11-py3`, which bundles an old PyTorch and CUDA pairing. That image predates Hopper-generation GPUs such as the H200, so it has no kernels built for `sm_90`. My recorded run landed on a Tesla V100 (`sm_70`), which the pinned wheels also support. Pinning newer versions keeps the environment usable on whichever GPU type the scheduler hands out.

`environment.yml` pins PyTorch 2.2.2 with CUDA 12.1 wheels and keeps the same package set the original image installs (h5py, timm, einops, ruamel.yaml, tqdm, wandb), since `inference/inference.py` imports all of them directly.

I dropped mpi4py and benchy on purpose. Both are training-time dependencies: `benchy` is a profiling tool called from `submit_batch.sh`, and `mpi4py` handles distributed data loading across nodes. This workflow runs on a single GPU and imports neither.

## Constraints I found by reading `inference/inference.py`

- **numpy below 2.** The script imports `numpy.core.numeric.False_`, which numpy 2 removed.
- **torch below 2.6.** It calls `torch.load` without `weights_only=False`, and torch 2.6 changed that default and rejects the checkpoint.
- **Verified on PACE-ICE.** torch 2.2.2+cu121 imports in the `fourcastnet-inference` environment, and `scripts/smoke_test.py` enforces both pins.

## wandb

`inference.py` imports `wandb`, and I don't need remote logging, so the job script sets `WANDB_MODE=disabled` and `configure_paths.py` sets `log_to_wandb = False`. That lets it start without an account or network access.
