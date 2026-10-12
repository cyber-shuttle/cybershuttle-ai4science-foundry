#!/usr/bin/env python
"""Extract spatial and pooled embeddings with the pretrained ClimaX encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


EXPECTED_CHECKPOINT_SHA256 = "6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43"
CHECKPOINT_REVISION = "6f9d4a5c0c2602965b2a381e60636fed7f12b8f4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climax-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Pinned pretraining config")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned pretrained checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True, help="Preprocessed single-variable ERA5 data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable", default="geopotential_500")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sample-stride", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.climax_source.resolve()
    workflow_repo = Path(__file__).resolve().parents[3]
    source_src = source / "src"
    if not source_src.is_dir():
        raise FileNotFoundError("ClimaX source must contain src/: {}".format(source))
    sys.path.insert(0, str(source_src))

    from climax.arch import ClimaX
    from climax.pretrain.module import PretrainModule

    if not args.config.is_file():
        raise FileNotFoundError("Pretraining config not found: {}".format(args.config))
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Pretrained checkpoint not found: {}".format(args.checkpoint))

    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Pretrained checkpoint SHA-256 mismatch: got {}, expected {}".format(
                checkpoint_hash, EXPECTED_CHECKPOINT_SHA256
            )
        )

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_config = config["model"]["net"]["init_args"]
    if args.variable not in model_config["default_vars"]:
        raise ValueError(
            "Variable {!r} is not supported by the pinned pretrained model config".format(args.variable)
        )

    shard = args.shard
    if shard is None:
        candidates = sorted((args.data_dir / args.split).glob("*.npz"))
        candidates = [path for path in candidates if path.name != "climatology.npz"]
        if not candidates:
            raise FileNotFoundError("No .npz shards found under {}".format(args.data_dir / args.split))
        shard = candidates[0]
    elif not shard.is_absolute():
        shard = args.data_dir / args.split / shard
    if not shard.is_file():
        raise FileNotFoundError("Selected shard not found: {}".format(shard))
    if args.time_index < 0 or args.num_samples <= 0 or args.sample_stride <= 0:
        raise ValueError("Time index must be non-negative; sample count and stride must be positive")

    with np.load(shard) as archive:
        if args.variable not in archive.files:
            raise KeyError("Required variable {!r} is absent from {}".format(args.variable, shard))
        values = np.asarray(archive[args.variable], dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(
            "Variable {!r} must have shape [time, 1, lat, lon]; got {}".format(
                args.variable, values.shape
            )
        )
    time_count, _, height, width = values.shape
    if (height, width) != tuple(model_config["img_size"]):
        raise ValueError(
            "Input grid {} x {} does not match pretrained model grid {}".format(
                height, width, model_config["img_size"]
            )
        )
    lat = np.asarray(np.load(args.data_dir / "lat.npy"), dtype=np.float32)
    lon = np.asarray(np.load(args.data_dir / "lon.npy"), dtype=np.float32)
    if lat.shape != (height,) or lon.shape != (width,):
        raise ValueError(
            "Latitude/longitude coordinates {} / {} do not match input grid {} x {}".format(
                lat.shape, lon.shape, height, width
            )
        )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Latitude/longitude coordinates contain non-finite values")

    indices = args.time_index + np.arange(args.num_samples) * args.sample_stride
    if indices[-1] >= time_count:
        raise ValueError(
            "Requested input index {} exceeds {} timesteps in {}".format(
                int(indices[-1]), time_count, shard
            )
        )

    with np.load(args.data_dir / "normalize_mean.npz") as archive:
        if args.variable not in archive.files:
            raise KeyError("Normalization mean is missing {!r}".format(args.variable))
        mean = float(np.asarray(archive[args.variable]).reshape(-1)[0])
    mean_path = args.data_dir / "normalize_mean.npz"
    with np.load(args.data_dir / "normalize_std.npz") as archive:
        if args.variable not in archive.files:
            raise KeyError("Normalization standard deviation is missing {!r}".format(args.variable))
        std = float(np.asarray(archive[args.variable]).reshape(-1)[0])
    std_path = args.data_dir / "normalize_std.npz"
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Normalization statistics must be finite and standard deviation positive")

    inputs = values[indices, 0]
    normalized = (inputs - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized ERA5 inputs contain non-finite values")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    net = ClimaX(**model_config)
    module = PretrainModule(net=net)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    if "state_dict" not in checkpoint:
        raise KeyError("Pretrained checkpoint has no 'state_dict' key")
    # Compatibility with checkpoints saved before channel_* was renamed var_*.
    state_dict = checkpoint["state_dict"]
    key_mapping = {
        "net.channel_embed": "net.var_embed",
        "net.channel_query": "net.var_query",
        "net.channel_agg.": "net.var_agg.",
    }

    for old_key in list(state_dict.keys()):
        new_key = old_key
        for old_name, new_name in key_mapping.items():
            if old_name.endswith("."):
                if old_key.startswith(old_name):
                    new_key = new_name + old_key[len(old_name):]
                    break
            elif old_key == old_name:
                new_key = new_name
                break

        if new_key != old_key:
            if new_key in state_dict:
                raise RuntimeError(
                    "Checkpoint contains both {!r} and {!r}".format(
                        old_key, new_key
                    )
                )
            state_dict[new_key] = state_dict.pop(old_key)

    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()

    input_tensor = torch.from_numpy(normalized[:, None]).to(device=device, dtype=torch.float32)
    # Encoder features do not forecast; zero selects the neutral lead-time embedding.
    lead_times = torch.zeros(args.num_samples, device=device, dtype=torch.float32)
    start_time = time.monotonic()
    with torch.no_grad():
        token_embeddings = module.net.forward_encoder(input_tensor, lead_times, [args.variable])
        repeated_embeddings = module.net.forward_encoder(
            input_tensor, lead_times, [args.variable]
        )
    runtime_seconds = time.monotonic() - start_time
    tokens = token_embeddings.detach().cpu().numpy().astype(np.float32)
    pooled = tokens.mean(axis=1)
    expected_token_count = (height // int(model_config["patch_size"])) * (
        width // int(model_config["patch_size"])
    )
    expected_tokens_shape = (args.num_samples, expected_token_count, int(model_config["embed_dim"]))
    expected_pooled_shape = (args.num_samples, int(model_config["embed_dim"]))
    if tokens.shape != expected_tokens_shape or pooled.shape != expected_pooled_shape:
        raise ValueError(
            "Unexpected embedding shapes: spatial {}, pooled {}; expected {} and {}".format(
                tokens.shape, pooled.shape, expected_tokens_shape, expected_pooled_shape
            )
        )
    if not np.isfinite(tokens).all() or not np.isfinite(pooled).all():
        raise ValueError("Encoder embeddings contain non-finite values")
    if float(tokens.std()) <= 0:
        raise ValueError("Encoder returned a constant representation")
    deterministic = torch.allclose(
        token_embeddings,
        repeated_embeddings,
        rtol=1e-5,
        atol=1e-6,
    )
    if not deterministic:
        difference = (token_embeddings[:1] - repeated_first).abs()
        max_abs_diff = difference.max().item()
        mean_abs_diff = difference.mean().item()
        reference_max = token_embeddings[:1].abs().max().item()
        raise ValueError(
            "Repeated evaluation differs beyond tolerance: "
            "max_abs_diff={}, mean_abs_diff={}, reference_max={}, "
            "rtol=1e-5, atol=1e-6".format(
                max_abs_diff, mean_abs_diff, reference_max
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "climax-embeddings.npz"
    np.savez_compressed(
        output_path,
        spatial_embeddings=tokens,
        pooled_embeddings=pooled.astype(np.float32),
        variable=np.asarray(args.variable),
        sample_indices=indices.astype(np.int64),
        lat=lat,
        lon=lon,
        lead_time_hours=np.asarray(0, dtype=np.int32),
    )

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    metrics: Dict[str, Any] = {
        "smoke_test": {
            "spatial_embedding_shape": list(tokens.shape),
            "pooled_embedding_shape": list(pooled.shape),
            "all_embedding_values_finite": bool(np.isfinite(tokens).all()),
            "nonconstant_embedding": bool(float(tokens.std()) > 0),
            "repeat_inference_deterministic": bool(deterministic),
            "status": "passed",
        },
        "representation": {
            "variable": args.variable,
            "split": args.split,
            "shard": str(shard.resolve()),
            "sample_indices": indices.tolist(),
            "input_shape": list(inputs.shape),
            "grid_shape": [height, width],
            "latitude_min": float(lat.min()),
            "latitude_max": float(lat.max()),
            "longitude_min": float(lon.min()),
            "longitude_max": float(lon.max()),
            "patch_size": int(model_config["patch_size"]),
            "pooled_embedding_dimension": int(model_config["embed_dim"]),
            "spatial_token_count": int(tokens.shape[1]),
            "mean_token_std": float(tokens.std(axis=(1, 2)).mean()),
            "pooled_vector_l2_norm_mean": float(np.linalg.norm(pooled, axis=1).mean()),
        },
        "run": {
            "runtime_seconds_encoder": runtime_seconds,
            "device": str(device),
            "gpu_name": gpu_name,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "input_shard": str(shard.resolve()),
            "input_shard_sha256": sha256_file(shard),
            "climax_source": str(source),
            "climax_source_commit": git_commit(source),
            "workflow_commit": git_commit(workflow_repo),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "normalization_mean_sha256": sha256_file(mean_path),
            "normalization_std_sha256": sha256_file(std_path),
            "data_normalization_mean": mean,
            "data_normalization_std": std,
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "output_file": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
