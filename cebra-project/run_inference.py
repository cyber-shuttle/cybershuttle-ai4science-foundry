"""Train, save, reload and transform the official Achilles hippocampus data."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import traceback


DATASET_ID = "rat-hippocampus-single-achilles"
MODEL_PARAMETERS = {
    "model_architecture": "offset10-model",
    "batch_size": 512,
    "learning_rate": 3e-4,
    "output_dimension": 3,
    "distance": "cosine",
    "conditional": "time_delta",
    "time_offsets": 10,
    "temperature": 1,
    "max_iterations": 5000,
    "device": "cuda",
    "verbose": True,
    "pad_before_transform": True,
}


def file_checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hippocampus"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new output directory; {output_dir} is not empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    run = {
        "status": "running",
        "stage": "environment",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "seed": args.seed,
        "dataset_id": DATASET_ID,
        "model_parameters": MODEL_PARAMETERS,
        "sample_policy": "Train and transform every sample in original temporal order; no holdout.",
        "data_dir": str(data_dir),
        "outputs": {"metadata": str(metadata_path)},
        "source_sha256": file_checksum(Path(__file__)),
    }
    print(json.dumps(run, indent=2), flush=True)

    try:
        # Include the versions used by this run in its execution evidence.
        versions = {}
        for name in ("cebra", "numpy", "scikit-learn", "scipy", "matplotlib", "torch"):
            try:
                versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError:
                versions[name] = "NOT INSTALLED"
        run["package_versions"] = versions
        print("Installed versions:", json.dumps(versions, indent=2), flush=True)
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("This workflow expects the documented Python 3.10 environment.")
        if versions["cebra"] != "0.6.1":
            raise RuntimeError("This workflow requires cebra==0.6.1.")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Run this workflow inside an allocated PACE SLURM job.")

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["CEBRA_DATADIR"] = str(data_dir)
        import numpy as np
        import torch
        import cebra

        cuda_available = torch.cuda.is_available()
        run["cuda_available"] = cuda_available
        run["torch_cuda_build"] = torch.version.cuda
        run["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if cuda_available else []
        print("CUDA available:", cuda_available, flush=True)
        print("PyTorch CUDA build:", torch.version.cuda, flush=True)
        print("Visible GPU names:", run["gpu_names"], flush=True)
        if not cuda_available:
            raise RuntimeError("CUDA is required; check the GPU allocation and PyTorch build.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if os.environ.get("SLURM_CPUS_PER_TASK"):
            torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
        run["determinism"] = {
            "deterministic_algorithms": "enabled with warn_only=True",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "note": "CEBRA does not guarantee identical retraining results across runs or hardware.",
        }

        run["stage"] = "data retrieval"
        # Register hippocampus even if imports of unrelated optional datasets fail.
        import cebra.datasets
        import cebra.datasets.hippocampus

        data_dir.mkdir(parents=True, exist_ok=True)
        dataset = cebra.datasets.init(DATASET_ID, root=str(data_dir), download=True)
        neural = dataset.neural.detach().cpu().numpy()
        behavior = dataset.continuous_index.detach().cpu().numpy()
        run["neural_shape"] = list(neural.shape)
        run["behavior_shape"] = list(behavior.shape)
        run["neural_dtype"] = str(neural.dtype)
        run["behavior_dtype"] = str(behavior.dtype)
        run["neural_finite"] = bool(np.isfinite(neural).all())
        run["behavior_finite"] = bool(np.isfinite(behavior).all())
        print("Neural shape / dtype:", neural.shape, neural.dtype, flush=True)
        print("Behavior shape / dtype:", behavior.shape, behavior.dtype, flush=True)
        print("Finite neural / behavior:", run["neural_finite"], run["behavior_finite"], flush=True)
        if neural.ndim != 2 or neural.shape[0] <= 10 or neural.shape[1] == 0:
            raise ValueError("Expected a nonempty time-by-neuron array with more than 10 samples.")
        if behavior.shape != (neural.shape[0], 3):
            raise ValueError("Expected aligned behavior columns: position, right, left.")
        if not run["neural_finite"] or not run["behavior_finite"]:
            raise ValueError("Neural activity and behavior must contain no NaN or Inf values.")

        data_path = data_dir / "rat_hippocampus" / "achilles.jl"
        observed_md5 = file_checksum(data_path, "md5")
        run["dataset_artifact"] = {
            "path": str(data_path),
            "url": dataset.data_url,
            "expected_md5": dataset.data_checksum,
            "observed_md5": observed_md5,
            "sha256": file_checksum(data_path),
            "original_data_doi": "10.6080/K0862DC5",
        }
        if observed_md5 != dataset.data_checksum:
            raise ValueError("The cached dataset does not match the official loader checksum.")

        run["stage"] = "training"
        print("Stage: training on all samples", flush=True)
        print("Model parameters:", json.dumps(MODEL_PARAMETERS, indent=2), flush=True)
        model = cebra.CEBRA(**MODEL_PARAMETERS)
        training_started = time.perf_counter()
        model.fit(neural, behavior)
        torch.cuda.synchronize()
        run["training_seconds"] = time.perf_counter() - training_started
        print("Training seconds:", run["training_seconds"], flush=True)

        run["stage"] = "checkpoint save and reload"
        checkpoint_path = output_dir / "model.pt"
        model.save(str(checkpoint_path))
        run["outputs"]["checkpoint"] = str(checkpoint_path)
        run["checkpoint_sha256"] = file_checksum(checkpoint_path)
        del model
        model = cebra.CEBRA.load(str(checkpoint_path))
        print("Reloaded checkpoint:", checkpoint_path, flush=True)

        run["stage"] = "inference"
        inference_started = time.perf_counter()
        embeddings = model.transform(neural)
        torch.cuda.synchronize()
        run["inference_seconds"] = time.perf_counter() - inference_started
        run["embedding_shape"] = list(embeddings.shape)
        run["embedding_dtype"] = str(embeddings.dtype)
        run["embedding_finite"] = bool(np.isfinite(embeddings).all())
        print("Embedding shape / dtype:", embeddings.shape, embeddings.dtype, flush=True)
        print("Embedding finite:", run["embedding_finite"], flush=True)
        print("Inference seconds:", run["inference_seconds"], flush=True)
        if embeddings.shape != (len(neural), 3) or not run["embedding_finite"]:
            raise ValueError("Expected finite embeddings of shape (number of neural samples, 3).")

        embedding_path = output_dir / "embeddings.npz"
        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            behavior=behavior,
            sample_indices=np.arange(len(neural)),
        )
        run["outputs"]["embeddings"] = str(embedding_path)
        run["embeddings_sha256"] = file_checksum(embedding_path)
        run["status"] = "success"
        run["stage"] = "complete"
        print("Output paths:", json.dumps(run["outputs"], indent=2), flush=True)
    except Exception as error:
        run["status"] = "failed"
        run["error_type"] = type(error).__name__
        run["error"] = str(error)
        print(f"FAILED during {run['stage']}: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run["elapsed_seconds"] = time.perf_counter() - started
        metadata_path.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        print("Run status:", run["status"], flush=True)
        print("Elapsed seconds:", run["elapsed_seconds"], flush=True)
        print("Metadata:", metadata_path, flush=True)


if __name__ == "__main__":
    main()
