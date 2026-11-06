#!/usr/bin/env python
"""Verify the ClimaX Python environment and, optionally, its allocated GPU."""

import argparse
import importlib
from importlib import metadata
import os
import shlex
import sys
from pathlib import Path


EXPECTED_TORCH_VERSION = "1.12.1"
EXPECTED_CUDA_VERSION = "11.3"
REQUIRED_IMPORTS = (
    "numpy",
    "xarray",
    "jsonargparse",
    "rich",
    "tensorboard",
    "timm",
    "torchdata",
    "torchvision",
    "pytorch_lightning",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_scratch = Path(os.environ.get("SCRATCH_ROOT", Path.home() / "scratch"))
    default_prefix = default_scratch / ".conda" / "envs" / os.environ.get(
        "ENV_NAME", "climax-inference"
    )
    parser.add_argument(
        "--expected-prefix",
        type=Path,
        default=default_prefix,
        help="Expected Conda environment path (defaults to the workflow scratch environment).",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA is available and a CUDA tensor operation succeeds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expected_prefix = args.expected_prefix.expanduser().resolve()
    actual_prefix = Path(sys.prefix).resolve()
    setup_script = Path(__file__).resolve().with_name("setup_env.sh")
    setup_command = f"bash {shlex.quote(str(setup_script))}"
    debug_command = f"bash -x {shlex.quote(str(setup_script))}"

    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"Active environment: {actual_prefix}", flush=True)
    print(f"Expected environment: {expected_prefix}", flush=True)

    errors = []
    if actual_prefix != expected_prefix:
        errors.append(
            "This is not the expected Conda environment. Activate it with "
            f'`conda activate "{expected_prefix}"`, or run this checker with '
            f'`"{expected_prefix / "bin" / "python"}" '
            f'"{Path(__file__).resolve()}"`.'
        )

    try:
        import torch
    except (ImportError, OSError) as exc:
        errors.append(
            f"PyTorch could not be imported ({type(exc).__name__}: {exc}). "
            "The environment may be incomplete or the wrong Python may be active. "
            f"Run `{setup_command}`, then activate the exact environment "
            f"shown above. For details, retry setup with "
            f"`{debug_command}` and keep the complete output."
        )
        torch = None

    if torch is not None:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        for distribution in (
            "numpy",
            "xarray",
            "torchvision",
            "pytorch-lightning",
            "timm",
            "torchdata",
            "jsonargparse",
            "rich",
            "tensorboard",
        ):
            try:
                print(f"{distribution}: {metadata.version(distribution)}")
            except metadata.PackageNotFoundError:
                errors.append(
                    f"Required distribution `{distribution}` is not installed. "
                    f"Rerun `{setup_command}`."
                )

        if torch.__version__.split("+")[0] != EXPECTED_TORCH_VERSION:
            errors.append(
                f"Expected PyTorch {EXPECTED_TORCH_VERSION}; found {torch.__version__}."
            )
        if not (torch.version.cuda or "").startswith(EXPECTED_CUDA_VERSION):
            errors.append(
                f"Expected a PyTorch build for CUDA {EXPECTED_CUDA_VERSION}; "
                f"found {torch.version.cuda!r}."
            )

        for module_name in REQUIRED_IMPORTS:
            try:
                importlib.import_module(module_name)
            except (ImportError, OSError) as exc:
                errors.append(
                    f"Required package `{module_name}` failed to import "
                    f"({type(exc).__name__}: {exc}). Rerun `{setup_command}`."
                )

        try:
            callbacks = importlib.import_module("pytorch_lightning.callbacks")
            loggers = importlib.import_module("pytorch_lightning.loggers")
            getattr(callbacks, "RichProgressBar")
            getattr(loggers, "TensorBoardLogger")
        except (AttributeError, ImportError, OSError) as exc:
            errors.append(
                "The ClimaX Lightning progress/logger imports failed "
                f"({type(exc).__name__}: {exc}). Rerun `{setup_command}`."
            )

        if args.require_cuda:
            if not torch.cuda.is_available():
                errors.append(
                    "CUDA is not available. Confirm this shell has a GPU allocation "
                    "(not a login node), inspect `nvidia-smi`, and check "
                    "`CUDA_VISIBLE_DEVICES`."
                )
            else:
                try:
                    value = torch.ones(1, device="cuda")
                    torch.cuda.synchronize()
                    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
                    print(f"CUDA tensor smoke test: {value.item()}")
                except RuntimeError as exc:
                    errors.append(
                        f"The CUDA tensor smoke test failed ({type(exc).__name__}: "
                        f"{exc}). Check the allocated GPU model and driver; the pinned "
                        "PyTorch 1.12.1/CUDA 11.3.1 stack was verified on a V100."
                    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Environment verification failed. Do not start ClimaX jobs until each "
            "error is resolved.",
            file=sys.stderr,
        )
        return 1

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
