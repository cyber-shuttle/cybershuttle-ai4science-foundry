#!/usr/bin/env python3
"""Check that the Clay environment matches the lock file and imports cleanly.

Run inside the activated environment from the Clay model folder:
    python scripts/clay_check_env.py --lock scripts/clay_requirements.txt
It prints what it found; the job logs keep that output.

Checks, each one fatal:
  1. Every package pinned in the lock is installed at exactly that version.
  2. torch, timm, lightning, the geo stack and claymodel import.
  3. torch and numpy exchange arrays (catches numpy ABI mismatches).
  4. timm can build the DINOv2 teacher architecture that Clay uses
     (no weights are downloaded here).
On a GPU node it also records the GPU name, compute capability and whether
torch ships kernels for it.
"""
import argparse
import importlib.metadata as md
import platform
import re
import sys
from pathlib import Path


def read_lock(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s\\;]+)", line)
        if m:
            pins[m.group(1)] = m.group(3)
    return pins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lock", required=True, help="scripts/clay_requirements.txt")
    p.add_argument("--teacher", default="vit_large_patch14_reg4_dinov2.lvd142m")
    args = p.parse_args()

    failures = []
    info = {"python": sys.version.split()[0], "platform": platform.platform()}

    # 1. Installed versions match the lock exactly.
    pins = read_lock(args.lock)
    mismatched = {}
    for name, want in pins.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != want:
            mismatched[name] = {"lock": want, "installed": have}
    info["lock_packages"] = len(pins)
    if mismatched:
        failures.append(f"{len(mismatched)} packages differ from the lock: {mismatched}")
    print(f"[lock] {len(pins)} pinned packages, {len(mismatched)} mismatched")

    # 2. Imports.
    import numpy as np
    import torch
    import timm
    import lightning
    import rasterio
    import pyarrow
    import shapely
    import pyproj
    import pystac_client  # noqa: F401
    import planetary_computer  # noqa: F401
    from claymodel.module import ClayMAEModule  # noqa: F401

    for pkg in ["torch", "torchvision", "numpy", "timm", "lightning", "jsonargparse", "einops",
                "rasterio", "pyarrow", "shapely", "pyproj", "geopandas", "pystac-client",
                "planetary-computer", "huggingface-hub", "safetensors", "claymodel"]:
        info[pkg] = md.version(pkg)
    info["torch_cuda_build"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["gdal"] = rasterio.__gdal_version__
    info["proj"] = pyproj.proj_version_str
    info["geos"] = shapely.geos_version_string
    info["torch_arch_list"] = torch.cuda.get_arch_list()
    info["float32_matmul_precision_after_claymodel_import"] = torch.get_float32_matmul_precision()
    del lightning, pyarrow

    # 3. numpy <-> torch.
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    if not np.array_equal(torch.from_numpy(a).numpy(), a):
        failures.append("torch.from_numpy round trip failed")

    # 4. Teacher architecture builds (pretrained=False: no download).
    teacher = timm.create_model(args.teacher, pretrained=False, num_classes=0)
    info["teacher"] = args.teacher
    info["teacher_num_features"] = teacher.num_features
    if teacher.num_features != 1024:
        failures.append(f"teacher has {teacher.num_features} features, expected 1024")
    del teacher

    # GPU details, only when a GPU is visible.
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_compute_capability"] = f"{major}.{minor}"
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        x = torch.randn(256, 256, device="cuda")
        info["gpu_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())

    for k, v in info.items():
        print(f"{k:50s} {v}")

    if failures:
        print("ENV CHECK FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ENV CHECK PASSED")


if __name__ == "__main__":
    main()
