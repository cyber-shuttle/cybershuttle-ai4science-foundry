#!/usr/bin/env python3
"""
Pre-flight smoke test. Run on a login node (or inside an srun session) BEFORE
submitting the GPU job, so path/version mistakes cost seconds instead of a queue wait.

Checks: package imports, numpy/torch version guards (inference.py needs numpy<2 and
torch<2.6), CUDA visibility (informational on a login node), required artifact files
and sizes, evaluation-file shape, config section, and that inference.py supports --vis.

Usage:
    python scripts/smoke_test.py --fourcastnet-repo ~/FourCastNet --artifact-root ~/scratch/fcn-artifacts
Exit 0 = safe to submit.
"""
import argparse
import importlib
import os
import sys

REQUIRED_MODULES = ["numpy", "h5py", "torch", "torchvision", "timm", "einops",
                    "ruamel.yaml", "tqdm", "wandb", "matplotlib"]
CKPT_BYTES = 896430597


def check_imports():
    bad = []
    for m in REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append((m, e))
    for m, e in bad:
        print(f"FAIL: import {m}: {e}")
    if not bad:
        print(f"OK: {len(REQUIRED_MODULES)} modules import.")
    return not bad


def check_versions():
    import numpy
    import torch
    ok = True
    np_major = int(numpy.__version__.split(".")[0])
    t = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    print(f"numpy {numpy.__version__}, torch {torch.__version__}")
    if np_major >= 2:
        print("FAIL: numpy>=2 breaks inference.py (imports numpy.core.numeric.False_). Pin numpy<2.")
        ok = False
    if t >= (2, 6):
        print("FAIL: torch>=2.6 defaults torch.load(weights_only=True) and rejects the checkpoint. Pin torch<2.6.")
        ok = False
    if ok:
        print("OK: numpy<2 and torch<2.6.")
    return ok


def check_cuda():
    import torch
    if torch.cuda.is_available():
        print(f"OK: CUDA available, device 0 = {torch.cuda.get_device_name(0)}")
    else:
        print("INFO: no CUDA device here (expected on a login node). Confirm with srun before sbatch.")
    return True


def check_artifacts(root):
    b = os.path.join(root, "ccai_demo")
    files = [os.path.join(b, "model_weights", "FCN_weights_v0", "backbone.ckpt"),
             os.path.join(b, "data", "FCN_ERA5_data_v0", "out_of_sample", "2018.h5")] + \
            [os.path.join(b, "additional", "stats_v0", n)
             for n in ("global_means.npy", "global_stds.npy", "time_means.npy")]
    ok = True
    for p in files:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            print(f"FAIL: missing/empty {p}")
            ok = False
    ck = files[0]
    if os.path.isfile(ck) and os.path.getsize(ck) != CKPT_BYTES:
        print(f"FAIL: backbone.ckpt is {os.path.getsize(ck)} bytes, expected {CKPT_BYTES} (re-run download script)")
        ok = False
    if ok:
        print("OK: all artifact files present, checkpoint size matches.")
        import h5py
        with h5py.File(files[1], "r") as f:
            shape = f["fields"].shape
        print(f"OK: evaluation file fields shape = {shape}")
        if len(shape) != 4 or shape[1] < 20 or shape[2] < 720 or shape[3] != 1440:
            print("FAIL: unexpected evaluation-file shape")
            ok = False
    return ok


def check_repo(repo):
    cfg = os.path.join(repo, "config", "AFNO.yaml")
    inf = os.path.join(repo, "inference", "inference.py")
    ok = True
    if not os.path.isfile(cfg) or not os.path.isfile(inf):
        print(f"FAIL: {cfg} or {inf} not found")
        return False
    from ruamel.yaml import YAML
    with open(cfg) as f:
        c = YAML().load(f)
    if "afno_backbone" not in c:
        print("FAIL: no afno_backbone section in config/AFNO.yaml")
        ok = False
    else:
        print("OK: config/AFNO.yaml has afno_backbone.")
    with open(inf) as f:
        src = f.read()
    if "--vis" not in src and "'vis'" not in src and '"vis"' not in src:
        print("WARN: could not find a --vis option in inference.py; the sbatch script passes --vis.")
    else:
        print("OK: inference.py references vis option.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourcastnet-repo", required=True)
    ap.add_argument("--artifact-root", required=True)
    a = ap.parse_args()
    res = [check_imports()]
    if res[0]:
        res += [check_versions(), check_cuda()]
    res += [check_artifacts(os.path.expanduser(a.artifact_root)),
            check_repo(os.path.expanduser(a.fourcastnet_repo))]
    if all(res):
        print("\nSMOKE TEST PASSED. Safe to submit slurm/run_inference.sbatch.")
        sys.exit(0)
    print("\nSMOKE TEST FAILED. Fix the issues above first.")
    sys.exit(1)


if __name__ == "__main__":
    main()
