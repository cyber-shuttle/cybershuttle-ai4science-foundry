#!/usr/bin/env python3
"""Standalone evaluation of the SatMAE ViT-Base group_c fine-tuned checkpoint
on the fMoW-Sentinel validation split.

Imports the upstream model/dataset code directly from a sustainlab-group/SatMAE
checkout (models_vit_group_channels.py, util/datasets.py) rather than reusing
main_finetune.py / engine_finetune.py, so this script needs no
torch.distributed launcher, no wandb, and never imports util/misc.py (which
pulls in `from torch._six import inf`, removed in torch>=1.13 -- see
scripts/satmae_environment.yml for why torch is pinned to 1.12.1).

The checkout is found via $SATMAE_REPO (default: $SATMAE_WORKDIR/SatMAE,
i.e. ~/scratch/satmae/SatMAE); scripts/satmae_build_env.sbatch clones it at
the pinned commit 0b210aceb37a14bbbd897110db5b104b3271d818.

Fixed args match the upstream README's ViT-Base group_c finetune command:
    --input_size 96 --patch_size 8 --model_type group_c
    --dataset_type sentinel --dropped_bands 0 9 10
    --grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9
    --nb_classes 62
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SATMAE_WORKDIR = Path(os.environ.get("SATMAE_WORKDIR", Path.home() / "scratch" / "satmae"))
SATMAE_ROOT = Path(os.environ.get("SATMAE_REPO", SATMAE_WORKDIR / "SatMAE"))
sys.path.insert(0, str(SATMAE_ROOT))

import models_vit_group_channels  # noqa: E402
from util.datasets import SentinelIndividualImageDataset, CATEGORIES  # noqa: E402

INPUT_SIZE = 96
PATCH_SIZE = 8
DROPPED_BANDS = [0, 9, 10]
CHANNEL_GROUPS = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
NB_CLASSES = 62


def build_model():
    model = models_vit_group_channels.vit_base_patch16(
        patch_size=PATCH_SIZE,
        img_size=INPUT_SIZE,
        in_chans=13 - len(DROPPED_BANDS),
        channel_groups=CHANNEL_GROUPS,
        num_classes=NB_CLASSES,
        drop_path_rate=0.0,
        global_pool=True,
    )
    return model


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys ({len(msg.missing_keys)}): {sorted(msg.missing_keys)}")
    print(f"Unexpected keys ({len(msg.unexpected_keys)}): {sorted(msg.unexpected_keys)}")
    return msg


def topk_correct(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Prepared val CSV (image_path already rebuilt)")
    parser.add_argument("--data-root", default="data",
                         help="Root dir that the CSV's image_path values are relative to")
    parser.add_argument("--checkpoint", default=None, help="Path to finetune-vit-base-e7.pth")
    parser.add_argument("--random-init", action="store_true",
                         help="Skip checkpoint loading; evaluate a randomly initialized model "
                              "as a sanity baseline (expected ~1/62 = 1.6%% top-1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N images (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if not args.random_init and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --random-init is set")

    csv_path = str(Path(args.csv).resolve())
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(args.checkpoint).resolve()) if args.checkpoint else None

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    model = build_model()
    if args.random_init:
        print("--random-init set: evaluating randomly initialized weights (sanity baseline).")
    else:
        load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()

    # SentinelIndividualImageDataset reads image_path straight from the CSV
    # with rasterio.open(image_path) and does not join it to any root dir,
    # so we chdir into data_root to resolve the relative
    # fmow-sentinel/val/... paths written by prepare_csv.py.
    os.chdir(data_root)

    mean = SentinelIndividualImageDataset.mean
    std = SentinelIndividualImageDataset.std
    transform = SentinelIndividualImageDataset.build_transform(False, INPUT_SIZE, mean, std)
    dataset = SentinelIndividualImageDataset(
        csv_path, transform, masked_bands=None, dropped_bands=DROPPED_BANDS
    )

    if args.limit is not None:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_images = 0
    top1_correct = 0.0
    top5_correct = 0.0
    class_correct = np.zeros(NB_CLASSES, dtype=np.int64)
    class_total = np.zeros(NB_CLASSES, dtype=np.int64)
    confusion = np.zeros((NB_CLASSES, NB_CLASSES), dtype=np.int64)

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else None

    start = time.time()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if autocast_ctx is not None:
                with autocast_ctx():
                    output = model(images)
            else:
                output = model(images)

            c1, c5 = topk_correct(output, targets, topk=(1, 5))
            top1_correct += c1
            top5_correct += c5
            n_images += images.size(0)

            preds = output.argmax(dim=1).cpu().numpy()
            tgt_np = targets.cpu().numpy()
            for t, p in zip(tgt_np, preds):
                class_total[t] += 1
                confusion[t, p] += 1
                if t == p:
                    class_correct[t] += 1
    runtime_seconds = time.time() - start

    top1_acc = 100.0 * top1_correct / n_images
    top5_acc = 100.0 * top5_correct / n_images

    per_class_acc = {}
    for i, cat in enumerate(CATEGORIES):
        per_class_acc[cat] = (
            100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else None
        )

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    )

    metrics = {
        "checkpoint": ckpt_path,
        "random_init": args.random_init,
        "csv": csv_path,
        "n_images": n_images,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "per_class_accuracy": per_class_acc,
        "runtime_seconds": runtime_seconds,
        "gpu_name": gpu_name,
        "peak_gpu_memory_gb": peak_mem_gb,
        "device": str(device),
        "limit": args.limit,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    confusion_path = out_dir / "confusion_matrix.csv"
    import csv as csv_module
    with open(confusion_path, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["true\\pred"] + CATEGORIES)
        for i, cat in enumerate(CATEGORIES):
            writer.writerow([cat] + confusion[i].tolist())
    print(f"Wrote confusion matrix to {confusion_path}")

    print(f"n_images={n_images} top1={top1_acc:.2f}% top5={top5_acc:.2f}% "
          f"runtime={runtime_seconds:.1f}s gpu={gpu_name} peak_mem={peak_mem_gb}")


if __name__ == "__main__":
    main()
