#!/usr/bin/env python3
"""Single-GPU temporal SatMAE evaluation on fMoW-Temporal.

Imports only the upstream pieces needed for the 79.99% no-TTA setup:

  * models_vit_temporal.vit_large_patch16
  * util.datasets.CustomDatasetFromImagesTemporal

It does not import main_finetune.py (wandb entity, 8-GPU launch) or
util.misc (torch._six.inf). Official eval uses RandomCrop(224) and a
random choice of companion timestamps even at test time; we keep that
behavior and seed it so a re-run is comparable.

Environment:
  SATMAE_CODE_DIR   cloned SatMAE repo (pinned commit)
  SATMAE_TEMPORAL_WORKDIR   default scratch prefix
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FMOW_CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo",
]


def add_satmae_to_path(code_dir: str) -> None:
    code = str(Path(code_dir).resolve())
    if code not in sys.path:
        sys.path.insert(0, code)


def restore_scale_transform() -> None:
    """SatMAE calls transforms.Scale(224). This conda torchvision 0.13.1
    build no longer ships that alias (job 6195919). Resize is the same op.
    """
    from torchvision import transforms

    if not hasattr(transforms, "Scale"):
        transforms.Scale = transforms.Resize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(nb_classes: int, drop_path: float, global_pool: bool):
    import models_vit_temporal

    model = models_vit_temporal.vit_large_patch16(
        num_classes=nb_classes,
        drop_path_rate=drop_path,
        global_pool=global_pool,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    msg = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
        "n_tensors": len(state),
    }


def select_eval_indices(dataset, eval_csv: str):
    """Score a subset CSV while keeping the full val index.

    CustomDatasetFromImagesTemporal.parse_timestamp looks up companion
    frames in name2index, which is built only from the CSV passed to the
    constructor. A 500-row CSV therefore KeyErrors when the temporal glob
    finds a crop that is on disk but not in that subset (job 6197109).
    Load the full prepared CSV, then evaluate these paths.
    """
    df = pd.read_csv(eval_csv)
    path_col = df.columns[1]
    wanted = {str(p).strip() for p in df[path_col]}
    image_arr = list(dataset.image_arr)
    indices = [i for i, path in enumerate(image_arr) if str(path).strip() in wanted]
    if not indices:
        wanted_names = {Path(p).name for p in wanted}
        indices = [i for i, path in enumerate(image_arr) if Path(str(path)).name in wanted_names]
    if not indices:
        raise SystemExit(f"no rows from {eval_csv} matched the full val CSV")
    return indices


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = max(topk)
    batch = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        out.append(correct[:k].reshape(-1).float().sum(0) * (100.0 / batch))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Prepared val CSV (paths relative to its directory)")
    parser.add_argument("--checkpoint", default=None, help="finetune_fmow_temporal.pth")
    parser.add_argument("--code-dir", default=os.environ.get("SATMAE_CODE_DIR", ""))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows (smoke)")
    parser.add_argument(
        "--eval-csv",
        default=None,
        help="Score these rows only; --csv must stay the full val CSV so companion timestamps resolve",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nb-classes", type=int, default=1000, help="Official checkpoint legacy head size")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--no-global-pool", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Skip checkpoint load (chance baseline)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.code_dir:
        raise SystemExit("Pass --code-dir or set SATMAE_CODE_DIR to the pinned SatMAE clone")
    add_satmae_to_path(args.code_dir)
    restore_scale_transform()
    set_seed(args.seed)

    from util.datasets import CustomDatasetFromImagesTemporal

    dataset = CustomDatasetFromImagesTemporal(args.csv)
    if args.eval_csv:
        dataset = torch.utils.data.Subset(dataset, select_eval_indices(dataset, args.eval_csv))
    elif args.limit and args.limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(args.limit)))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.nb_classes, args.drop_path, global_pool=not args.no_global_pool)
    ckpt_info = {"missing_keys": [], "unexpected_keys": [], "n_tensors": 0}
    if args.random_init:
        print("random-init: skipping checkpoint load")
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --random-init is set")
        ckpt_info = load_checkpoint(model, args.checkpoint)
        print(
            f"loaded {args.checkpoint}: "
            f"missing={len(ckpt_info['missing_keys'])} "
            f"unexpected={len(ckpt_info['unexpected_keys'])}"
        )
        if ckpt_info["missing_keys"]:
            print("missing_keys", ckpt_info["missing_keys"][:20])
        if ckpt_info["unexpected_keys"]:
            print("unexpected_keys", ckpt_info["unexpected_keys"][:20])

    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    n_classes = 62
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total = 0
    correct1 = 0.0
    correct5 = 0.0
    loss_sum = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        for images, timestamps, target in loader:
            images = images.to(device, non_blocking=True)
            timestamps = timestamps.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(images, timestamps)
                loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch = target.size(0)
            total += batch
            correct1 += acc1.item() * batch / 100.0
            correct5 += acc5.item() * batch / 100.0
            loss_sum += loss.item() * batch

            pred = output.argmax(dim=1)
            for y, p in zip(target.tolist(), pred.tolist()):
                if 0 <= y < n_classes and 0 <= p < n_classes:
                    confusion[y, p] += 1

            if total % max(args.batch_size * 10, 1) == 0:
                print(f"processed {total}  running_top1={100.0 * correct1 / total:.2f}")

    runtime = time.time() - t0
    top1 = 100.0 * correct1 / max(total, 1)
    top5 = 100.0 * correct5 / max(total, 1)
    support = confusion.sum(axis=1)
    per_class = {
        FMOW_CATEGORIES[i]: (float(confusion[i, i] / support[i]) if support[i] else None)
        for i in range(n_classes)
    }

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_images": int(total),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "loss": round(loss_sum / max(total, 1), 6),
        "per_class_accuracy": per_class,
        "runtime_seconds": round(runtime, 2),
        "gpu_name": gpu_name,
        "peak_gpu_memory_gb": round(peak_gb, 3),
        "missing_keys": ckpt_info["missing_keys"],
        "unexpected_keys": ckpt_info["unexpected_keys"],
        "checkpoint": args.checkpoint,
        "random_init": bool(args.random_init),
        "nb_classes": args.nb_classes,
        "seed": args.seed,
        "limit": args.limit,
        "csv": args.csv,
        "eval_csv": args.eval_csv,
        "upstream_code": args.code_dir,
        "tta": False,
        "reference_top1_repo": 79.99,
        "reference_top1_paper_no_tta": 79.69,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(confusion, index=FMOW_CATEGORIES, columns=FMOW_CATEGORIES).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    print(
        f"n={total} top1={top1:.2f} top5={top5:.2f} "
        f"runtime_s={runtime:.1f} peak_gpu_gb={peak_gb:.2f} gpu={gpu_name}"
    )
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"wrote {out_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
