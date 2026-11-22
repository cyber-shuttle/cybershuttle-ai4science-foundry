"""Configure and invoke official POYO train.py, then export held-out predictions."""

import argparse
import importlib.util
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from temporaldata import Data
from torch.utils.data import DataLoader
from torch_brain.data import collate
from torch_brain.data.sampler import SequentialFixedWindowSampler
from torch_brain.dataset import Dataset
from torch_brain.nn.loss import MSELoss
from torch_brain.registry import DataType, MODALITY_REGISTRY, register_modality

POYO_REVISION = "6b052f1"
READOUT = "achilles_behavior_3d"


class AchillesDataset(Dataset):
    """Local data adapter to the verified torch_brain.dataset.Dataset extension points."""

    def __init__(self, root, transform=None):
        super().__init__(dataset_dir=root, recording_ids=["achilles"],
                         transform=transform, keep_files_open=False)

    def get_recording_hook(self, data):
        data.config = {"readout": {"readout_id": READOUT}}

    def get_unit_ids(self):
        return self.get_recording("achilles").units.id.tolist()

    def get_sampling_intervals(self, split):
        if split not in ("train", "valid", "test"):
            raise ValueError("Unknown split.")
        return {"achilles": getattr(self.get_recording("achilles"), split + "_domain")}

    def get_recording_config_dict(self):
        return {"achilles": self.get_recording("achilles").config}


def register_readout():
    if READOUT not in MODALITY_REGISTRY:
        register_modality(READOUT, dim=3, type=DataType.CONTINUOUS,
                          timestamp_key="behavior.timestamps", value_key="behavior.values",
                          loss_fn=MSELoss())


def load_official_entrypoint(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if not revision.startswith(POYO_REVISION):
        raise ValueError("Use the reviewed official POYO revision documented in README.md.")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean official POYO checkout so source provenance is unambiguous.")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("poyo_official_train", repo / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Two narrow compatibility bridges for reviewed upstream callback code:
    # Callback has no log_dict; sparse embedding gradients have no dense mean.
    # Change bindings in this imported module only, never the upstream files.
    class LoggingEvaluator(module.DecodingStitchEvaluator):
        def on_validation_epoch_end(self, trainer, pl_module, prefix="val"):
            self.log_dict = pl_module.log_dict
            return super().on_validation_epoch_end(trainer, pl_module, prefix=prefix)

    module.DecodingStitchEvaluator = LoggingEvaluator
    module.ModelWeightStatsLogger = partial(module.ModelWeightStatsLogger, grads=False)
    return module, revision


def export_predictions(upstream, cfg, prepared, output):
    """Use official tokenization/forward; aggregate overlap by original target row."""
    checkpoints = [p for p in output.rglob("*.ckpt") if p.name != "last.ckpt"]
    if len(checkpoints) != 1:
        raise RuntimeError("Expected one best validation checkpoint; refusing ambiguous model selection.")
    data_module = upstream.DataModule(cfg)
    model = hydra.utils.instantiate(cfg.model, readout_spec=data_module.readout_spec)
    data_module.link_model(model)
    wrapper = upstream.TrainWrapper(cfg, model, data_module.readout_spec)
    # Only load a checkpoint just produced by this local training invocation.
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with h5py.File(prepared / "achilles.h5", "r") as handle:
        recording = Data.from_hdf5(handle, lazy=False)
    target_times = recording.behavior.timestamps
    targets = recording.behavior.values
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=data_module.eval_dataset.get_sampling_intervals("test"),
        window_length=model.sequence_length, step=model.sequence_length / 2,
        drop_short=True,
    )
    loader = DataLoader(data_module.eval_dataset, sampler=sampler, batch_size=1,
                        collate_fn=collate, num_workers=0)
    sums = np.zeros(targets.shape, dtype=np.float64)
    coverage = np.zeros(len(targets), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch["model_inputs"].items()}
            predicted = model(**inputs).cpu().numpy()[0]
            mask = batch["model_inputs"]["output_mask"][0].numpy().astype(bool)
            absolute = (batch["model_inputs"]["output_timestamps"][0].numpy()[mask].astype(np.float64)
                        + float(batch["absolute_start"][0]))
            insertion = np.searchsorted(target_times, absolute)
            right = np.clip(insertion, 0, len(target_times) - 1)
            left = np.clip(insertion - 1, 0, len(target_times) - 1)
            indices = np.where(abs(target_times[left] - absolute) < abs(target_times[right] - absolute), left, right)
            separation = np.min(np.diff(target_times))
            tolerance = min(1e-5, separation / 4)
            if np.any(abs(target_times[indices] - absolute) > tolerance):
                raise ValueError("Could not map output queries back to original behavior rows.")
            if np.any(recording.behavior.split[indices] != 2):
                raise ValueError("Test prediction crossed a split boundary.")
            np.add.at(sums, indices, predicted[mask])
            np.add.at(coverage, indices, 1)
    rows = np.flatnonzero(coverage > 0)
    if len(rows) < 2:
        raise RuntimeError("Insufficient held-out predictions.")
    predictions = sums[rows] / coverage[rows, None]
    truth = targets[rows]
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite decoding output.")
    mse = np.mean((predictions - truth) ** 2, axis=0)
    denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = [float(1 - np.sum((predictions[:, i] - truth[:, i]) ** 2) / d) if d > 0 else None
          for i, d in enumerate(denominator)]
    np.savez_compressed(output / "predictions.npz", predictions=predictions, targets=truth,
                        timestamps=target_times[rows], row_index=rows, overlap_count=coverage[rows])
    metrics = {"split": "test", "shape": list(predictions.shape),
               "target_columns": ["position", "right", "left"],
               "mse_per_column": mse.tolist(), "r2_per_column": r2,
               "covered_rows": len(rows), "total_test_rows": int(np.sum(recording.behavior.split == 2)),
               "note": "Per-column regression metrics; direction outputs are continuous scores."}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poyo-repo", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("Use at least two epochs for the official OneCycleLR schedule.")
    repo, prepared, output = args.poyo_repo.resolve(), args.prepared.resolve(), args.output.resolve()
    if not (prepared / "manifest.json").is_file() or not (prepared / "achilles.h5").is_file():
        parser.error("Prepare validated Achilles input first.")
    upstream, revision = load_official_entrypoint(repo)
    register_readout()
    # Hydra imports this script as a module for its dataset target.
    sys.modules["run_poyo"] = sys.modules[__name__]
    with initialize_config_dir(version_base="1.3", config_dir=str(repo / "configs")):
        cfg = compose(config_name="train_poyo_1.yaml")
    with open_dict(cfg):
        cfg.dataset = {"_target_": "run_poyo.AchillesDataset"}
        cfg.data_root = str(prepared)
        cfg.log_dir = str(output)
        cfg.epochs = args.epochs
        cfg.batch_size = 8
        cfg.eval_batch_size = 8
        cfg.num_workers = 0
        cfg.gpus = 1
        cfg.nodes = 1
        cfg.train_transforms = []
        cfg.eval_transforms = []
        cfg.wandb.enable = False
    output.mkdir(parents=True, exist_ok=False)
    (output / "upstream_revision.txt").write_text(revision + "\n")
    # main is the real @hydra.main entrypoint from official train.py. Passing an
    # already composed DictConfig uses Hydra's supported cfg_passthrough route.
    upstream.main(cfg)
    export_predictions(upstream, cfg, prepared, output)


if __name__ == "__main__":
    main()
