#!/usr/bin/env python3
"""
Patches config/AFNO.yaml (inside the cloned NVlabs/FourCastNet repo) so the
`afno_backbone` section points at the downloaded bundle, instead of
hand-editing YAML. It also sets prediction_length to match the evaluation
file, because the repo's default (41 steps) is longer than the 40-step demo
file and would run off the end of the data.

The patch is applied to the clone in place; the job script saves
`git diff config/AFNO.yaml` into logs/ so the exact change is on record.

Usage:
    python scripts/configure_paths.py \
        --fourcastnet-repo ~/FourCastNet \
        --artifact-root ~/scratch/fcn-artifacts
"""
import argparse
import os
import sys

try:
    import h5py
    from ruamel.yaml import YAML
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Activate the fourcastnet-inference conda env first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourcastnet-repo", required=True, help="Path to cloned NVlabs/FourCastNet repo")
    ap.add_argument("--artifact-root", required=True,
                    help="Directory containing the extracted ccai_demo/ folder")
    ap.add_argument("--prediction-length", type=int, default=None,
                    help="Number of states in the rollout, including the initial one. "
                         "Default: every time step in the evaluation file.")
    args = ap.parse_args()

    config_path = os.path.join(args.fourcastnet_repo, "config", "AFNO.yaml")
    bundle = os.path.join(args.artifact_root, "ccai_demo")
    stats = os.path.join(bundle, "additional", "stats_v0")
    inf_dir = os.path.join(bundle, "data", "FCN_ERA5_data_v0", "out_of_sample")
    data_file = os.path.join(inf_dir, "2018.h5")
    weights = os.path.join(bundle, "model_weights", "FCN_weights_v0", "backbone.ckpt")

    required = [config_path, weights, data_file] + [
        os.path.join(stats, f) for f in ("time_means.npy", "global_means.npy", "global_stds.npy")
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        sys.exit("Missing required file(s):\n  " + "\n  ".join(missing)
                 + "\nRun scripts/download_artifacts.sh first.")

    with h5py.File(data_file, "r") as f:
        n_steps, n_ch, n_lat, n_lon = f["fields"].shape
    print(f"Evaluation file shape: steps={n_steps}, channels={n_ch}, lat={n_lat}, lon={n_lon}")
    if n_ch < 20 or n_lat < 720 or n_lon != 1440:
        sys.exit("Unexpected data shape: the model reads the first 20 channels on a 720x1440 grid.")

    pred_len = args.prediction_length or n_steps
    if pred_len > n_steps:
        sys.exit(f"--prediction-length {pred_len} exceeds the {n_steps} steps in the data file.")

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as f:
        cfg = yaml.load(f)

    backbone = cfg["afno_backbone"]
    backbone["orography"] = False
    backbone["orography_path"] = None
    backbone["inf_data_path"] = inf_dir
    backbone["time_means_path"] = os.path.join(stats, "time_means.npy")
    backbone["global_means_path"] = os.path.join(stats, "global_means.npy")
    backbone["global_stds_path"] = os.path.join(stats, "global_stds.npy")
    backbone["prediction_length"] = pred_len
    backbone["log_to_wandb"] = False

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Patched {config_path}:")
    for k in ("inf_data_path", "time_means_path", "global_means_path", "global_stds_path",
              "prediction_length", "log_to_wandb"):
        print(f"  {k} = {backbone[k]}")
    print(f"Checkpoint to pass to inference.py --weights: {weights}")


if __name__ == "__main__":
    main()
