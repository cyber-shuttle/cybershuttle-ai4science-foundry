"""Run the NVlabs FourCastNet inference script on the ERA5 file built by fourcastnet_prefetch.py."""
import argparse
import json
import os
import subprocess
import sys


def write_config(path, nsteps, init_time):
    """Copy the NVlabs config and add a section that inherits afno_backbone, with our data and dates."""
    repo, weights, data = os.environ["FCN_REPO_DIR"], os.environ["WEIGHTS_DIR"], os.environ["DATA_DIR"]
    overrides = {
        "inf_data_path": os.path.join(data, "out_of_sample"),
        "global_means_path": os.path.join(weights, "stats_v0/global_means.npy"),
        "global_stds_path": os.path.join(weights, "stats_v0/global_stds.npy"),
        "time_means_path": os.path.join(weights, "stats_v0/time_means.npy"),
        "prediction_length": nsteps + 1,  # lead 0 plus nsteps forecasts
        "ics_type": "datetime",
        "date_strings": [init_time.replace("T", " ")],
        "log_to_wandb": False,
    }
    with open(os.path.join(repo, "config/AFNO.yaml")) as f:
        text = f.read()
    text += "\npace_inference:\n  <<: *backbone\n"
    text += "".join(f"  {k}: {json.dumps(v)}\n" for k, v in overrides.items())
    with open(path, "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=int(os.environ["NSTEPS"]))
    parser.add_argument("--out", required=True, help="output directory (in scratch)")
    args = parser.parse_args()

    # One run per init time: the NVlabs --vis output only supports a single initial condition
    for init_time in os.environ["INIT_TIMES"].split():
        out = os.path.join(os.path.abspath(args.out), init_time[:13])
        os.makedirs(out, exist_ok=True)
        config = os.path.join(out, "config.yaml")
        write_config(config, args.nsteps, init_time)
        subprocess.run(
            [sys.executable, "inference/inference.py",
             "--yaml_config", config, "--config", "pace_inference",
             "--override_dir", out,
             "--weights", os.path.join(os.environ["WEIGHTS_DIR"], "backbone.ckpt"),
             "--vis"],  # also save the full forecast and truth fields, for plots
            cwd=os.environ["FCN_REPO_DIR"], check=True,
        )


if __name__ == "__main__":
    main()
