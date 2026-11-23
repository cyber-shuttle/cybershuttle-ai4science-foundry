"""Check a FourCastNet run: smoke checks, then skill vs ERA5 and persistence. Writes results to outputs/."""
import argparse
import glob
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNELS = {"z500": 14, "t850": 5, "t2m": 2, "u10": 0}  # index in the model's 20 channels
RANGES = {"z500": (4.5e4, 6.0e4), "t850": (220, 320), "t2m": (180, 340), "u10": (-80, 80)}
SKILL_VARS = ["z500", "t850"]  # must beat persistence (the paper's headline variables)


def lat_weights(n=720):
    """cos(latitude) weights, same formula as NVlabs utils/weighted_acc_rmse.py."""
    w = np.cos(np.deg2rad(90 - np.arange(n) * 180 / (n - 1)))
    return (w / w.mean())[:, None]


def rmse(a, b, w):
    return float(np.sqrt((w * (a - b) ** 2).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="output of fourcastnet_inference.py, one folder per init time")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # NVlabs saves predicted/ground_truth normalized; convert back with the model's stats
    stats = os.path.join(os.environ["WEIGHTS_DIR"], "stats_v0")
    means = np.load(os.path.join(stats, "global_means.npy")).reshape(-1)
    stds = np.load(os.path.join(stats, "global_stds.npy")).reshape(-1)

    files = sorted(glob.glob(os.path.join(args.run_dir, "*", "autoregressive_predictions_z500_vis.h5")))
    w = lat_weights()
    checks, model, pers, acc = [], {v: [] for v in CHANNELS}, {v: [] for v in CHANNELS}, {v: [] for v in CHANNELS}

    for path in files:
        init = os.path.basename(os.path.dirname(path))
        with h5py.File(path, "r") as f:
            pred, truth = f["predicted"], f["ground_truth"]
            n_lead = pred.shape[1]
            checks.append((pred.shape[2:] == (20, 720, 1440), f"{init}: output shape {pred.shape}"))
            for var, c in CHANNELS.items():
                p = pred[0, :, c] * stds[c] + means[c]
                t = truth[0, :, c] * stds[c] + means[c]
                lo, hi = RANGES[var]
                ok = bool(np.isfinite(p).all() and lo <= p.min() and p.max() <= hi)
                checks.append((ok, f"{init}: {var} finite, range {p.min():.1f} to {p.max():.1f} (expected {lo:g} to {hi:g})"))
                model[var].append(f["rmse"][0, :, c])
                acc[var].append(f["acc"][0, :, c])
                pers[var].append([rmse(t[0], t[k], w) for k in range(n_lead)])  # persistence: start state held fixed
            if path == files[0]:
                k = min(12, n_lead - 1)  # 72 h, or the last lead for short runs
                c = CHANNELS["z500"]
                map_data = (init, 6 * k, pred[0, k, c] * stds[c] + means[c], truth[0, k, c] * stds[c] + means[c])

    hours = 6 * np.arange(n_lead)
    model = {v: np.mean(x, axis=0) for v, x in model.items()}
    pers = {v: np.mean(x, axis=0) for v, x in pers.items()}
    acc = {v: np.mean(x, axis=0) for v, x in acc.items()}

    for var in CHANNELS:
        checks.append((model[var][-1] > model[var][1], f"{var}: error grows with lead time ({model[var][1]:.2f} at {hours[1]} h -> {model[var][-1]:.2f} at {hours[-1]} h)"))
    for var in SKILL_VARS:
        worse = [int(h) for h, m, p in zip(hours[1:], model[var][1:], pers[var][1:]) if m >= p]
        checks.append((not worse, f"{var}: FourCastNet beats persistence at every lead" + (f" (fails at {worse} h)" if worse else "")))

    lines = [f"Run: {args.run_dir}   init times: {len(files)}   leads: 0-{hours[-1]} h", ""]
    lines += [f"{'PASS' if ok else 'FAIL'}  {msg}" for ok, msg in checks]
    lines += ["", "RMSE (lat-weighted, averaged over init times) and ACC:", f"{'var':6s} {'lead':>5s} {'FourCastNet':>12s} {'persistence':>12s} {'ACC':>7s}"]
    for var in CHANNELS:
        for i, h in enumerate(hours):
            if h in (6, 12, 24, 48, 72, 96, 120):
                lines.append(f"{var:6s} {h:4d}h {model[var][i]:12.3f} {pers[var][i]:12.3f} {acc[var][i]:7.4f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "validation.txt"), "w") as f:
        f.write(text + "\n")

    fig, axes = plt.subplots(1, len(SKILL_VARS), figsize=(10, 4))
    for ax, var in zip(axes, SKILL_VARS):
        ax.plot(hours, model[var], "o-", label="FourCastNet")
        ax.plot(hours, pers[var], "s--", label="persistence")
        ax.set(title=f"{var} RMSE", xlabel="lead time (h)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rmse_vs_lead.png"), dpi=120)

    init, h, p, t = map_data
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, data, title, cmap in zip(axes, [p, t, p - t], ["FourCastNet", "ERA5", "difference"], ["viridis", "viridis", "RdBu_r"]):
        im = ax.imshow(data, cmap=cmap, extent=[0, 360, -90, 90])
        ax.set_title(f"z500 {title}, {init} +{h} h")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "z500_map.png"), dpi=120)

    if not all(ok for ok, _ in checks):
        raise SystemExit("validation FAILED")


if __name__ == "__main__":
    main()
