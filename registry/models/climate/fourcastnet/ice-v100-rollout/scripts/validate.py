#!/usr/bin/env python3
"""
Validation for a completed FourCastNet backbone inference run.

Run automatically at the end of slurm/run_inference.sbatch, or standalone:

    python scripts/validate.py --run-dir /path/to/output_dir [--plot-out skill_vs_leadtime.png]

Smoke test (structural): output file exists, is HDF5, has `rmse` and `acc`,
arrays are non-empty and finite.

Substantive checks (scientific), on the Z500 channel (index 14 in the 20-channel
ordering used by inference.py):
  1. ACC in [-1, 1] and RMSE >= 0 (definitions; a violation means a pipeline bug,
     most often normalization).
  2. The first state (lead 0) is the initial condition, so ACC there should be ~1.
     This also tells us whether index 0 is the IC or the first forecast step, and
     the script adapts its lead-time labels accordingly.
  3. ACC at 24 h >= 0.6, the conventional NWP "useful forecast" threshold.
  4. Skill degrades with lead time: ACC(24 h) > ACC(last step) and
     RMSE(last step) > RMSE(24 h).

The array layout written by inference.py is not assumed: the channel axis is
found as the axis of size 20 and the other (non-singleton) axis is lead time.
The detected layout is printed so it lands in the job log.
"""
import argparse
import os
import sys

import h5py
import numpy as np

EXPECTED_DATASETS = ["rmse", "acc"]
N_CHANNELS = 20
Z500_CHANNEL = 14
STEP_HOURS = 6
ACC_USEFUL = 0.6
IC_ACC_MIN = 0.99


def find_output(run_dir):
    cands = sorted(f for f in os.listdir(run_dir)
                   if f.startswith("autoregressive_predictions") and f.endswith(".h5"))
    return os.path.join(run_dir, cands[0]) if cands else None


def to_channel_by_lead(arr, name):
    """Return array shaped (channels, lead) from whatever inference.py wrote."""
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 3:  # (n_ics, ., .) with several initial conditions: average over ICs
        a = a.mean(axis=0)
    if a.ndim != 2:
        raise ValueError(f"'{name}' has unexpected shape {np.asarray(arr).shape}")
    if a.shape[0] == N_CHANNELS and a.shape[1] != N_CHANNELS:
        return a
    if a.shape[1] == N_CHANNELS and a.shape[0] != N_CHANNELS:
        return a.T
    raise ValueError(f"cannot identify the channel axis (size {N_CHANNELS}) in '{name}' "
                     f"with shape {np.asarray(arr).shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--z500-channel", type=int, default=Z500_CHANNEL)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--csv-out", default=None,
                    help="Write Z500 lead-time/ACC/RMSE to this small CSV (committed as evidence)")
    args = ap.parse_args()

    # ---- smoke test ----
    path = find_output(args.run_dir)
    if path is None:
        print(f"FAIL (smoke): no autoregressive_predictions*.h5 in {args.run_dir}")
        sys.exit(1)
    print(f"Output file: {path}")
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        print(f"FAIL (smoke): cannot open as HDF5: {e}")
        sys.exit(1)
    print(f"Datasets present: {list(f.keys())}")
    raw = {}
    for k in EXPECTED_DATASETS:
        if k not in f:
            print(f"FAIL (smoke): missing dataset '{k}'")
            sys.exit(1)
        arr = f[k][:]
        if arr.size == 0 or not np.isfinite(arr).all():
            print(f"FAIL (smoke): '{k}' is empty or contains NaN/Inf")
            sys.exit(1)
        raw[k] = arr
        print(f"OK: '{k}' shape={arr.shape} dtype={arr.dtype}, finite")
    print("SMOKE TEST PASSED\n")

    # ---- substantive ----
    try:
        acc_all = to_channel_by_lead(raw["acc"], "acc")
        rmse_all = to_channel_by_lead(raw["rmse"], "rmse")
    except ValueError as e:
        print(f"FAIL (substantive): {e}")
        sys.exit(1)
    print(f"Detected layout -> (channels, lead steps) = {acc_all.shape}")
    acc = acc_all[args.z500_channel]
    rmse = rmse_all[args.z500_channel]
    n = acc.shape[0]

    ok = True

    def check(cond, msg_ok, msg_fail):
        nonlocal ok
        print(("OK: " + msg_ok) if cond else ("FAIL: " + msg_fail))
        ok = ok and bool(cond)

    check(acc_all.min() >= -1.0001 and acc_all.max() <= 1.0001,
          f"ACC (all channels) in [-1, 1] (min={acc_all.min():.4f}, max={acc_all.max():.4f})",
          f"ACC out of [-1, 1] (min={acc_all.min():.4f}, max={acc_all.max():.4f}); suspect normalization/stats mismatch")
    check(rmse_all.min() >= 0, f"RMSE non-negative (min={rmse_all.min():.4f})",
          f"RMSE negative (min={rmse_all.min():.4f})")

    # Is index 0 the initial condition (ACC ~ 1) or the first forecast step?
    offset = 0 if acc[0] >= IC_ACC_MIN else 1
    print(f"Z500 ACC at index 0 = {acc[0]:.4f} -> index 0 treated as "
          f"{'the initial condition (lead 0 h)' if offset == 0 else 'the first forecast step (lead 6 h)'}")
    if offset == 0:
        check(acc[0] >= IC_ACC_MIN, f"ACC at lead 0 h is {acc[0]:.4f} (>= {IC_ACC_MIN})", "")

    def idx(hours):
        return hours // STEP_HOURS - offset

    lead_hours = [(i + offset) * STEP_HOURS for i in range(n)]
    i24 = idx(24)
    if 0 <= i24 < n:
        check(acc[i24] >= ACC_USEFUL,
              f"Z500 ACC at 24 h = {acc[i24]:.4f} >= {ACC_USEFUL} (conventional 'useful forecast' threshold)",
              f"Z500 ACC at 24 h = {acc[i24]:.4f} < {ACC_USEFUL}; check channel index, stats files, checkpoint pairing")
        check(acc[i24] > acc[-1],
              f"ACC decays with lead time ({acc[i24]:.4f} at 24 h -> {acc[-1]:.4f} at {lead_hours[-1]} h)",
              "ACC does not decay with lead time")
        check(rmse[-1] > rmse[i24],
              f"RMSE grows with lead time ({rmse[i24]:.2f} at 24 h -> {rmse[-1]:.2f} at {lead_hours[-1]} h)",
              "RMSE does not grow with lead time")
    else:
        check(False, "", f"run too short to evaluate 24 h skill (only {n} steps)")

    print("\nZ500 skill summary (lead hours: ACC / RMSE):")
    for h in (24, 48, 72, 120, 168):
        j = idx(h)
        if 0 <= j < n:
            print(f"  {h:>4d} h: ACC={acc[j]:.4f}  RMSE={rmse[j]:.2f}")
    print(f"  {lead_hours[-1]:>4d} h (last): ACC={acc[-1]:.4f}  RMSE={rmse[-1]:.2f}")

    below = [i for i in range(n) if acc[i] < ACC_USEFUL]
    if below:
        j = below[0]
        print(f"  Z500 ACC first drops below {ACC_USEFUL} at lead {lead_hours[j]} h "
              f"({lead_hours[j] / 24:.2f} days); last value before: {acc[j - 1]:.4f}" if j > 0 else
              f"  Z500 ACC is already below {ACC_USEFUL} at the first step")
    else:
        print(f"  Z500 ACC stays above {ACC_USEFUL} for the whole {lead_hours[-1]} h run")

    if args.csv_out:
        with open(args.csv_out, "w") as fh:
            fh.write("lead_hours,z500_acc,z500_rmse\n")
            for h, a_, r_ in zip(lead_hours, acc, rmse):
                fh.write(f"{h},{a_:.6f},{r_:.6f}\n")
        print(f"Saved CSV: {args.csv_out}")

    # ---- plot ----
    out_png = args.plot_out or os.path.join(args.run_dir, "skill_vs_leadtime.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(lead_hours, acc, marker="o", ms=3)
        ax[0].axhline(ACC_USEFUL, ls="--", c="gray", label="0.6 useful-skill threshold")
        ax[0].set(title="Z500 ACC vs lead time", xlabel="lead time (hours)", ylabel="ACC")
        ax[0].legend()
        ax[1].plot(lead_hours, rmse, marker="o", ms=3)
        ax[1].set(title="Z500 RMSE vs lead time", xlabel="lead time (hours)", ylabel="RMSE")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved plot: {out_png}")
    except Exception as e:  # plotting is evidence, not a pass/fail criterion
        print(f"WARNING: plot not generated ({e})")

    print("\nVALIDATION PASSED" if ok else "\nVALIDATION FAILED (see docs/TROUBLESHOOTING.md)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
