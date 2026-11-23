"""Validate a completed HyenaDNA training run.

With `wandb=null` (the flag the quick-start command uses to avoid requiring
a wandb account), the repo's train.py persists no metrics file at all —
PyTorch Lightning's progress-bar output is the only record of how training
went. So unlike hyenadna_validate.py (inference), this script has to parse
the captured stdout log rather than load a clean metrics file, in addition
to checking the checkpoint it produced.

Smoke test: a checkpoint file exists and loads as a non-empty, finite state
dict (catches a run that crashed before writing anything, or wrote a
corrupt/empty checkpoint).

Validation check: training loss actually decreased from the first to the
last logged step, and the final validation accuracy is a finite number in
[0, 1] (catches a run that executed every step without the model learning
anything, e.g. a frozen or disconnected parameter).

Usage:
    python hyenadna_train_validate.py --log-file train.log --checkpoint-dir outputs/.../checkpoints
"""

import argparse
import os
import re
import sys

import torch

# Every progress-bar refresh re-prints the line; a line can appear partway
# through a refresh, so the last MATCH in the file -- not the last line --
# is what reflects the final state. Must anchor on "train/loss=" specifically:
# a bare "loss=([0-9.eE+-]+)" pattern also matches inside "val/loss=..." and
# "train/loss=...", silently mixing three different metrics into one series
# (confirmed against a real log line: it pulled out the running loss, the
# val loss, AND the train loss as if they were the same quantity over time).
LOSS_RE = re.compile(r"train/loss=([0-9.eE+-]+)")
VAL_ACC_RE = re.compile(r"val/accuracy=([0-9.eE+-]+)")


def find_checkpoint(checkpoint_dir):
    for name in ("last.ckpt",):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            return path
    # save_top_k's filename is literally the monitored metric name (e.g.
    # "val/accuracy"), which Lightning treats as a subdirectory separator.
    for root, _, files in os.walk(checkpoint_dir):
        for f in files:
            if f.endswith(".ckpt"):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, help="Captured stdout from the training run")
    parser.add_argument("--checkpoint-dir", required=True, help="The run's checkpoints/ directory")
    args = parser.parse_args()

    failures = []

    # --- smoke test: a usable checkpoint exists ---------------------------
    ckpt_path = find_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        failures.append(f"no .ckpt file found under {args.checkpoint_dir}")
        print(f"[smoke test] no checkpoint found under {args.checkpoint_dir}")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", {})
        n_params = len(state_dict)
        any_nonfinite = any(not torch.isfinite(v).all() for v in state_dict.values() if torch.is_tensor(v))
        print(f"[smoke test] checkpoint: {ckpt_path} ({n_params} tensors, "
              f"epoch={ckpt.get('epoch')}, global_step={ckpt.get('global_step')})")
        if n_params == 0:
            failures.append("checkpoint state_dict is empty")
        if any_nonfinite:
            failures.append("checkpoint contains NaN/Inf weights")

    # --- validation: loss decreased, accuracy is sane ----------------------
    with open(args.log_file, errors="replace") as f:
        log_text = f.read()

    losses = [float(m) for m in LOSS_RE.findall(log_text)]
    val_accs = [float(m) for m in VAL_ACC_RE.findall(log_text)]

    if len(losses) < 2:
        failures.append(f"found only {len(losses)} loss values in the log; expected many logged steps")
    else:
        first_loss, last_loss = losses[0], losses[-1]
        print(f"[validation] loss: first={first_loss:.4f} last={last_loss:.4f} "
              f"(n={len(losses)} logged steps)")
        if not (last_loss < first_loss):
            failures.append(f"loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}")

    if not val_accs:
        failures.append("no val/accuracy values found in the log")
    else:
        final_val_acc = val_accs[-1]
        print(f"[validation] final val/accuracy: {final_val_acc:.4f}")
        if not (0.0 <= final_val_acc <= 1.0):
            failures.append(f"val/accuracy out of [0, 1] range: {final_val_acc}")

    if failures:
        print("\nVALIDATION FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
