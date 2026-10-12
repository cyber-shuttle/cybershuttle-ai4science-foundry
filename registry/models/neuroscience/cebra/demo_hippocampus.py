#!/usr/bin/env python
# I basically just copied cells 1-30 from the official CEBRA notebook to make it easier to run.
# Notebook: https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html
# Run:      python demo_hippocampus.py --out-dir outputs/run1
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
 
import numpy as np
import matplotlib
 
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
 
T0 = time.time()
 
 
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)
 
 
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
 
 
# helpers
def save_embedding_plot(emb, color, title, path):
    # saves the notebook's plotly embedding plots as PNG + interactive HTML
    import cebra
    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    cebra.plot_embedding(emb, embedding_labels=color, ax=ax, cmap="rainbow", markersize=2, title=title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    try:  # interactive copy, if plotly is available
        import cebra.integrations.plotly
        f = cebra.integrations.plotly.plot_embedding_interactive(
            emb, embedding_labels=color, title=title, markersize=2, cmap="rainbow")
        f.write_html(str(path).replace(".png", ".html"))
    except Exception:
        pass
 
 
def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    # kNN decoder (same as the notebook's cell 61), used here to validate the embeddings
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    import sklearn.metrics
    pos_decoder = KNeighborsRegressor(n_neighbors, metric="cosine")
    dir_decoder = KNeighborsClassifier(n_neighbors, metric="cosine")
    pos_decoder.fit(emb_train, label_train[:, 0])
    dir_decoder.fit(emb_train, label_train[:, 1])
    prediction = np.stack([pos_decoder.predict(emb_test), dir_decoder.predict(emb_test)], axis=1)
    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])
    return test_score, pos_test_err, pos_test_score, prediction
 
 
# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CEBRA hippocampus demo, notebook cells 1-30")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-iterations", type=int, default=5000, help="notebook cell 11")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda_if_available")
    ap.add_argument("--infonce-batches", type=int, default=500,
                    help="batches for goodness of fit (notebook uses 5; more = less noisy)")
    args = ap.parse_args()
 
    out = Path(args.out_dir)
    figs, ckdir = out / "figures", out / "checkpoints"
    for d in (out, figs, ckdir):
        d.mkdir(parents=True, exist_ok=True)
 
    import torch
    import sklearn
    import cebra
    import cebra.datasets
    import cebra.integrations.sklearn.metrics  # noqa: F401
    from cebra import CEBRA
 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    metrics, checkpoints, timings = {}, {}, {}
 
    def make(output_dimension):
        # notebook cell 12
        return CEBRA(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                     temperature=1, output_dimension=output_dimension,
                     max_iterations=args.max_iterations, distance="cosine",
                     conditional="time_delta", device=args.device, verbose=True, time_offsets=10)
 
    def fit_save_load(name, model, *fit_args):
        # fit, save checkpoint, reload from disk
        log(f"fit {name}")
        t = time.time()
        model.fit(*fit_args)
        timings[f"fit_{name}"] = round(time.time() - t, 2)
        path = ckdir / f"{name}.pt"
        model.save(str(path))
        checkpoints[name] = {"path": str(path), "sha256": sha256(path)}
        loaded = cebra.CEBRA.load(str(path))
        return loaded, np.asarray(model.state_dict_["loss"])
 
    def infonce(model, X, y):
        try:
            return float(cebra.sklearn.metrics.infonce_loss(model, X, y, num_batches=args.infonce_batches))
        except Exception as e:
            log(f"infonce_loss failed: {e}")
            return None
 
    log(f"cebra {cebra.__version__} torch {torch.__version__} CUDA={cuda} "
        f"GPU={torch.cuda.get_device_name(0) if cuda else None}")
 
    # ---- cells 6-8: load + visualize data ---------------------------------------------
    hippocampus_pos = cebra.datasets.init("rat-hippocampus-single-achilles")
    neural = hippocampus_pos.neural.numpy()
    label = hippocampus_pos.continuous_index.numpy()
    log(f"data: neural {neural.shape}, labels {label.shape}")
 
    fig = plt.figure(figsize=(9, 3), dpi=150)
    plt.subplots_adjust(wspace=0.3)
    ax = plt.subplot(121)
    ax.imshow(neural[:1000].T, aspect="auto", cmap="gray_r")
    plt.ylabel("Neuron #"); plt.xlabel("Time [s]")
    plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025 * 1000, 5, dtype=int))
    ax2 = plt.subplot(122)
    ax2.scatter(np.arange(1000), label[:1000, 0], c="gray", s=1)
    plt.ylabel("Position [m]"); plt.xlabel("Time [s]")
    plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025 * 1000, 5, dtype=int))
    fig.savefig(figs / "01_data_raster_and_position.png", bbox_inches="tight"); plt.close(fig)
 
    # ---- cells 11-22: CEBRA-Behavior 3D on 80/20 split -----------------------------------
    split_idx = int(0.8 * len(neural))
    train_data, valid_data = neural[:split_idx], neural[split_idx:]
    train_lbl, valid_lbl = label[:split_idx], label[split_idx:]
 
    m_posdir3_split, loss = fit_save_load("cebra_posdir3_split", make(3), train_data, train_lbl)
    t = time.time()
    train_embedding = m_posdir3_split.transform(train_data)
    valid_embedding = m_posdir3_split.transform(valid_data)
    timings["inference_posdir3_split"] = round(time.time() - t, 3)
    gof = infonce(m_posdir3_split, train_data, train_lbl)
    log(f"goodness of fit - train: {gof}")
    save_embedding_plot(train_embedding, train_lbl[:, 0], "CEBRA-Behavior Train", figs / "02_behavior_train.png")
    save_embedding_plot(valid_embedding, valid_lbl[:, 0], "CEBRA-Behavior-validation", figs / "03_behavior_validation.png")
    _, err, r2, pred = decoding_pos_dir(train_embedding, valid_embedding, train_lbl, valid_lbl)
    metrics["cebra_posdir"] = {"infonce_train": gof, "pos_median_abs_err_m": float(err), "pos_r2": float(r2)}
 
    # Files that validate.py checks (the 3-D split model is the validated inference artifact)
    import shutil
    shutil.copy(ckdir / "cebra_posdir3_split.pt", out / "cebra_posdir.pt")
    np.save(out / "embedding_train.npy", train_embedding)
    np.save(out / "embedding_test.npy", valid_embedding)
    np.save(out / "pred_test.npy", pred)
    np.save(out / "labels_test.npy", valid_lbl)
    np.save(out / "train_loss.npy", loss)
 
    # ---- cells 23-30: shuffle control on the split ---------------------------------------
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(label)
    sh_train_lbl, sh_valid_lbl = shuffled[:split_idx], shuffled[split_idx:]
    m_sh_split, _ = fit_save_load("cebra_posdir3_shuffled_split", make(3), train_data, sh_train_lbl)
    sh_train_emb = m_sh_split.transform(train_data)
    sh_valid_emb = m_sh_split.transform(valid_data)
    gof_sh = infonce(m_sh_split, train_data, sh_train_lbl)  # as in cell 28: with shuffled labels
    log(f"goodness of fit - shuffle train: {gof_sh}")
    save_embedding_plot(sh_train_emb, sh_train_lbl[:, 0], "CEBRA-Behavior Shuffle Train", figs / "04_shuffle_train.png")
    save_embedding_plot(sh_valid_emb, sh_valid_lbl[:, 0], "CEBRA-Behavior Shuffle Validation", figs / "05_shuffle_validation.png")
    _, err_s, r2_s, _ = decoding_pos_dir(sh_train_emb, sh_valid_emb, train_lbl, valid_lbl)  # vs TRUE labels
    metrics["shuffled_control"] = {"infonce_train": gof_sh, "pos_median_abs_err_m": float(err_s), "pos_r2": float(r2_s)}
 
    # ---- write metrics + run info ---------------------------------------------------------
    timings["total"] = round(time.time() - T0, 2)
    run_info = {
        "notebook": "https://cebra.ai/docs/demo_notebooks/Demo_hippocampus.html",
        "args": {**vars(args), "dataset": "rat-hippocampus-single-achilles", "output_dimension": 3},
        "sections": "cells 0-30 (experiment 1 + shuffle control)",
        "python": sys.version.split()[0], "platform": platform.platform(), "hostname": platform.node(),
        "versions": {"cebra": cebra.__version__, "torch": torch.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "cuda_available": cuda, "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "data": {"n_timesteps": int(len(neural)), "n_neurons": int(neural.shape[1]),
                 "n_train": int(split_idx), "n_test": int(len(neural) - split_idx),
                 "label_columns": ["position_m", "right_running", "left_running"]},
        "checkpoint": {"path": str(out / "cebra_posdir.pt"), "sha256": sha256(out / "cebra_posdir.pt")},
        "all_checkpoints": checkpoints,
        "timing_s": timings,
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    json.dump(run_info, open(out / "run_info.json", "w"), indent=2)
    log(f"done: {len(list(figs.glob('*.png')))} figures, {len(checkpoints)} checkpoints -> {out.resolve()}")
 
 
if __name__ == "__main__":
    main()
