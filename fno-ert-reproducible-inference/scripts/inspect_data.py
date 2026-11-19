import h5py, numpy as np, sys
for f in sys.argv[1:]:
    print("=====", f)
    with h5py.File(f, 'r') as h:
        print("keys:", list(h.keys()))
        for k in ['coeff', 'II', 'sol']:
            raw = np.array(h[k])
            a = np.transpose(raw, tuple(range(raw.ndim - 1, -1, -1)))
            print(f"{k}: file shape {raw.shape} -> per-sample layout {a.shape}, "
                  f"min {a.min():.4g} max {a.max():.4g} mean {a.mean():.4g} NaNs {np.isnan(a).sum()}")
        II = np.transpose(np.array(h['II']), (2, 1, 0))
        print("II unique values (first 10):", np.unique(II)[:10], "| nonzero pixels in sample 0:", int((II[0] != 0).sum()))
