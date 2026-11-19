import os, sys
import numpy as np, h5py, torch, yaml

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def setup_demo(cfg):
    """Import the authors' model definition (FNO_ERT/fno_demo_min.py), unmodified."""
    d = cfg['data']['fno_ert_dir']
    if d not in sys.path:
        sys.path.insert(0, d)
    import fno_demo_min as demo
    return demo

def read_field(path, key):
    # MATLAB v7.3 (HDF5) stores dims reversed: file (W,H,N) -> (N,H,W)
    with h5py.File(path, 'r') as h:
        a = np.array(h[key])
    a = np.transpose(a, tuple(range(a.ndim - 1, -1, -1)))
    return torch.from_numpy(a.astype(np.float32))

def load_fields(path):
    coeff = 1.0 / read_field(path, 'coeff')   # demo inverts coeff
    return coeff, read_field(path, 'II'), read_field(path, 'sol')

class Norm:
    """Per-pixel unit-Gaussian normalizer (same math as utilities3.UnitGaussianNormalizer)."""
    def __init__(self, x=None, eps=1e-5):
        self.eps = eps
        if x is not None:
            self.mean = x.mean(0)
            self.std = x.std(0)
    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)
    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)
    def state(self):
        return {'mean': self.mean.cpu(), 'std': self.std.cpu(), 'eps': self.eps}
    @classmethod
    def from_state(cls, s):
        n = cls(eps=s['eps']); n.mean = s['mean']; n.std = s['std']
        return n

def make_inputs(coeff, II, n_coeff, n_II):
    return torch.stack([n_II.encode(II), n_coeff.encode(coeff)], dim=-1)  # (B,H,W,2), order as in demo

def build_model(cfg, demo):
    m = cfg['model']
    return demo.FNO2d(m['modes1'], m['modes2'], m['width'])

def rel_l2(pred, true):
    b = pred.shape[0]
    return torch.linalg.vector_norm((pred - true).reshape(b, -1), dim=1) / \
           torch.linalg.vector_norm(true.reshape(b, -1), dim=1)

@torch.no_grad()
def predict(model, x, n_sol, dev, bs=50):
    model.eval(); outs = []
    for i in range(0, x.shape[0], bs):
        out = model(x[i:i+bs].to(dev)).squeeze(-1)
        outs.append(n_sol.decode(out).cpu())
    return torch.cat(outs)
