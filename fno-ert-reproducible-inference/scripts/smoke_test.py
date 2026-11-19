import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='outputs/fno_ert.pt')
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
cfg = ck['cfg']; demo = setup_demo(cfg)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
print('checkpoint:', a.ckpt, '| best_epoch', ck['best_epoch'], '| best_val_rel_l2', ck['best_val_rel_l2'])
model = build_model(cfg, demo); model.load_state_dict(ck['model']); model.to(dev).eval()
nc, nI, ns = (Norm.from_state(ck['norm'][k]) for k in ('coeff', 'II', 'sol'))
coeff, II, sol = load_fields(cfg['data']['test_path'])
x = make_inputs(coeff[:1], II[:1], nc, nI)
s1, s2 = ck['shape']
t0 = time.time()
with torch.no_grad():
    out = model(x.to(dev))
    pred = ns.decode(out.squeeze(-1)).cpu()
print(f'forward pass on 1 test sample: {time.time()-t0:.3f}s')
checks = {
    'output shape == (1, %d, %d, 1)' % (s1, s2): tuple(out.shape) == (1, s1, s2, 1),
    'output is finite (no NaN/Inf)': bool(torch.isfinite(out).all() and torch.isfinite(pred).all()),
    'decoded prediction has same shape as ground truth': tuple(pred.shape) == tuple(sol[:1].shape),
    'relative L2 error on this sample is < 1.0': bool(rel_l2(pred, sol[:1]).item() < 1.0),
}
for k, v in checks.items():
    print(('PASS' if v else 'FAIL'), '-', k)
print('sample rel L2 error:', rel_l2(pred, sol[:1]).item())
sys.exit(0 if all(checks.values()) else 1)
