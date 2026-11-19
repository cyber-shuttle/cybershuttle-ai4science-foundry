import argparse, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='outputs/fno_ert.pt')
ap.add_argument('--out', default='outputs/predictions.npz')
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
cfg = ck['cfg']; demo = setup_demo(cfg)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
model = build_model(cfg, demo); model.load_state_dict(ck['model']); model.to(dev).eval()
nc, nI, ns = (Norm.from_state(ck['norm'][k]) for k in ('coeff', 'II', 'sol'))
coeff, II, sol = load_fields(cfg['data']['test_path'])
x = make_inputs(coeff, II, nc, nI)
n = x.shape[0]
print(f'device {gpu} | test samples {n} | input {tuple(x.shape)}', flush=True)

predict(model, x[:50], ns, dev)
sync(); t0 = time.time()
pred = predict(model, x, ns, dev)
sync(); total = time.time() - t0
xs = x[:1].to(dev)
with torch.no_grad():
    for _ in range(5): model(xs)
    sync(); t0 = time.time()
    for _ in range(50): model(xs)
    sync(); single = (time.time() - t0) / 50

np.savez_compressed(a.out, pred=pred.numpy())
summary = {'device': gpu, 'torch': torch.__version__, 'test_samples': n,
           'batched_total_s': total, 'batched_ms_per_sample': 1000 * total / n,
           'single_sample_forward_ms': 1000 * single,
           'checkpoint': a.ckpt, 'checkpoint_sha256': hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest(),
           'test_file': cfg['data']['test_path'], 'predictions': a.out}
json.dump(summary, open('outputs/infer_summary.json', 'w'), indent=2)
print('SUMMARY', summary, flush=True)
