import argparse, json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='outputs/fno_ert.pt')
ap.add_argument('--pred', default='outputs/predictions.npz')
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
cfg = ck['cfg']
pred = torch.from_numpy(np.load(a.pred)['pred'])
_, _, true = load_fields(cfg['data']['test_path'])
assert pred.shape == true.shape, (pred.shape, true.shape)
n = true.shape[0]

err = rel_l2(pred, true) * 100
base = ck['norm']['sol']['mean'].unsqueeze(0).expand_as(true)
err_base = rel_l2(base, true) * 100
nmae = ((pred - true).abs().reshape(n, -1).mean(1) / true.abs().reshape(n, -1).amax(1)) * 100

m = {
  'test_samples': n,
  'model_rel_l2_pct': {'mean': err.mean().item(), 'std': err.std().item(), 'median': err.median().item(),
                       'p95': err.quantile(0.95).item(), 'max': err.max().item()},
  'frac_samples_below_5pct': (err < 5).float().mean().item(),
  'frac_samples_below_10pct': (err < 10).float().mean().item(),
  'mean_baseline_rel_l2_pct': err_base.mean().item(),
  'model_vs_baseline_error_ratio': (err.mean() / err_base.mean()).item(),
  'normalized_mae_pct_of_peak': nmae.mean().item(),
  'paper_reported': {'mean_rel_error_pct_below': 0.3, 'std_pct_below': 5.0,
                     'note': 'paper metric definition and test set differ (49,500 solutions); not directly comparable'},
}
checks = {
  'predictions finite': bool(torch.isfinite(pred).all()),
  'shape matches ground truth': tuple(pred.shape) == tuple(true.shape),
  'model error < 0.5 x trivial mean-field baseline (criterion set before running)': m['model_vs_baseline_error_ratio'] < 0.5,
}
m['checks'] = checks
m['paper_target_met'] = bool(err.mean().item() < 0.3)
json.dump(m, open('outputs/metrics.json', 'w'), indent=2)

print('| metric | expected (paper) | observed |')
print('|---|---|---|')
print(f"| mean rel. L2 error | < 0.3 % | {m['model_rel_l2_pct']['mean']:.2f} % |")
print(f"| std of rel. error | < 5 % | {m['model_rel_l2_pct']['std']:.2f} % |")
print(f"| median / p95 / max | n/a | {m['model_rel_l2_pct']['median']:.2f} / {m['model_rel_l2_pct']['p95']:.2f} / {m['model_rel_l2_pct']['max']:.2f} % |")
print(f"| trivial mean-field baseline | n/a | {m['mean_baseline_rel_l2_pct']:.2f} % |")
for k, v in checks.items(): print(('PASS' if v else 'FAIL'), '-', k)
print('Paper target met:', m['paper_target_met'])

order = err.argsort()
picks = [('best', order[0].item()), ('median', order[n // 2].item()), ('worst', order[-1].item())]
fig, ax = plt.subplots(3, 3, figsize=(12, 9))
for r, (name, i) in enumerate(picks):
    for c, (title, im) in enumerate([('truth', true[i]), ('prediction', pred[i]), ('abs difference', (pred[i] - true[i]).abs())]):
        h = ax[r, c].imshow(im.numpy(), aspect='auto'); plt.colorbar(h, ax=ax[r, c])
        ax[r, c].set_title(f'{name} (sample {i}, {err[i]:.2f}%): {title}', fontsize=8)
plt.tight_layout(); plt.savefig('outputs/example_predictions.png', dpi=120); plt.close()
plt.figure(figsize=(6, 4)); plt.hist(err.numpy(), bins=40); plt.axvline(0.3, color='r', ls='--', label='paper mean target 0.3%')
plt.xlabel('per-sample relative L2 error (%)'); plt.ylabel('count'); plt.legend(); plt.tight_layout()
plt.savefig('outputs/error_hist.png', dpi=120)
