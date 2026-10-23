import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fno_ert.yaml')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--out', default='outputs/fno_ert.pt')
    a = ap.parse_args()
    cfg = load_cfg(a.config); t = cfg['train']
    if a.epochs: t['epochs'] = a.epochs
    demo = setup_demo(cfg)
    torch.manual_seed(t['seed']); np.random.seed(t['seed'])
    os.makedirs('outputs', exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    print('device:', dev, gpu, '| torch', torch.__version__, flush=True)

    coeff, II, sol = load_fields(cfg['data']['eval_path'])
    n = coeff.shape[0]; s1, s2 = coeff.shape[1], coeff.shape[2]
    g = torch.Generator().manual_seed(cfg['data']['split_seed'])
    perm = torch.randperm(n, generator=g)
    nval = int(round(cfg['data']['val_frac'] * n))
    vi, ti = perm[:nval], perm[nval:]
    np.save('outputs/split_train_idx.npy', ti.numpy()); np.save('outputs/split_val_idx.npy', vi.numpy())
    n_coeff, n_II, n_sol = Norm(coeff[ti]), Norm(II[ti]), Norm(sol[ti])
    xtr = make_inputs(coeff[ti], II[ti], n_coeff, n_II); ytr = sol[ti]
    xva = make_inputs(coeff[vi], II[vi], n_coeff, n_II); yva = sol[vi]
    print(f'samples: total {n}, train {len(ti)}, val {len(vi)}, grid {s1}x{s2}', flush=True)

    model = build_model(cfg, demo).to(dev)
    nparams = demo.count_params(model)
    print('parameters (utilities3.count_params):', nparams, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr, ytr),
                                         batch_size=t['batch_size'], shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t['epochs'] * len(loader)))

    best, best_ep, hist = float('inf'), -1, []
    t0 = time.time()
    for ep in range(t['epochs']):
        model.train(); t1 = time.time(); tl = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = n_sol.decode(model(x).squeeze(-1))
            loss = rel_l2(pred, y).mean()
            loss.backward(); opt.step(); sched.step()
            tl += loss.item() * x.size(0)
        tl /= len(xtr)
        vl = rel_l2(predict(model, xva, n_sol, dev), yva).mean().item()
        hist.append({'epoch': ep, 'train_rel_l2': tl, 'val_rel_l2': vl})
        flag = ''
        if vl < best:
            best, best_ep, flag = vl, ep, ' *best'
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'shape': (s1, s2),
                        'norm': {'coeff': n_coeff.state(), 'II': n_II.state(), 'sol': n_sol.state()},
                        'best_epoch': ep, 'best_val_rel_l2': vl}, a.out)
        print(f'epoch {ep:03d} | {time.time()-t1:.2f}s | train_rel_l2 {tl:.4e} | val_rel_l2 {vl:.4e}{flag}', flush=True)

    summary = {'device': gpu, 'torch': torch.__version__, 'params': int(nparams), 'epochs': t['epochs'],
               'train_samples': len(ti), 'val_samples': len(vi), 'best_epoch': best_ep,
               'best_val_rel_l2': best, 'train_time_s': time.time() - t0, 'checkpoint': a.out}
    json.dump(summary, open('outputs/train_summary.json', 'w'), indent=2)
    json.dump(hist, open('outputs/train_history.json', 'w'))
    print('SUMMARY', summary, flush=True)

if __name__ == '__main__':
    main()
