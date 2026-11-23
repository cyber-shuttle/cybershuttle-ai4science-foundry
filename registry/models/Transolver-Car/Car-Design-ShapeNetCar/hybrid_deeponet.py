"""
Transolver x DeepONet proof of concept on ShapeNet-Car.
Place this file in Car-Design-ShapeNetCar/ and run it from that directory.

Question: how much of Transolver's accuracy can a much lighter DeepONet keep, and at what saving in
inference cost and input requirements (a plain DeepONet only needs the surface point cloud, not the
full volume mesh with SDF/normal features)?

Models (same data, same loss, same metric code, so results are directly comparable):
  transolver : evaluate the trained Transolver checkpoint (teacher / accuracy ceiling, no training)
  deeponet   : lightweight DeepONet. Branch input = flattened, subsampled, x-sorted surface point cloud.
               --distill        : train on the Transolver's predictions instead of ground truth
               --distill_alpha a: target = a * ground truth + (1 - a) * teacher prediction
  hybrid     : DeepONet whose branch input is an embedding from the FROZEN Transolver encoder
               (slice tokens of the last physics-attention block, or mean+max pooled hidden features).
               Ablation: how much does Transolver's learned representation help a DeepONet?

All DeepONet variants share the same trunk (Fourier-feature MLP on the 7-d node input
[pos, sdf, normal]) and the same 4-channel output (3 velocity components + pressure).

--suite runs the whole list of configurations in SUITE inside one process (data loaded once, each
run isolated by try/except, results written per run), so a single GPU allocation covers everything.
"""
import argparse
import copy
import gc
import json
import math
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Priority order: if the allocation runs out of time, the most important results already exist.
SUITE = [
    dict(model='deeponet'),                                              # lightweight baseline (ground truth)
    dict(model='deeponet', distill=True, distill_alpha=0.0),             # lightweight student (teacher targets)
    dict(model='transolver', enc_epochs=200),                            # teacher / ceiling
    dict(model='hybrid', enc_epochs=200, embed='slice'),                 # frozen slice tokens -> DeepONet
    dict(model='deeponet', distill=True, distill_alpha=0.5),             # blended targets
    dict(model='hybrid', enc_epochs=200, embed='meanmax'),               # ablation: pooled features
    dict(model='hybrid', enc_epochs=100, embed='slice'),                 # ablation: encoder quality
]

GT_DRAG_CACHE = {}  # ground-truth drag coefficient per car (identical across runs, VTK is slow)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['transolver', 'deeponet', 'hybrid'], default='deeponet')
    p.add_argument('--suite', action='store_true', help='run every configuration in SUITE in this process')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--fold_id', default=0, type=int)
    p.add_argument('--enc_epochs', default=200, type=int, help='which Transolver checkpoint (100 or 200)')
    p.add_argument('--weight', default=0.5, type=float, help='only used to build the checkpoint path')
    p.add_argument('--embed', default='slice', choices=['slice', 'meanmax'])
    p.add_argument('--distill', action='store_true', help='use Transolver predictions as training targets')
    p.add_argument('--distill_alpha', default=0.0, type=float,
                   help='target = alpha * ground truth + (1 - alpha) * teacher (0 = pure distillation)')
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--batch_size', default=4, type=int, help='cars per optimizer step')
    p.add_argument('--n_vol_query', default=8192, type=int, help='volume query points sampled per car per step')
    p.add_argument('--n_sensors', default=2048, type=int, help='surface points in the plain DeepONet branch')
    p.add_argument('--latent', default=128, type=int)
    p.add_argument('--hidden', default=256, type=int)
    p.add_argument('--reg', default=1.0, type=float, help='pressure loss weight (same role as in train.py)')
    p.add_argument('--val_every', default=10, type=int)
    p.add_argument('--drag', action='store_true', help='also compute drag-coefficient error + Spearman at the end')
    p.add_argument('--cpu_time', action='store_true', help='also time CPU-only inference on a few test cars')
    p.add_argument('--cpu_cars', default=10, type=int)
    p.add_argument('--max_train', default=0, type=int, help='debug: limit number of training cars')
    p.add_argument('--max_val', default=0, type=int, help='debug: limit number of test cars')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--out_dir', default='results_poc')
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- branch inputs
def pc_normalize(pc):
    pc = pc - pc.mean(0)
    return pc / torch.sqrt((pc ** 2).sum(1)).max()


def make_sensors(pos, surf, n, gen):
    """Plain-DeepONet branch input: n surface points (fixed seed), centred/scaled, sorted by x, flattened."""
    pts = pc_normalize(pos[surf].float())
    if len(pts) >= n:
        idx = torch.randperm(len(pts), generator=gen)[:n]
    else:
        idx = torch.randint(len(pts), (n,), generator=gen)
    pts = pts[idx]
    pts = pts[torch.argsort(pts[:, 0])]
    return pts.reshape(-1)


@torch.no_grad()
def encode(model, x, pos, mode):
    """Frozen Transolver encoder -> one global vector per car.
    Replays Model.forward up to (not including) the last block, then either pools the hidden
    features or extracts the slice tokens of the last block's physics attention."""
    x = x[None]
    if model.unified_pos:
        x = torch.cat((x, model.get_grid(pos[None])), dim=-1)
    fx = model.preprocess(x) + model.placeholder[None, None, :]
    for blk in model.blocks[:-1]:
        fx = blk(fx)
    if mode == 'meanmax':
        return torch.cat([fx[0].mean(0), fx[0].max(0).values])
    blk = model.blocks[-1]
    at = blk.Attn
    h = blk.ln_1(fx)
    B, N, _ = h.shape
    fx_mid = at.in_project_fx(h).reshape(B, N, at.heads, at.dim_head).permute(0, 2, 1, 3)
    x_mid = at.in_project_x(h).reshape(B, N, at.heads, at.dim_head).permute(0, 2, 1, 3)
    w = at.softmax(at.in_project_slice(x_mid) / at.temperature)          # B H N G
    tok = torch.einsum('bhnc,bhng->bhgc', fx_mid, w) / (w.sum(2) + 1e-5)[..., None]  # B H G Dh
    return tok[0].permute(1, 0, 2).reshape(-1)                            # G*H*Dh (e.g. 32*256 = 8192)


# ----------------------------------------------------------------------------- DeepONet
class FourierFeatures(nn.Module):
    def __init__(self, d_in, n_feat=64, scale=1.0):
        super().__init__()
        self.register_buffer('B', torch.randn(d_in, n_feat) * scale)
        self.out_dim = d_in + 2 * n_feat

    def forward(self, x):
        proj = 2 * math.pi * x @ self.B
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class DeepONet(nn.Module):
    """out[n, c] = sum_k branch(b)[c, k] * trunk(x_n)[c, k] + bias[c]"""

    def __init__(self, branch_in, trunk_in, latent=128, hidden=256, n_out=4):
        super().__init__()
        self.n_out, self.latent = n_out, latent
        self.branch = nn.Sequential(
            nn.LayerNorm(branch_in), nn.Linear(branch_in, 512), nn.GELU(),
            nn.Linear(512, hidden), nn.GELU(), nn.Linear(hidden, n_out * latent))
        self.ff = FourierFeatures(trunk_in)
        self.trunk = nn.Sequential(
            nn.Linear(self.ff.out_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out * latent))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, b, xq):
        bc = self.branch(b[None]).view(self.n_out, self.latent)
        t = self.trunk(self.ff(xq)).view(-1, self.n_out, self.latent)
        return (t * bc[None]).sum(-1) / math.sqrt(self.latent) + self.bias


# ----------------------------------------------------------------------------- metrics
def car_errors(out, it, mean, std):
    s = it['surf']
    y = it['y']
    pp = out[s, -1] * std[-1] + mean[-1]
    gp = y[s, -1] * std[-1] + mean[-1]
    ps = out[s, :-1] * std[:-1] + mean[:-1]
    gs = y[s, :-1] * std[:-1] + mean[:-1]
    pv = out[~s, :-1] * std[:-1] + mean[:-1]
    gv = y[~s, :-1] * std[:-1] + mean[:-1]
    return pp, gp, ps, gs, pv, gv


@torch.no_grad()
def evaluate(predict, items, vallst, coef_norm, device, drag=False):
    """Same definitions as main_evaluation.py: relative L2 on denormalised surface pressure and
    volume velocity, optional drag-coefficient relative error and Spearman correlation.
    If teacher outputs are present, also reports agreement with the teacher on surface pressure."""
    mean = torch.tensor(coef_norm[2]).to(device)
    std = torch.tensor(coef_norm[3]).to(device)
    l2p, l2v, times, gt_c, pr_c, fid = [], [], [], [], [], []
    coef_err = 0.0
    for i, it in enumerate(items):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        tic = time.time()
        out = predict(it)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - tic)
        pp, gp, ps, gs, pv, gv = car_errors(out, it, mean, std)
        l2p.append((torch.norm(pp - gp) / torch.norm(gp)).item())
        l2v.append((torch.norm(pv - gv) / torch.norm(gv)).item())
        if 'y_teacher' in it:
            tp = it['y_teacher'][it['surf'], -1] * std[-1] + mean[-1]
            fid.append((torch.norm(pp - tp) / torch.norm(tp)).item())
        if drag:
            from utils.drag_coefficient import cal_coefficient
            name = vallst[i].split('/')[1]
            pc = cal_coefficient(name, pp[:, None].cpu().numpy(), ps.cpu().numpy())
            if name not in GT_DRAG_CACHE:
                GT_DRAG_CACHE[name] = cal_coefficient(name, gp[:, None].cpu().numpy(), gs.cpu().numpy())
            gc_ = GT_DRAG_CACHE[name]
            pr_c.append(pc)
            gt_c.append(gc_)
            coef_err += abs(pc - gc_) / gc_
    res = {'rel_l2_press': float(np.mean(l2p)), 'rel_l2_velo': float(np.mean(l2v)),
           'time_per_car_s': float(np.mean(times)), 'n_cars': len(items)}
    if fid:
        res['rel_l2_press_vs_teacher'] = float(np.mean(fid))
    if drag:
        import scipy.stats
        res['c_d_rel_err'] = float(coef_err / len(items))
        res['spearman_rho_d'] = float(scipy.stats.spearmanr(np.array(gt_c), np.array(pr_c))[0])
    return res


def cpu_timing(args, model, head, items_val):
    """Inference time per car on CPU only (copies of the models; includes input construction for
    the plain DeepONet and the encoder pass for the hybrid). First car is dropped as warm-up."""
    try:
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
        cpu = torch.device('cpu')
        m = copy.deepcopy(model).to(cpu).eval() if model is not None else None
        h = copy.deepcopy(head).to(cpu).eval() if head is not None else None
        gen = torch.Generator().manual_seed(args.seed)
        times = []
        with torch.no_grad():
            for it in items_val[:args.cpu_cars]:
                c = {k: v.cpu() for k, v in it.items() if torch.is_tensor(v)}
                tic = time.time()
                if args.model == 'transolver':
                    m((SimpleNamespace(x=c['x'], pos=c['pos']), None))
                elif args.model == 'hybrid':
                    h(encode(m, c['x'], c['pos'], args.embed), c['x'])
                else:
                    h(make_sensors(c['pos'], c['surf'], args.n_sensors, gen), c['x'])
                times.append(time.time() - tic)
        return float(np.mean(times[1:] if len(times) > 1 else times))
    except Exception as e:  # e.g. unified_pos models call .cuda() inside get_grid
        return f'failed: {e!r}'


# ----------------------------------------------------------------------------- data / training
def build_items(data_list, model, args, device):
    gen = torch.Generator().manual_seed(args.seed)
    items = []
    for d in data_list:
        x, y = d.x.to(device), d.y.to(device)
        surf, pos = d.surf.to(device), d.pos.to(device)
        it = {'x': x, 'y': y, 'surf': surf, 'pos': pos,
              'surf_idx': torch.where(surf)[0], 'vol_idx': torch.where(~surf)[0]}
        if args.model == 'deeponet':
            it['branch'] = make_sensors(d.pos, d.surf, args.n_sensors, gen).to(device)
        elif args.model == 'hybrid':
            it['branch'] = encode(model, x, pos, args.embed)
        if args.distill:
            with torch.no_grad():
                it['y_teacher'] = model((SimpleNamespace(x=x, pos=pos), None))
        items.append(it)
    return items


def train_head(head, items_train, items_val, vallst, coef_norm, args, device, log_path):
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    steps = args.epochs * math.ceil(len(items_train) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=args.lr * 1e-3)
    mse = nn.MSELoss()
    history, t0 = [], time.time()
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(items_train)).tolist()
        tot = []
        for s in range(0, len(perm), args.batch_size):
            opt.zero_grad()
            losses = []
            for j in perm[s:s + args.batch_size]:
                it = items_train[j]
                vi = it['vol_idx'][torch.randperm(len(it['vol_idx']), device=device)[:args.n_vol_query]]
                idx = torch.cat([it['surf_idx'], vi])
                ns = len(it['surf_idx'])
                pred = head(it['branch'], it['x'][idx])
                tgt = it['y'][idx]
                if args.distill:
                    tgt = args.distill_alpha * tgt + (1 - args.distill_alpha) * it['y_teacher'][idx]
                l_velo = mse(pred[:, :-1], tgt[:, :-1])
                l_press = mse(pred[:ns, -1], tgt[:ns, -1])
                losses.append(l_velo + args.reg * l_press)
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            sched.step()
            tot.append(loss.item())
        rec = {'epoch': ep + 1, 'train_loss': float(np.mean(tot)), 'elapsed_s': time.time() - t0}
        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            head.eval()
            v = evaluate(lambda it: head(it['branch'], it['x']), items_val, vallst, coef_norm, device)
            rec.update({'val_rel_l2_press': v['rel_l2_press'], 'val_rel_l2_velo': v['rel_l2_velo']})
            if 'rel_l2_press_vs_teacher' in v:
                rec['val_rel_l2_press_vs_teacher'] = v['rel_l2_press_vs_teacher']
        history.append(rec)
        print(json.dumps(rec), flush=True)
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=1)
    return history, time.time() - t0


def get_model(args, device, cache):
    if args.enc_epochs not in cache:
        path = f'metrics/Transolver/{args.fold_id}/{args.enc_epochs}_{args.weight}/model_{args.enc_epochs}.pth'
        print('loading Transolver checkpoint:', path, flush=True)
        m = torch.load(path, weights_only=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        cache[args.enc_epochs] = m
    return cache[args.enc_epochs]


def run_name(args):
    name = args.model
    if args.model == 'hybrid':
        name += f'_{args.embed}_enc{args.enc_epochs}'
    elif args.model == 'transolver':
        name += f'_{args.enc_epochs}'
    if args.distill:
        name += f'_distill{args.distill_alpha:g}_enc{args.enc_epochs}'
    return name + f'_seed{args.seed}'


def run(args, train_data, val_data, coef_norm, vallst, device, model_cache):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.distill and args.model == 'transolver':
        raise ValueError('--distill makes no sense for --model transolver')
    model = None
    if args.model in ('transolver', 'hybrid') or args.distill:
        model = get_model(args, device, model_cache)

    name = run_name(args)
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'--- {name}', flush=True)

    t = time.time()
    items_val = build_items(val_data, model, args, device)
    items_train = build_items(train_data, model, args, device) if args.model != 'transolver' else []
    setup_time = time.time() - t
    print(f'built items (train {len(items_train)}, val {len(items_val)}) in {setup_time:.1f}s', flush=True)

    train_time, head = 0.0, None
    if args.model == 'transolver':
        predict = lambda it: model((SimpleNamespace(x=it['x'], pos=it['pos']), None))
        n_params = sum(p.numel() for p in model.parameters())
        n_frozen = 0
    else:
        ref = items_val[0]
        head = DeepONet(ref['branch'].numel(), ref['x'].shape[1], args.latent, args.hidden,
                        ref['y'].shape[1]).to(device)
        n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in model.parameters()) if args.model == 'hybrid' else 0
        print(f'branch input dim {ref["branch"].numel()}, trainable params {n_params}, frozen {n_frozen}', flush=True)
        _, train_time = train_head(head, items_train, items_val, vallst, coef_norm, args, device,
                                   os.path.join(out_dir, 'train_log.json'))
        head.eval()
        torch.save(head.state_dict(), os.path.join(out_dir, 'head.pt'))
        if args.model == 'hybrid':  # live encoder so timing includes the Transolver forward pass
            predict = lambda it: head(encode(model, it['x'], it['pos'], args.embed), it['x'])
        else:
            predict = lambda it: head(it['branch'], it['x'])

    base_mem = None
    if device.type == 'cuda':
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    res = evaluate(predict, items_val, vallst, coef_norm, device, drag=args.drag)
    if base_mem is not None:
        res['infer_peak_extra_gpu_mem_mb'] = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
    if args.cpu_time:
        res['cpu_time_per_car_s'] = cpu_timing(args, model if args.model != 'deeponet' else None, head, items_val)
    res.update({'run': name, 'trainable_params': int(n_params), 'frozen_params': int(n_frozen),
                'train_time_s': train_time, 'setup_time_s': setup_time, 'args': vars(args)})
    print('FINAL', json.dumps(res), flush=True)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


def main():
    args = parse_args()
    print(args, flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device, flush=True)

    from dataset.load_dataset import load_train_val_fold_file
    t = time.time()
    train_data, val_data, coef_norm, vallst = load_train_val_fold_file(args, preprocessed=True)
    print(f'data loaded in {time.time() - t:.1f}s', flush=True)
    if args.max_train:
        train_data = train_data[:args.max_train]
    if args.max_val:
        val_data, vallst = val_data[:args.max_val], vallst[:args.max_val]

    configs = []
    if args.suite:
        for ov in SUITE:
            a = copy.copy(args)
            for k, v in ov.items():
                setattr(a, k, v)
            configs.append(a)
    else:
        configs.append(args)

    model_cache, failed = {}, []
    for i, a in enumerate(configs):
        print(f'\n===== run {i + 1}/{len(configs)}: {run_name(a)}', flush=True)
        try:
            run(a, train_data, val_data, coef_norm, vallst, device, model_cache)
        except Exception:
            if not args.suite:
                raise
            traceback.print_exc()
            failed.append(run_name(a))
        finally:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if args.suite:
        print('\nSUITE DONE. failed runs:', failed if failed else 'none', flush=True)


if __name__ == '__main__':
    main()
