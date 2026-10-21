"""E16 (g38 A): how many labelled windows of the target year does the identity refresh need? Label-budget curve, g29 protocol.

g33 used the last 2,048 train windows of the target year three times: to refit the embedding table of the frozen
five-year-old STAEFormer, to fit the per-sensor calibrator, and (through the first 1,024 validation windows) to choose
the calibrator's ridge and forgetting factor. Here all three are cut to a budget b in BUDGETS:

  embedding refit     the last b train windows (the last b/4 of them as holdout), g33's fit otherwise
  calibrator fit      exact ridge on the last b train windows
  ridge / forget      chosen on the first b/2 validation windows

and the unchanged protocol (e3_confirm.run: one stream over validation + test, labels released after 12 windows, absorbed
every 64) is run once per budget, with three arms: same (same-year checkpoint), stale5o (five-year-old checkpoint, its
own embeddings), fit_emb (its table refitted on the budget). At b = 2048 this is g33.

One thing is kept equal across budgets so that only the amount of labelled data changes: the number of optimiser steps.
An epoch of the fit always visits as many windows as the full budget's epoch does (768, or 384 above 1,600 sensors); with
a smaller budget the same few windows are visited repeatedly (complete random passes over the pool, the last one cut).
Epochs, patience, batch sizes, learning rate (3e-3) and the holdout rule are g33's.

Outputs: <out>/b<budget>/{e3_metrics.jsonl, e3_setup.json, e3_selection.json, e3_sums.npz, completed.json} per budget and
<out>/{e10_info.json, completed.json}.
"""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn as nn
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import g28_common as g

FULL = 2048
BUDGETS = (64, 256, 1024, 2048)


def set_budget(b):
    """Everything that reads target-year labels before the stream is cut to b windows."""
    assert b in BUDGETS
    e3.FIT_WINDOWS, e3.SELECT_WINDOWS, e10.FIT['holdout'] = b, b // 2, b // 4


class Memo:
    """A bundle archive whose arrays are read once, so that a raw split is one object wherever it is asked for. The forecast cache
    of e5_stae is keyed by the identity of the raw split; with several runs of the protocol in one process an array read anew
    for every run could take the identity of one that was freed."""

    def __init__(self, z):
        self._z, self.files, self._seen = z, z.files, {}

    def __getitem__(self, key):
        if key not in self._seen:
            self._seen[key] = self._z[key]
        return self._seen[key]


class BudgetBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.z = Memo(self.z)

    def fit(self, device, ckpt=None, start=None, tag='fit'):
        """e10_refresh.fit on the current budget, with the full budget's number of windows per epoch (see the module text)."""
        ckpt = self.old if ckpt is None else ckpt
        began, raw = time.time(), self.z['raw_train']
        n_fit = min(e3.FIT_WINDOWS, g.n_windows(raw))
        first = g.n_windows(raw) - n_fit
        stride = e10.fit_stride(self.n)
        train = np.arange(first, first + n_fit - e10.FIT['holdout'], stride)
        hold = np.arange(first + n_fit - e10.FIT['holdout'], first + n_fit, stride)
        per_epoch = len(np.arange(0, FULL - FULL // 4, stride))                 # windows per epoch of the full budget
        model = self.model(ckpt, device)
        for p in model.parameters():
            p.requires_grad_(False)
        start = e5.StaeBundle.table(self, ckpt, device) if start is None else start
        mean, std = self.norm[ckpt]

        def batch(idx):
            x = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in idx])
            y = g.clean_y(np.stack([np.array(g.get_windows(raw, int(i), 1)[1][0]) for i in idx])).to(device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y)

        def predict(x, table):
            model.backbone.adaptive_embedding = table if isinstance(table, nn.Parameter) else nn.Parameter(table, requires_grad=False)
            return model.fc(model.activation(model.backbone(x, None) + x))

        size = [e10.fit_batch(self.n)]

        def guarded(fn):
            while True:
                try:
                    return fn(size[0])
                except torch.cuda.OutOfMemoryError:
                    assert size[0] > 1, 'out of GPU memory with one window per step'
                    size[0] = max(1, size[0] // 2)
                    torch.cuda.empty_cache()

        def holdout(table):
            def run(b):
                total = count = 0.
                with torch.no_grad():
                    for s in range(0, len(hold), b):
                        x, y, valid = batch(hold[s:s + b])
                        total += float(((predict(x, table) - y).abs() * valid).sum())
                        count += float(valid.sum())
                return total / max(count, 1.)
            return guarded(run)

        table = nn.Parameter(start.clone())
        optim = torch.optim.Adam([table], lr=e10.FIT['lr'])
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)
        history = [holdout(start)]
        best = (history[0], start.clone(), 0)
        for epoch in range(e10.FIT['epochs']):
            passes = []
            while sum(len(p) for p in passes) < per_epoch:                      # one pass at the full budget: g33's order exactly
                passes.append(train[torch.randperm(len(train), generator=gen).numpy()])
            order = np.concatenate(passes)[:per_epoch]
            s = 0
            while s < len(order):
                def step(b):
                    x, y, valid = batch(order[s:s + b])
                    with torch.enable_grad():
                        loss = ((predict(x, table) - y).abs() * valid).sum() / valid.sum().clamp_min(1)
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                    optim.step()
                    return b
                s += guarded(step)
            history.append(holdout(table.detach()))
            if history[-1] < best[0]:
                best = (history[-1], table.detach().clone(), epoch + 1)
            elif epoch + 1 - best[2] >= e10.FIT['patience']:
                break
        model.backbone.adaptive_embedding = nn.Parameter(start.clone(), requires_grad=False)
        self.info[tag] = dict(checkpoint_year=int(ckpt), budget=int(n_fit), lr=e10.FIT['lr'], max_epochs=e10.FIT['epochs'], patience=e10.FIT['patience'], stride=stride,
                              train_windows=int(len(train)), holdout_windows=int(len(hold)), windows_per_epoch=int(per_epoch), batch_windows=size[0],
                              holdout_mae=history, best_epoch=best[2], mean_abs_change=float((best[1] - start).abs().mean()), seconds=time.time() - began)
        return best[1]

    def variant(self, ckpt, name, device):
        if name.startswith('fit@'):                                              # the refit on one budget; a table of its own per budget
            key = (ckpt, name)
            if key not in self._variant:
                assert e3.FIT_WINDOWS == int(name[4:]) and ckpt == self.old
                self._variant[key] = self.fit(device, ckpt, tag=name).contiguous()
            return self._variant[key]
        return super().variant(ckpt, name, device)


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    return [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit@%d' % e3.FIT_WINDOWS)]


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


e3.Bundle = BudgetBundle
e3.arms_of = arms_of
e3.check_reference = check_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-windows', type=int, default=0, help='bounded smoke only; 0 = complete split')
    ap.add_argument('--role', default='unspecified')
    ap.add_argument('--budgets', type=int, nargs='+', default=list(BUDGETS), help='bounded smoke only')
    opt = ap.parse_args()
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else ('cpu' if opt.device == 'auto' else opt.device))
    start = time.time()
    B = BudgetBundle(opt.bundle)
    assert opt.seed in B.meta['seeds']
    bounded = bool(opt.max_windows or tuple(opt.budgets) != BUDGETS or e10.FIT['epochs'] != 16)
    for b in opt.budgets:
        set_budget(b)
        sub, began = out / ('b%d' % b), time.time()
        sub.mkdir()
        with torch.no_grad():
            rows, arms, reference, extra = e3.run(B, opt.seed, device, opt.max_windows, sub)
        with (sub / 'e3_metrics.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(dict(row, budget=b), allow_nan=False) + '\n')
        g.save_json(sub / 'e3_setup.json', dict(dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n, budget=b, fit_windows=e3.FIT_WINDOWS,
            select_windows=e3.SELECT_WINDOWS, fit_holdout=e10.FIT['holdout'], arms=arms, reference_check=reference, device=str(device),
            gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, torch=torch.__version__, max_windows=opt.max_windows, role=opt.role, **extra))
        g.save_json(sub / 'completed.json', dict(status='completed', rows=len(rows), seconds=time.time() - began, bounded_smoke=bounded, role=opt.role, budget=b))
        print(json.dumps(dict(status='budget_completed', budget=b, seconds=round(time.time() - began, 1), fit=dict(best_epoch=B.info['fit@%d' % b]['best_epoch'],
              holdout=[round(v, 3) for v in B.info['fit@%d' % b]['holdout_mae'][:6]]))), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
    B.info.update(sensors=B.n, budgets=list(opt.budgets), peak_gpu_gib=peak)
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
    g.save_json(out / 'completed.json', dict(status='completed', budgets=list(opt.budgets), seconds=time.time() - start, bounded_smoke=bounded, role=opt.role))
    print(json.dumps(dict(status='e16_completed', dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n, budgets=list(opt.budgets),
                          seconds=round(time.time() - start, 1), peak_gpu_gib=peak)), flush=True)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    main()
