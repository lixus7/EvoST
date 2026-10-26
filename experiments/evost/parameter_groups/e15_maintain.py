"""E15 (g37): which part of a five-year-old frozen STAEFormer should be updated? Fair maintenance baselines, g29 protocol.

g33 refitted only the per-sensor embedding table of the old checkpoint (fit_emb). Here the same data, the same holdout,
the same epochs / patience / batches / order of windows are given to other parameter groups of the same old model, and
the same refit is given to the same-year model:

  same          same-year checkpoint (reference)
  stale5o       five-year-old checkpoint, its own embeddings and input statistics (the arm to improve)
  fit_emb       g33's arm: only the embedding table, learning rate 3e-3
  fit_emb_sel   only the embedding table, learning rate chosen from EMB_RATES on the fit's holdout
  fit_all       every parameter: table at the rate fit_emb_sel chose in this run, shared weights at a rate chosen from
                ALL_RATES on the holdout (so that it contains fit_emb_sel as the limit of a vanishing shared rate)
  fit_shared    shared weights only (table frozen at the stale table), rate from SHARED_RATES
  fit_head      output projection of the backbone and the final linear layer only, rate from HEAD_RATES
  fit_same      same-year checkpoint, only its embedding table refitted (rate from SAME_RATES): what a second pass over the
                most recent windows is worth for a model that was trained on them already

Every fit starts from the checkpoint as it is (mean embedding for sensors the old checkpoint does not know), keeps the
model in evaluation mode (no dropout) as g33 does, and keeps the state with the lowest error on the holdout inside the
fit windows. For every arm and candidate rate the wall time, the peak GPU memory and the number of trained values are
recorded. No validation or test value is read by a fit.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn as nn
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import g28_common as g

EMB_RATE = 3e-3                                                          # the rate of g33
EMB_RATES = (1e-3, 3e-3, 1e-2)                                            # candidates placed by the development probe (PRESPEC.md, "Candidate rates")
SAME_RATES = (3e-4, 1e-3, 3e-3)
ALL_RATES = (3e-5, 1e-4, 3e-4)                                            # rate of the shared weights when everything is trained
SHARED_RATES = (1e-4, 3e-4, 1e-3)                                         # shared weights alone
HEAD_RATES = (3e-3, 1e-2, 3e-2)
# arm -> (checkpoint: 'old' | 'same', trainable groups, candidate rates of the group that is searched)
ARMS = dict(fit_emb_sel=('old', ('table',), EMB_RATES), fit_all=('old', ('table', 'shared'), ALL_RATES),
            fit_shared=('old', ('shared',), SHARED_RATES), fit_head=('old', ('head',), HEAD_RATES), fit_same=('same', ('table',), SAME_RATES))
if os.environ.get('G37_ARMS'):                                           # bounded smoke only
    ARMS = {k: ARMS[k] for k in os.environ['G37_ARMS'].split(',')}


def head_parameters(model):
    return list(model.backbone.output_proj.parameters()) + list(model.fc.parameters())


class MaintainBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self._tuned, self.peak_seen = {}, 0.

    def tune(self, device, ckpt, groups, rate, tag, table_rate=EMB_RATE):
        """One fit of the given parameter groups of checkpoint `ckpt`; returns (model, table, record). Mirrors e10_refresh.fit."""
        began, raw = time.time(), self.z['raw_train']
        n_fit = min(e3.FIT_WINDOWS, g.n_windows(raw))
        first = g.n_windows(raw) - n_fit
        stride = e10.fit_stride(self.n)
        train = np.arange(first, first + n_fit - e10.FIT['holdout'], stride)
        hold = np.arange(first + n_fit - e10.FIT['holdout'], first + n_fit, stride)
        model = copy.deepcopy(self.model(ckpt, device)) if ('shared' in groups or 'head' in groups) else self.model(ckpt, device)
        for p in model.parameters():
            p.requires_grad_(False)
        start = e5.StaeBundle.table(self, ckpt, device)
        mean, std = self.norm[ckpt]
        table = nn.Parameter(start.clone(), requires_grad='table' in groups)
        params = []
        if 'table' in groups:
            params.append(dict(params=[table], lr=rate if groups == ('table',) else table_rate))
        if 'shared' in groups:                                           # everything except the embedding table
            params.append(dict(params=[p for name, p in model.named_parameters() if name != 'backbone.adaptive_embedding'], lr=rate))
        if 'head' in groups:
            params.append(dict(params=head_parameters(model), lr=rate))
        for group in params:
            for p in group['params']:
                p.requires_grad_(True)
        trained = int(sum(p.numel() for group in params for p in group['params']))

        def batch(idx):
            x = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in idx])
            y = g.clean_y(np.stack([np.array(g.get_windows(raw, int(i), 1)[1][0]) for i in idx])).to(device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y)

        def predict(x):
            model.backbone.adaptive_embedding = table
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

        def holdout():
            def run(b):
                total = count = 0.
                with torch.no_grad():
                    for s in range(0, len(hold), b):
                        x, y, valid = batch(hold[s:s + b])
                        total += float(((predict(x) - y).abs() * valid).sum())
                        count += float(valid.sum())
                return total / max(count, 1.)
            return guarded(run)

        def snapshot():
            return (table.detach().clone(), copy.deepcopy(model.state_dict()) if ('shared' in groups or 'head' in groups) else None)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            self.peak_seen = max(self.peak_seen, torch.cuda.max_memory_allocated() / 2 ** 30)   # the run's peak survives the reset
            torch.cuda.reset_peak_memory_stats()
        optim = torch.optim.Adam(params)
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)     # the order of windows of g33, the same for every fit of a run
        history = [holdout()]
        best = (history[0], snapshot(), 0)
        epoch_seconds = []
        for epoch in range(e10.FIT['epochs']):
            t0 = time.time()
            order = train[torch.randperm(len(train), generator=gen).numpy()]
            s = 0
            while s < len(order):
                def step(b):
                    x, y, valid = batch(order[s:s + b])
                    with torch.enable_grad():
                        loss = ((predict(x) - y).abs() * valid).sum() / valid.sum().clamp_min(1)
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                    optim.step()
                    return b
                s += guarded(step)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            epoch_seconds.append(time.time() - t0)
            history.append(holdout())
            if history[-1] < best[0]:
                best = (history[-1], snapshot(), epoch + 1)
            elif epoch + 1 - best[2] >= e10.FIT['patience']:
                break
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
        self.peak_seen = max(self.peak_seen, peak or 0.)
        best_table, best_state = best[1]
        if best_state is not None:
            model.load_state_dict(best_state)
        for p in model.parameters():
            p.requires_grad_(False)
        model.backbone.adaptive_embedding = nn.Parameter(e5.StaeBundle.table(self, ckpt, device).clone(), requires_grad=False)
        record = dict(checkpoint_year=int(ckpt), groups=list(groups), lr=rate, table_lr=(table_rate if groups == ('table', 'shared') else (rate if 'table' in groups else None)),
                      max_epochs=e10.FIT['epochs'], patience=e10.FIT['patience'], stride=stride, train_windows=int(len(train)), holdout_windows=int(len(hold)),
                      batch_windows=size[0], holdout_mae=history, best_epoch=best[2], epochs_run=len(history) - 1, trained_values=trained,
                      mean_abs_change_table=float((best_table - start).abs().mean()), seconds=time.time() - began,
                      seconds_per_epoch=float(np.mean(epoch_seconds)) if epoch_seconds else None, peak_gpu_gib=peak)
        return model, best_table.contiguous(), record

    def tuned(self, name, device):
        """(model, table) of an arm: one fit per candidate rate, the one with the lowest holdout error kept (the smaller rate on a tie)."""
        if name not in self._tuned:
            which, groups, rates = ARMS[name]
            ckpt = self.old if which == 'old' else self.year
            best, runs = None, {}
            table_rate = EMB_RATE
            if name == 'fit_all' and 'fit_emb_sel' in ARMS:                      # the table moves at the rate the embedding-only fit chose
                self.tuned('fit_emb_sel', device)
                table_rate = self.info['fit_emb_sel']['selected_lr']
            for rate in rates:
                model, table, record = self.tune(device, ckpt, groups, rate, name, table_rate)
                runs['%g' % rate] = record
                if best is None or min(record['holdout_mae']) < min(best[2]['holdout_mae']):
                    best = (model, table, record)
                if name == 'fit_emb_sel' and rate == EMB_RATE:                   # g33's arm is this candidate
                    self._tuned['fit_emb'] = (model, table)
                    self.info['fit'] = dict(record)
            self._tuned[name] = best[:2]
            self.info[name] = dict(best[2], selected_lr=best[2]['lr'], candidates=runs, seconds=sum(r['seconds'] for r in runs.values()),
                                   peak_gpu_gib=max((r['peak_gpu_gib'] or 0.) for r in runs.values()) if device.type == 'cuda' else None)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        return self._tuned[name]

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if variant not in ARMS and variant != 'fit_emb':
            return super()._forward(key, x_raw, device, sub)
        if variant == 'fit_emb' and 'fit_emb' not in self._tuned:
            self.tuned('fit_emb_sel', device)
        model, table = self._tuned[variant] if variant == 'fit_emb' else self.tuned(variant, device)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full')]
    if 'fit_emb_sel' in ARMS:
        arms.append(arm('fit_emb', B.old, 'fit_emb'))
    for name, (which, _, _) in ARMS.items():
        arms.append(arm(name, B.old if which == 'old' else B.year, name))
    return arms


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


e3.Bundle = MaintainBundle
e3.arms_of = arms_of
e3.check_reference = check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    peak = max(B.peak_seen, torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None
    B.info.update(sensors=B.n, peak_gpu_gib_run=peak, shared_values=int(sum(p.numel() for name, p in B.model(B.old, torch.device('cpu')).named_parameters() if name != 'backbone.adaptive_embedding')),
                  table_values=int(np.prod(B.z['emb/%d/y%d' % (B.seed, B.old)].shape)), rates=dict(emb=EMB_RATES, same=SAME_RATES, all=ALL_RATES, shared=SHARED_RATES, head=HEAD_RATES, emb_fixed=EMB_RATE))
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
