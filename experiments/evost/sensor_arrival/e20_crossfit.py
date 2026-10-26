"""E20 (g41 family xft): is the calibrator hurt by being fitted on the windows the identity table was just fitted on?

g33's protocol (e10_refresh, arm fit_emb) refits the five-year-old STAEFormer's embedding table on the last 2,048 train
windows and then fits every sensor's calibrator by exact ridge on the same 2,048 windows, with forecasts of the table that
was fitted on them: the calibrator's initial fit sees in-sample forecasts. Arm fit_emb_x keeps everything of fit_emb - the
same refitted table for every forecast of the stream, the same calibrator, ridge/forgetting grid, selection and stream -
and changes one thing: the forecasts its initial ridge fit sees are out-of-fold. The 2,048 fit windows are cut in two
halves; a table is fitted on each half (from the stale table, g33's rate, epochs and patience, the windows of a half that
could share a row with the other half left out: a gap of SPAN windows at the cut), and every window of the calibrator's
fit piece gets the forecast of the table fitted on the other half. The selection of ridge and forgetting factor, which
also starts from that ridge fit, sees the same out-of-fold forecasts.

Arms: same, stale5o, fit_emb (g33's, unchanged) and fit_emb_x. Rule X1 in ../PRESPEC.md.
"""
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

GAP = g.SPAN                                                             # 24: windows i and j share a row when |i - j| < 24
HALF = e3.FIT_WINDOWS // 2


def fit_on_windows(B, device, start, trainable, windows, tag):
    """The embedding-table fit of e18_handover.fit_windows (g40) for an explicit list of windows [(raw, index)] in time order:
    g33's rate, epochs and patience, the last quarter of the windows as holdout, a fixed number of windows per epoch (as many
    as g33's fit on 2,048 windows), loss on the trainable sensors only, shared weights frozen."""
    began = time.time()
    n_fit, stride = len(windows), e10.fit_stride(B.n)
    hold_n = n_fit // 4
    train = np.arange(0, n_fit - hold_n, stride)
    hold = np.arange(n_fit - hold_n, n_fit, stride)
    per_epoch = len(np.arange(0, e3.FIT_WINDOWS - e3.FIT_WINDOWS // 4, stride))
    model = B.model(B.old, device)
    for p in model.parameters():
        p.requires_grad_(False)
    mean, std = B.norm[B.old]
    weight = trainable.to(device)[None, :, None]

    def batch(idx):
        x = np.stack([np.array(g.get_windows(windows[i][0], int(windows[i][1]), 1)[0][0], dtype=np.float32) for i in idx])
        y = g.clean_y(np.stack([np.array(g.get_windows(windows[i][0], int(windows[i][1]), 1)[1][0]) for i in idx])).to(device)
        return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y) & weight

    def predict(x, table):
        model.backbone.adaptive_embedding = table if isinstance(table, nn.Parameter) else nn.Parameter(table, requires_grad=False)
        return model.fc(model.activation(model.backbone(x, None) + x))

    size = [e10.fit_batch(B.n)]

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

    frozen = (~trainable).to(device)
    table = nn.Parameter(start.clone())
    optim = torch.optim.Adam([table], lr=e10.FIT['lr'])
    gen = torch.Generator().manual_seed(7919 * B.seed + B.year + 17 * len(tag))
    history = [holdout(start)]
    best = (history[0], start.clone(), 0)
    for epoch in range(e10.FIT['epochs']):
        passes = []
        while sum(len(p) for p in passes) < per_epoch:
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
                table.grad[:, frozen] = 0.
                optim.step()
                return b
            s += guarded(step)
        history.append(holdout(table.detach()))
        if history[-1] < best[0]:
            best = (history[-1], table.detach().clone(), epoch + 1)
        elif epoch + 1 - best[2] >= e10.FIT['patience']:
            break
    model.backbone.adaptive_embedding = nn.Parameter(e5.StaeBundle.table(B, B.old, device).clone(), requires_grad=False)
    assert torch.equal(best[1][:, frozen], start[:, frozen])
    B.info[tag] = dict(network_sensors=int(B.n), trained_sensors=int(trainable.sum()), fit_windows=int(n_fit), train_windows=int(len(train)),
                       holdout_windows=int(len(hold)), windows_per_epoch=int(per_epoch), batch_windows=size[0], lr=e10.FIT['lr'],
                       holdout_mae=history, best_epoch=best[2], epochs_run=len(history) - 1,
                       mean_abs_change=float((best[1][:, trainable.to(device)] - start[:, trainable.to(device)]).abs().mean()), seconds=time.time() - began)
    return best[1]


class Memo:
    """e16_budget.Memo, copied: a bundle archive whose arrays are read once, so that a raw split is one object wherever it is
    asked for (the routing below and e5_stae's forecast cache rely on the identity of the raw split)."""

    def __init__(self, z):
        self._z, self.files, self._seen = z, z.files, {}

    def __getitem__(self, key):
        if key not in self._seen:
            self._seen[key] = self._z[key]
        return self._seen[key]


class CrossBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.z = Memo(self.z)

    def fit_piece_first(self):
        n = g.n_windows(self.z['raw_train'])
        assert n >= e3.FIT_WINDOWS, 'cross-fitting needs the full fit piece'
        return n - e3.FIT_WINDOWS                                        # e3.run's fit piece: train windows [first, first + 2048)

    def variant(self, ckpt, name, device):
        if name not in ('xfold:A', 'xfold:B'):
            return super().variant(ckpt, name, device)
        key = (ckpt, name)
        if key not in self._variant:
            assert ckpt == self.old
            first, raw = self.fit_piece_first(), self.z['raw_train']
            span = range(first, first + HALF - GAP) if name.endswith('A') else range(first + HALF + GAP, first + e3.FIT_WINDOWS)
            t = fit_on_windows(self, device, e5.StaeBundle.table(self, ckpt, device), torch.ones(self.n, dtype=torch.bool),
                               [(raw, i) for i in span], 'fit_' + name.replace(':', '_'))
            assert t.shape == (12, self.n, 80) and bool(torch.isfinite(t).all())
            self._variant[key] = t.contiguous()
        return self._variant[key]

    def base(self, key, raw, first, count, device):
        """Arm fit_emb_x: in the calibrator's fit piece the forecast of the table fitted on the other half, elsewhere fit_emb's."""
        if key[3] != 'xfit':
            return super().base(key, raw, first, count, device)
        full = key[:3] + ('fit',)
        if raw is not self.z['raw_train']:
            return super().base(full, raw, first, count, device)
        lo = self.fit_piece_first()
        assert lo <= first and first + count <= lo + e3.FIT_WINDOWS, 'fit_emb_x reads train windows of the fit piece only'
        cut = lo + HALF
        parts = []
        if first < cut:
            parts.append(super().base(key[:3] + ('xfold:B',), raw, first, min(first + count, cut) - first, device))
        if first + count > cut:
            parts.append(super().base(key[:3] + ('xfold:A',), raw, max(first, cut), first + count - max(first, cut), device))
        return torch.cat(parts)


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit'), arm('fit_emb_x', B.old, 'xfit')]
    B.info.update(sensors=B.n, half=HALF, gap=GAP, fit_windows=int(e3.FIT_WINDOWS))
    return arms


e3.Bundle = CrossBundle
e3.arms_of = arms_of
e3.make_chunks = e5.make_chunks
e3.check_reference = e10.check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(e10.RefreshBundle.last.info, indent=2) + '\n')
