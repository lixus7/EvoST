"""E10 (g33): refreshing the per-sensor identity of a five-year-old frozen STAEFormer, under the g29 protocol.

g30 showed that a five-year-old STAEFormer plus the per-sensor calibrator is about 19% worse than the same-year one,
g31 C2 that almost all of this is the per-sensor embedding table: the shared weights hardly age. Here the shared weights
of the five-year-old checkpoint stay frozen and only the embedding table (12 x sensors x 80) is replaced.

Built on e5_stae; the stream, the calibrator and the metrics are the unchanged e3_confirm. Arms (base forecast):

  same            same-year checkpoint (reference)
  same_mean       same-year checkpoint, every sensor gets the mean embedding            (g31 C2)
  same_prof       same-year checkpoint, every sensor gets the mean embedding of the K other sensors with the most
                  similar time-of-day profile                                           (is identity a function of behaviour?)
  same_geo        ... of the K other sensors with the largest adjacency weight
  same_rand       ... of K random other sensors
  stale5o         five-year-old checkpoint with its own embeddings (mean embedding for sensors it does not know) and its
                  own input statistics                                                  (g30; the arm to improve)
  stale5o_mean    five-year-old checkpoint, mean embedding for every sensor             (g31 C2)
  don_prof        every sensor gets the mean old embedding of the K sensors of the old year whose profile then is most
                  similar to the sensor's profile now; the sensor's own old embedding is excluded
  don_geo         ... of the K known sensors with the largest adjacency weight
  don_rand        ... of K random sensors of the old year
  new_prof        as stale5o, but sensors the old checkpoint does not know take the don_prof embedding
  new_geo         as stale5o, but sensors the old checkpoint does not know take the don_geo embedding
  fit_emb         the embedding table is fitted on the fit windows of the target year (shared weights frozen), starting
                  from the stale5o table
  nobb_zero, nobb_last   no backbone

Everything that defines a table is computed from the train tail of the target year (the windows the calibrator is fitted
on), from the old year's bundle and from the adjacency: no validation or test value is read.
"""
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # required by deterministic CUDA matrix products
import numpy as np
import torch
import torch.nn as nn
import e3_confirm as e3
import e5_stae as e5
import g28_common as g

K = 5
FIT = dict(lr=3e-3, epochs=int(os.environ.get('G33_FIT_EPOCHS', '16')), patience=3, holdout=512)    # the override is for smokes only
OLD_VARIANTS = ('mean', 'don_prof', 'don_geo', 'don_rand', 'new_prof', 'new_geo', 'fit')
SAME_VARIANTS = ('mean', 'loo_prof', 'loo_geo', 'loo_rand')


def fit_stride(n):
    return 2 if n <= 1600 else 4


def fit_batch(n):
    """Windows per optimiser step (halved on an out-of-memory error). g33 ran graphs up to 2,587 sensors; larger ones start at one window."""
    return 16 if n <= 800 else (4 if n <= 1600 else (2 if n <= 2600 else 1))


def steps_per_day(dataset):
    return 24 if dataset.upper().startswith('TFNSW') else 288


def profile(raw, first_row, period):
    """Mean value per time of day, [sensors, period]; `first_row` is the position of raw[0] in its year."""
    x = np.asarray(raw, np.float64)
    fin = np.isfinite(x)
    phase = (first_row + np.arange(len(x))) % period
    onehot = np.zeros((period, len(x)))
    onehot[phase, np.arange(len(x))] = 1.
    count = onehot @ fin
    total = onehot @ np.where(fin, x, 0.)
    level = np.where(fin, x, 0.).sum(0) / np.maximum(fin.sum(0), 1)
    prof = np.where(count > 0, total / np.maximum(count, 1), level[None])
    return prof.T, fin.mean(0) >= .2                                   # profile and "has enough data to be described"


def nearest(distance, k, banned=None):
    """Indices of the k smallest distances per row; banned [rows] holds one column per row to skip (or -1)."""
    d = np.array(distance, np.float64)
    if banned is not None:
        rows = np.flatnonzero(banned >= 0)
        d[rows, banned[rows]] = np.inf
    k = min(k, d.shape[1] - 1)
    part = np.argpartition(d, k - 1, axis=1)[:, :k]
    assert np.isfinite(np.take_along_axis(d, part, 1)).all(), 'not enough candidates'
    return part


def squared_distance(a, b):
    return (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T


class RefreshBundle(e5.StaeBundle):
    last = None

    def __init__(self, path):
        super().__init__(path)
        self.path = Path(path).resolve()
        self.old = self.year - 5
        assert self.old in self.ckpt_years[1:], 'five-year-old checkpoint missing'
        self._variant, self._donors, self.info = {}, None, {}
        RefreshBundle.last = self

    def adjacency(self, device):
        gcn = self.path.parent.parent / 'gcn' / self.path.name
        if not gcn.exists():
            return None
        zg, _ = g.load_bundle(gcn)
        assert [str(s) for s in zg['sensor_ids']] == [str(s) for s in self.z['sensor_ids']]
        return torch.as_tensor(np.asarray(zg['adj_norm'], np.float32), device=device)

    # ---- donor sets: from the train tails and the adjacency only ----
    def donors(self):
        if self._donors is not None:
            return self._donors
        m, z, seed = self.meta, self.z, self.seed
        period = steps_per_day(m['dataset'])
        tail = z['raw_train']
        now, _ = profile(tail, m['split_lengths']['train'] - len(tail), period)
        old_path = self.path.parent / ('%s_%d.npz' % (m['dataset'], self.old))
        zo, mo = g.load_bundle(old_path)
        assert mo['layout'] == 'g30-stae' and int(mo['year']) == self.old
        tail_old = zo['raw_train']
        then, described = profile(tail_old, mo['split_lengths']['train'] - len(tail_old), period)
        ids, ids_old = [str(s) for s in z['sensor_ids']], [str(s) for s in zo['sensor_ids']]
        position = {s: i for i, s in enumerate(ids_old)}
        index = np.array([position.get(s, -1) for s in ids])           # the sensor's own row in the old year, or -1
        known = index >= 0
        assert np.array_equal(known, np.asarray(z['known/y%d' % self.old], bool)) and known.any()
        table_old = np.asarray(zo['emb/%d/y%d' % (seed, self.old)], np.float32)          # [12, old sensors, 80]
        mapped = np.asarray(z['emb/%d/y%d' % (seed, self.old)], np.float32)
        assert np.array_equal(mapped[:, known], table_old[:, index[known]]), 'the two bundles disagree on the old embeddings'
        rng = np.random.default_rng(1000003 * seed + self.year)
        n, n_old = self.n, len(ids_old)

        d_old = squared_distance(now, then)
        d_old[:, ~described] = np.inf                                   # an old sensor without data is nobody's donor
        own_first = float((np.argmin(d_old[known], 1) == index[known]).mean())
        don_prof = nearest(d_old, K, index)
        don_rand = np.stack([rng.choice(np.delete(np.arange(n_old), index[i]) if index[i] >= 0 else n_old, K, replace=False) for i in range(n)])
        d_now = squared_distance(now, now)
        loo_prof = nearest(d_now, K, np.arange(n))
        loo_rand = np.stack([rng.choice(np.delete(np.arange(n), i), K, replace=False) for i in range(n)])
        out = dict(index=index, known=known, table_old=table_old, don_prof=don_prof, don_rand=don_rand, loo_prof=loo_prof, loo_rand=loo_rand,
                   don_geo=None, loo_geo=None)
        gcn = self.path.parent.parent / 'gcn' / self.path.name
        if gcn.exists():
            zg, _ = g.load_bundle(gcn)
            assert [str(s) for s in zg['sensor_ids']] == ids
            adj = np.asarray(zg['adj_norm'], np.float64)
            far = -adj
            out['loo_geo'] = nearest(far, K, np.arange(n))
            far_known = far.copy()
            far_known[:, ~known] = np.inf                               # only sensors the old checkpoint knows can lend an old embedding
            if int(known.sum()) > K:
                out['don_geo'] = index[nearest(far_known, K, np.arange(n))]
                assert (out['don_geo'] >= 0).all()
        assert (don_prof != index[:, None]).all() and (loo_prof != np.arange(n)[:, None]).all()
        self.info.update(steps_per_day=period, sensors=n, sensors_old_year=n_old, known=int(known.sum()), unknown=int((~known).sum()),
                         old_sensors_without_data=int((~described).sum()), neighbours=K,
                         share_of_known_sensors_whose_closest_old_profile_is_their_own=own_first,
                         geo_arms=bool(out['don_geo'] is not None), old_bundle=old_path.name)
        self._donors = out
        return out

    # ---- the embedding-only fit ----
    def fit(self, device, ckpt=None, start=None, tag='fit'):
        """Embedding table of checkpoint `ckpt` (default: five years old) fitted on the fit windows, from `start` (default: its stale table)."""
        ckpt = self.old if ckpt is None else ckpt
        began, raw = time.time(), self.z['raw_train']
        n_fit = min(e3.FIT_WINDOWS, g.n_windows(raw))
        first = g.n_windows(raw) - n_fit
        stride = fit_stride(self.n)
        train = np.arange(first, first + n_fit - FIT['holdout'], stride)
        hold = np.arange(first + n_fit - FIT['holdout'], first + n_fit, stride)
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

        size = [fit_batch(self.n)]

        def guarded(fn):
            """Run fn(batch size); on an out-of-memory error halve the batch and try again."""
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
        optim = torch.optim.Adam([table], lr=FIT['lr'])
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)     # the same order of windows for every fit of a run
        history = [holdout(start)]
        best = (history[0], start.clone(), 0)
        for epoch in range(FIT['epochs']):
            order = train[torch.randperm(len(train), generator=gen).numpy()]
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
            elif epoch + 1 - best[2] >= FIT['patience']:
                break
        model.backbone.adaptive_embedding = nn.Parameter(start.clone(), requires_grad=False)
        self.info[tag] = dict(checkpoint_year=int(ckpt), lr=FIT['lr'], max_epochs=FIT['epochs'], patience=FIT['patience'], stride=stride, train_windows=int(len(train)),
                                holdout_windows=int(len(hold)), batch_windows=size[0], holdout_mae=history, best_epoch=best[2],
                                mean_abs_change=float((best[1] - start).abs().mean()), seconds=time.time() - began)
        return best[1]

    def variant(self, ckpt, name, device):
        key = (ckpt, name)
        if key not in self._variant:
            if ckpt == self.year:
                full = e5.StaeBundle.table(self, self.year, device)
                if name == 'mean':
                    t = full.mean(1, keepdim=True).expand_as(full)
                else:
                    t = full[:, torch.as_tensor(self.donors()[name], device=device)].mean(2)
            else:
                stale = e5.StaeBundle.table(self, ckpt, device)
                if name == 'mean':
                    t = stale.mean(1, keepdim=True).expand_as(stale)
                elif name == 'fit':
                    t = self.fit(device, ckpt, tag='fit' if ckpt == self.old else 'fit%d' % (self.year - ckpt))
                elif name == 'fit_prof':                       # borrow by behaviour for unknown sensors, then fit
                    t = self.fit(device, ckpt, self.variant(ckpt, 'new_prof', device), tag='fit_prof')
                else:
                    assert ckpt == self.old, 'donor tables exist for the five-year-old checkpoint only'
                    D = self.donors()
                    old = torch.as_tensor(D['table_old'], device=device)
                    t = old[:, torch.as_tensor(D['don_' + name[4:]], device=device)].mean(2)
                    if name.startswith('new_'):
                        unknown = torch.as_tensor(~D['known'], device=device)
                        t = torch.where(unknown[None, :, None], t, stale)
                        assert torch.equal(t[:, ~unknown], stale[:, ~unknown])
            assert t.shape == (12, self.n, 80) and bool(torch.isfinite(t).all())
            self._variant[key] = t.contiguous()
        return self._variant[key]

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if variant in ('full', 'common'):
            return super()._forward(key, x_raw, device, sub)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model, table = self.model(ckpt, device), self.variant(ckpt, variant, device)
        out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))


def arms_of(B, seed):
    B.seed = seed
    D = B.donors()
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same', B.year, 'full'), arm('same_mean', B.year, 'mean'), arm('same_prof', B.year, 'loo_prof'), arm('same_rand', B.year, 'loo_rand'),
            arm('stale5o', B.old, 'full'), arm('stale5o_mean', B.old, 'mean'), arm('don_prof', B.old, 'don_prof'), arm('don_rand', B.old, 'don_rand'),
            arm('new_prof', B.old, 'new_prof'), arm('fit_emb', B.old, 'fit')]
    if D['loo_geo'] is not None:
        arms.append(arm('same_geo', B.year, 'loo_geo'))
    if D['don_geo'] is not None:
        arms += [arm('don_geo', B.old, 'don_geo'), arm('new_geo', B.old, 'new_geo')]
    arms.append(dict(name='nobb_zero', base='zero', ckpt=None, norm=B.year, age=None, variant=None))
    arms.append(dict(name='nobb_last', base='last', ckpt=None, norm=B.year, age=None, variant=None))
    return arms


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


e3.Bundle = RefreshBundle
e3.arms_of = arms_of
e3.check_reference = check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)          # so that a run can be repeated on the same machine
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(RefreshBundle.last.info, indent=2) + '\n')
