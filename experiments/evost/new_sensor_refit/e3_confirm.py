"""E3 (g29): stale frozen backbone + per-node Feature-RLS, confirmation protocol.

The calibration protocol is the one of g28 E1-B, unchanged: the same 29 features and
recursion (g28_common.NodeRLS), exact ridge on the last 2048 train windows of the target
year, ridge/forget chosen prequentially on the first 1024 validation windows, then one
continuous stream over validation + test with labels released after 12 windows and
absorbed every 64 windows. Only the base forecast differs between arms:

  same        same-year checkpoint (the yearly-retraining reference)
  stale<k>    checkpoint that is k years older, inputs normalised with target-year statistics
              (the development protocol)
  stale<k>o   the same checkpoint with its own training-year input statistics
  nobb_zero   no backbone: base forecast 0 (the calibrator is a per-node linear forecaster)
  nobb_last   no backbone: base forecast = last observed value

Reads g29 bundles and, for the regression check, the g28 development bundles.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import g28_common as g

RIDGES, FORGETS = (1., 100.), (1., .999, .995, .99)          # identical to g28 e1_topology
FIT_WINDOWS, SELECT_WINDOWS = 2048, 1024
DELAY, CADENCE, BLOCK = 12, 64, 256


class Bundle:
    def __init__(self, path):
        self.z, self.meta = g.load_bundle(path)
        m = self.meta
        self.year, self.n = int(m['year']), int(m['nodes'])
        self.g29 = m.get('layout') == 'g29'
        if self.g29:
            self.ckpt_years = [int(y) for y in m['ckpt_years']]
            self.norm = {int(k): (v['mean'], v['std']) for k, v in m['norm'].items()}
        else:                                                 # g28 development layout
            stale = m.get('stale_year')
            self.ckpt_years = [self.year] + ([int(stale)] if stale is not None else [])
            self.norm = {self.year: (m['mean'], m['std'])}
            if stale is not None:
                self.norm[int(stale)] = (m['mean_prev'], m['std_prev'])

    def key(self, seed, ckpt_year):
        return ('%d/y%d' % (seed, ckpt_year)) if self.g29 else ('%d/%s' % (seed, 'same' if ckpt_year == self.year else 'stale'))

    def available(self, seed, ckpt_year):
        return ('ckpt/%s/%s' % (self.key(seed, ckpt_year), g.GCN_KEYS[0])) in self.z.files

    def weights(self, seed, ckpt_year, device):
        prefix = 'ckpt/%s/' % self.key(seed, ckpt_year)
        return {k: torch.as_tensor(self.z[prefix + k], dtype=torch.float32, device=device) for k in g.GCN_KEYS}

    def adjacency(self, device):
        a = self.z['adj_norm'] if 'adj_norm' in self.z.files else g.normalize_adj(self.z['adj'])
        return torch.as_tensor(np.asarray(a, np.float32), device=device)

    def reference(self, seed, ckpt_year, norm_year):
        if not self.g29:
            return self.z['ref/%s/pred' % self.key(seed, ckpt_year)] if norm_year == self.year else None
        name = 'ref/%s/%s' % (self.key(seed, ckpt_year), 'year' if norm_year == self.year else 'own')
        return self.z[name] if name in self.z.files else None

    def new_mask(self, stale_year):
        if self.g29:
            name = 'new_mask/y%d' % stale_year
            return np.asarray(self.z[name], bool) if name in self.z.files else None
        return np.asarray(self.z['new_mask'], bool)


def arms_of(B, seed):
    arms = [dict(name='same', base='gcn', ckpt=B.year, norm=B.year, age=0)]
    for y in B.ckpt_years[1:]:
        if B.available(seed, y):
            arms.append(dict(name='stale%d' % (B.year - y), base='gcn', ckpt=y, norm=B.year, age=B.year - y))
            arms.append(dict(name='stale%do' % (B.year - y), base='gcn', ckpt=y, norm=y, age=B.year - y))
    arms.append(dict(name='nobb_zero', base='zero', ckpt=None, norm=B.year, age=None))
    arms.append(dict(name='nobb_last', base='last', ckpt=None, norm=B.year, age=None))
    return arms


def chunk_of(n_virtual):
    return 512 if n_virtual <= 12000 else (256 if n_virtual <= 24000 else 128)


def make_chunks(B, weights, adj, device, arms, pieces):
    """chunk_fn over a concatenation of raw pieces; the node axis tiles the arms (arm-major).

    pieces: list of (raw_array, first_window, count). Returns (T, chunk_fn).
    """
    meta = B.meta
    offsets = np.cumsum([0] + [c for _, _, c in pieces])
    T = int(offsets[-1])

    def gather(s):
        xs, ys = [], []
        t = s.start
        while t < s.stop:
            k = int(np.searchsorted(offsets, t, side='right') - 1)
            raw, first, _ = pieces[k]
            take = min(s.stop, int(offsets[k + 1])) - t
            x_raw, y_raw = g.get_windows(raw, first + t - int(offsets[k]), take)
            xs.append(np.array(x_raw, dtype=np.float32))
            ys.append(g.clean_y(y_raw))
            t += take
        return np.concatenate(xs), torch.cat(ys).to(device)

    def chunk_fn(s):
        x_raw, y = gather(s)
        memo, xs, bs = {}, [], []
        for arm in arms:
            key = (arm['base'], arm['ckpt'], arm['norm'])
            if key not in memo:
                x = g.normalize_x(x_raw, *B.norm[arm['norm']]).to(device)
                if arm['base'] == 'gcn':
                    b = g.gcn_predict(weights[arm['ckpt']], x, adj, meta['tcn_padding'], meta['tcn']['dilation'])
                elif arm['base'] == 'zero':
                    b = torch.zeros_like(y)
                else:
                    last = torch.nan_to_num(torch.from_numpy(x_raw[..., -1:]), nan=0., posinf=0., neginf=0.)
                    b = last.to(device).expand_as(y)
                memo[key] = (x.to(torch.float64), b.to(torch.float64))
            xs.append(memo[key][0])
            bs.append(memo[key][1])
        r = len(arms)
        return (torch.cat(xs, 1), torch.cat(bs, 1), y.repeat(1, r, 1).to(torch.float64),
                g.window_valid(y).repeat(1, r), None)
    return T, chunk_fn


def check_reference(B, seed, arms, weights, adj, device):
    report = {}
    raw, index = B.z['raw_test'], B.z['ref_index']
    for arm in arms:
        if arm['base'] != 'gcn':
            continue
        ref = B.reference(seed, arm['ckpt'], arm['norm'])
        if ref is None:
            report[arm['name']] = None
            continue
        x = torch.stack([g.normalize_x(g.get_windows(raw, int(i), 1)[0][0], *B.norm[arm['norm']])
                         for i in index]).to(device)
        ref = torch.as_tensor(ref, device=device)
        mine = g.gcn_predict(weights[arm['ckpt']], x, adj, B.meta['tcn_padding'], B.meta['tcn']['dilation'])
        worst, scale = float((mine - ref).abs().max()), float(ref.abs().max())
        assert worst <= 1e-3 * (1 + scale), ('standalone GCN differs from the original model', arm['name'], worst, scale)
        report[arm['name']] = dict(max_abs_difference=worst, reference_abs_max=scale, windows=int(len(index)))
    return report


def run(B, seed, device, max_windows, out):
    n, z = B.n, B.z
    arms = arms_of(B, seed)
    adj = B.adjacency(device)
    weights = {a['ckpt']: B.weights(seed, a['ckpt'], device) for a in arms if a['base'] == 'gcn'}
    reference = check_reference(B, seed, arms, weights, adj, device)
    train, val, test = z['raw_train'], z['raw_val'], z['raw_test']
    fit_n = min(FIT_WINDOWS, g.n_windows(train))
    fit_piece = (train, g.n_windows(train) - fit_n, fit_n)
    n_val = g.n_windows(val) if not max_windows else min(max_windows, g.n_windows(val))
    n_test = g.n_windows(test) if not max_windows else min(max_windows, g.n_windows(test))
    select_n = min(SELECT_WINDOWS, n_val)
    _, y_fit = g.get_windows(train, fit_piece[1], fit_n)
    y_fit = g.clean_y(y_fit)
    scale = float(y_fit[g.target_valid(y_fit)].std())
    del y_fit

    def fitted(arm_list, ridge, forget):
        """One RLS state per virtual node (arm-major), fitted on the train tail."""
        _, fn = make_chunks(B, weights, adj, device, arm_list, [fit_piece])
        model = g.NodeRLS(n * len(arm_list), g.Y_LEN, scale, ridge, forget, device)
        step = 256 if n * len(arm_list) <= 24000 else 64
        for s in range(0, fit_n, step):
            x, b, y, valid, _ = fn(slice(s, min(s + step, fit_n)))
            model.fit(x, b, y, valid)
        return model

    # ---- hyper-parameters: prequential on the first validation windows, per arm ----
    grid = [(r, f) for r in RIDGES for f in FORGETS]
    selection, chosen = {}, {}
    began = time.time()
    for arm in arms:
        arm_list = [arm] * len(grid)
        ridge = torch.tensor([r for r, _ in grid], dtype=torch.float64).repeat_interleave(n)
        forget = torch.tensor([f for _, f in grid], dtype=torch.float64).repeat_interleave(n)
        model = fitted(arm_list, ridge, forget)
        T, fn = make_chunks(B, weights, adj, device, arm_list, [(val, 0, select_n)])
        acc = g.ErrAcc(n * len(grid), device)
        g.rls_stream(model, T, fn, DELAY, CADENCE, lambda t, pred, y, count: acc.add(pred, y),
                     chunk=chunk_of(n * len(grid)))
        arrays = acc.arrays()
        scores = []
        for i, (r, f) in enumerate(grid):
            m = g.summarize(*(arrays[k][i * n:(i + 1) * n] for k in ('abs', 'sq', 'cnt')))
            scores.append(dict(ridge=r, forget=f, validation_H12_MAE=m['H12_MAE']))
        best = min(scores, key=lambda v: v['validation_H12_MAE'])
        selection[arm['name']] = dict(grid=scores, chosen=best, selection_windows=int(select_n),
                                      forget_on_grid_boundary=bool(best['forget'] == min(FORGETS)))
        chosen[arm['name']] = (best['ridge'], best['forget'])
        del model, acc

    # ---- final stream: fit on the train tail, then one continuous stream over val + test ----
    r = len(arms)
    select_seconds, began = time.time() - began, time.time()
    ridge = torch.tensor([chosen[a['name']][0] for a in arms], dtype=torch.float64).repeat_interleave(n)
    forget = torch.tensor([chosen[a['name']][1] for a in arms], dtype=torch.float64).repeat_interleave(n)
    model = fitted(arms, ridge, forget)
    T, fn = make_chunks(B, weights, adj, device, arms, [(val, 0, n_val), (test, 0, n_test)])
    acc = {p: (g.ErrAcc(n * r, device), g.ErrAcc(n * r, device)) for p in ('val', 'test')}
    snapshots = []

    def totals():
        a = acc['val'][0].abs + acc['test'][0].abs
        c = acc['val'][0].cnt + acc['test'][0].cnt
        return torch.stack([a.reshape(r, -1).sum(1), c.reshape(r, -1).sum(1)]).cpu().numpy()

    def on_window(t, pred, y, count):
        acc['val' if t < n_val else 'test'][0].add(pred, y)
        if (t + 1) % BLOCK == 0 or t + 1 == T:
            snapshots.append((t + 1, totals()))

    chunk = chunk_of(n * r)
    g.rls_stream(model, T, fn, DELAY, CADENCE, on_window, chunk=chunk)
    for start in range(0, T, chunk):                          # uncalibrated reference on the same windows
        x, b, y, _, _ = fn(slice(start, min(start + chunk, T)))
        cut = max(0, min(n_val - start, len(x)))
        if cut:
            acc['val'][1].add(b[:cut], y[:cut])
        if cut < len(x):
            acc['test'][1].add(b[cut:], y[cut:])

    ref_year = B.year - 5 if (B.year - 5) in B.ckpt_years[1:] else None      # old/new split: sensors absent 5 years earlier
    mask = B.new_mask(ref_year) if ref_year is not None else None
    cohorts = dict(all=np.ones(n, bool))
    if mask is not None and 0 < mask.sum() < n:
        cohorts.update(old=~mask, new=mask)
    rows, sums = [], {}
    for portion in ('val', 'test'):
        cal, raw_acc = (a.arrays() for a in acc[portion])
        for i, arm in enumerate(arms):
            sl = slice(i * n, (i + 1) * n)
            for kind, arrays in (('cal', cal), ('raw', raw_acc)):
                sums['%s/%s/%s/node' % (portion, arm['name'], kind)] = np.stack(
                    [arrays[k][sl].sum(1) for k in ('abs', 'sq', 'cnt')], 1)
                sums['%s/%s/%s/horizon' % (portion, arm['name'], kind)] = np.stack(
                    [arrays[k][sl].sum(0) for k in ('abs', 'sq', 'cnt')], 1)
            for cohort, m in cohorts.items():
                rows.append(dict(dataset=B.meta['dataset'], year=B.year, seed=seed, portion=portion, arm=arm['name'],
                                 base=arm['base'], checkpoint_year=arm['ckpt'], norm_year=arm['norm'], age=arm['age'],
                                 cadence=CADENCE, cohort=cohort, nodes=int(m.sum()),
                                 ridge=chosen[arm['name']][0], forget=chosen[arm['name']][1],
                                 windows=int(n_val if portion == 'val' else n_test),
                                 validation_portion_contains_selection_windows=bool(portion == 'val'),
                                 calibrated=g.summarize(cal['abs'][sl], cal['sq'][sl], cal['cnt'][sl], m),
                                 uncalibrated=g.summarize(raw_acc['abs'][sl], raw_acc['sq'][sl], raw_acc['cnt'][sl], m)))
    ends = np.asarray([e for e, _ in snapshots], np.int64)
    cumulative = np.stack([v for _, v in snapshots])           # [blocks, 2, arms]
    sums['block_end'] = ends
    sums['block_abs_cnt'] = np.diff(np.concatenate([np.zeros_like(cumulative[:1]), cumulative]), axis=0)
    sums['arm_names'] = np.asarray([a['name'] for a in arms])
    np.savez_compressed(out / 'e3_sums.npz', **sums)
    g.save_json(out / 'e3_selection.json', dict(selection=selection, scale=scale, fit_windows=int(fit_n),
                                                label_delay=DELAY, cadence=CADENCE, n_val=int(n_val), n_test=int(n_test),
                                                mask='all 12 targets finite and non-zero'))
    return rows, arms, reference, dict(cohort_reference_year=ref_year if 'new' in cohorts else None, chunk=chunk,
                                       select_seconds=select_seconds, stream_seconds=time.time() - began)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-windows', type=int, default=0, help='bounded smoke only; 0 = complete split')
    ap.add_argument('--role', default='unspecified', help='development | confirmation (recorded only)')
    opt = ap.parse_args()
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if opt.device == 'auto' else opt.device))
    start = time.time()
    B = Bundle(opt.bundle)
    assert opt.seed in B.meta['seeds']
    with torch.no_grad():
        rows, arms, reference, extra = run(B, opt.seed, device, opt.max_windows, out)
    with (out / 'e3_metrics.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
    g.save_json(out / 'e3_setup.json', dict(dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n,
        layout='g29' if B.g29 else 'g28', checkpoint_years=B.ckpt_years, arms=arms, reference_check=reference,
        norm={str(k): v for k, v in B.norm.items()}, device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, torch=torch.__version__,
        max_windows=opt.max_windows, role=opt.role, peak_gpu_gib=peak, **extra))
    g.save_json(out / 'completed.json', dict(status='completed', rows=len(rows), seconds=time.time() - start,
                                             bounded_smoke=bool(opt.max_windows), role=opt.role))
    print(json.dumps(dict(status='e3_completed', dataset=B.meta['dataset'], year=B.year, seed=opt.seed,
                          nodes=B.n, arms=len(arms), seconds=round(time.time() - start, 1), peak_gpu_gib=peak)), flush=True)


if __name__ == '__main__':
    main()
