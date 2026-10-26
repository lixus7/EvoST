"""E9 (g32): Relational Residual Adapter on top of a frozen backbone and the per-sensor Feature-RLS.

The local path is the g29 protocol, unchanged (e3_confirm / g28_common): exact ridge on the last 2048 train windows,
ridge/forget chosen on the first 1024 validation windows, one stream over validation + test with labels released after
12 windows and absorbed every 64 windows. Its forecast y_local is what the `base` arm reports.

Every adapter arm adds a correction that is computed from other sensors' current state,

    y_i = y_local_i + s_i * gate(c_i) * W_out m_i,      m_i = sum_j a_ij [phi(u_j) - phi(u_i)],
    a_ij = softmax_j (W_Q c_i)'(W_K c_j) / sqrt(d)      over K candidate neighbours j,

  rra    candidates = the K sensors with the most similar recent behaviour (rolling time-of-day profile and spread),
         attention learned
  fixed  the same candidates with fixed similarity weights (no learned attention)
  rand   K random sensors, attention learned
  geo    the K largest weights of the dataset adjacency, attention learned
  self   no neighbours: an adapter of the same size that reads only the sensor's own state

u_i (8) summarises the sensor's current input window in its own rolling units, plus the drift of its local forecast;
c_i (13) is u_i with statistics of the local forecast errors released so far, their age and an availability flag.
No arm has a per-sensor parameter. The output layer has no bias and starts at zero, so an untrained arm equals `base`.

Training sees only what was issued and what has been released: for windows whose labels have matured, the u, c,
candidate sets, scales and y_local cached at forecast time, with target y - y_local. Warm-up is a causal replay on the
second half of the fit windows (RLS fitted on the first half); online, one step per release. The RLS never sees the
correction.

g34: with --backbone refresh the local path is a five-year-old STAEFormer with a refreshed embedding table (e12_online);
for an online arm y_local is built on the base forecast as it was issued. Nothing else differs from g32.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # required by deterministic CUDA matrix products
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import g28_common as g
import e3_confirm as e3

DU, DM, DA, DE = 8, 16, 8, 5            # summary, message, attention, released-error context (ema, |ema|, last, age, available)
DC = DU + DE
DELAY, CADENCE = e3.DELAY, e3.CADENCE
ARMS = dict(rra=('attn', 'beh'), fixed=('fixed', 'beh'), rand=('attn', 'rand'), geo=('attn', 'geo'), self=('self', None))
CLIP_U, CLIP_E, MAX_AGE = 10., 5., 4.


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------

class Adapter(nn.Module):
    def __init__(self, kind, hidden=None):
        super().__init__()
        self.kind = kind
        if kind == 'attn':
            self.q, self.k = nn.Linear(DC, DA, bias=False), nn.Linear(DC, DA, bias=False)
        if kind == 'self':
            self.own = nn.Linear(DC, hidden)
        else:
            self.phi = nn.Linear(DU, DM)
        self.out = nn.Linear(hidden if kind == 'self' else DM, g.Y_LEN, bias=False)
        self.gate = nn.Linear(DC, 1)
        nn.init.zeros_(self.out.weight)

    def forward(self, u, c, idx=None, nbr=None, ok=None, weight=None, stats=None):
        """u [W, N, DU], c [W, N, DC]; targets idx [M] (every sensor when None); nbr [M, K] candidates of the targets;
        ok [W, N] sensors that may be read as neighbours; weight [M, K] (kind 'fixed').
        Returns the correction in the targets' rolling units, [W, M, 12]."""
        ci = c if idx is None else c[:, idx]
        gate = torch.sigmoid(self.gate(ci))
        if self.kind == 'self':
            m = F.relu(self.own(ci))
        else:
            p = F.relu(self.phi(u))
            pi = p if idx is None else p[:, idx]
            usable = ok[:, nbr]                                        # [W, M, K]
            if self.kind == 'attn':
                score = (self.q(ci).unsqueeze(2) * self.k(c)[:, nbr]).sum(-1) / math.sqrt(DA)
            else:
                score = weight.clamp_min(1e-30).log().unsqueeze(0).expand(usable.shape)
            a = torch.softmax(score.masked_fill(~usable, float('-inf')), -1)
            a = torch.nan_to_num(a, nan=0.)                            # no usable neighbour: no message
            m = (a.unsqueeze(-1) * (p[:, nbr] - pi.unsqueeze(2))).sum(2)
            if stats is not None:
                k = max(a.shape[-1], 2)
                stats['entropy'] = float((-(a * a.clamp_min(1e-12).log()).sum(-1)).mean() / math.log(k))
                stats['isolated'] = float((~usable.any(-1)).float().mean())
        if stats is not None:
            stats['gate'] = float(gate.mean())
        return gate * self.out(m)


def parameters_of(module):
    return int(sum(p.numel() for p in module.parameters()))


def matched_hidden(target):
    """Hidden width of the `self` adapter whose parameter count is closest to `target`."""
    return max(1, round((target - DC - 1) / (DC + 1 + g.Y_LEN)))


# --------------------------------------------------------------------------
# Causal inputs of the adapter
# --------------------------------------------------------------------------

class Timeline:
    """Raw rows in time order (train tail | validation | test) with a time-of-day phase that is consistent across splits."""

    def __init__(self, train, val, test, period):
        self.rows = np.concatenate([np.asarray(a, np.float32) for a in (train, val, test)])
        self.rows[~np.isfinite(self.rows)] = np.nan
        self.start = {id(train): 0, id(val): len(train), id(test): len(train) + len(val)}
        self.phase = ((np.arange(len(self.rows)) - len(train)) % period).astype(np.int64)
        self.period = int(period)

    def visible(self, raw, window):
        """Number of timeline rows known when the forecast of local window `window` of split `raw` is issued."""
        return self.start[id(raw)] + int(window) + g.X_LEN


class Graph:
    """Candidate sets and per-sensor rolling statistics valid from one refresh to the next (never modified afterwards)."""

    def __init__(self, timeline, rows_visible, history, k, static, device, bins=24):
        began = time.time()
        lo = max(0, rows_visible - history)
        x = torch.as_tensor(timeline.rows[lo:rows_visible], device=device)
        phase = torch.as_tensor(timeline.phase[lo:rows_visible], device=device)
        n = x.shape[1]
        fin = torch.isfinite(x)
        xz = torch.where(fin, x, torch.zeros_like(x))
        cnt = fin.sum(0)
        total = fin.sum().clamp_min(1)
        gmean = xz.sum() / total
        gstd = (torch.where(fin, (x - gmean) ** 2, torch.zeros_like(x)).sum() / total).sqrt().clamp_min(1e-6)
        mean = xz.sum(0) / cnt.clamp_min(1)
        std = (torch.where(fin, (x - mean) ** 2, torch.zeros_like(x)).sum(0) / cnt.clamp_min(1)).sqrt()
        known = cnt >= max(g.X_LEN, int(.2 * len(x)))                  # enough rows to describe the sensor
        self.mean = torch.where(known, mean, gmean.expand(n))
        self.std = torch.where(known, std, gstd.expand(n)).clamp_min(.1 * gstd)
        onehot = F.one_hot((phase * bins) // timeline.period, bins).to(x.dtype)       # [R, bins]
        den = onehot.T @ fin.to(x.dtype)
        prof = torch.where(den > 0, (onehot.T @ xz) / den.clamp_min(1), self.mean.expand(bins, n))
        desc = torch.cat([prof.T, self.std[:, None]], 1) / gstd                         # [N, bins + 1]
        dist = torch.cdist(desc, desc)
        dist.fill_diagonal_(float('inf'))
        if int(known.sum()) > k:
            dist[:, ~known] = float('inf')                             # a sensor without history is nobody's candidate
        self.k = min(k, n - 1)
        d, self.beh = torch.topk(dist, self.k, dim=1, largest=False)
        self.beh_w = torch.softmax(-d / d.mean(1, keepdim=True).clamp_min(1e-6), 1)
        self.rand, self.geo = static['rand'][:, :self.k], (None if static['geo'] is None else static['geo'][:, :self.k])
        self.rows = (lo, rows_visible)
        self.seconds = time.time() - began

    def candidates(self, key):
        return dict(beh=self.beh, rand=self.rand, geo=self.geo)[key]


def static_candidates(n, k, adjacency, seed, device):
    """Random candidates (fixed for the whole run) and the K largest adjacency weights."""
    k = min(k, n - 1)
    gen = torch.Generator().manual_seed(1000003 * seed + 17)
    rand = torch.stack([torch.randperm(n - 1, generator=gen)[:k] for _ in range(n)])
    rand = (rand + (rand >= torch.arange(n)[:, None]).long()).to(device)              # skip the sensor itself
    geo = None
    if adjacency is not None:
        a = adjacency.clone()
        a.fill_diagonal_(float('-inf'))
        geo = torch.topk(a, k, dim=1).indices
    return dict(rand=rand, geo=geo)


def summarise(x_raw, y_local, mean, std):
    """u [W, N, 8] and the sensors that may be read as neighbours [W, N]. x_raw [W, N, 12] with NaN kept."""
    z = (x_raw - mean[None, :, None]) / std[None, :, None]
    fin = torch.isfinite(z)
    z = torch.where(fin, z, torch.zeros_like(z))
    t = torch.arange(g.X_LEN, dtype=z.dtype, device=z.device)
    t = t - t.mean()
    m, last = z.mean(-1), z[..., -1]
    drift = (y_local.mean(-1) - mean) / std - last                    # where the local forecast goes from the last observation
    u = torch.stack([last, m, z.std(-1, unbiased=False), ((z - m[..., None]) * t).sum(-1) / (t * t).sum(),
                     last - z[..., -4], last - z[..., 0], z.amax(-1) - z.amin(-1), drift], -1)
    ok = (fin & (x_raw != 0)).sum(-1) >= g.X_LEN // 2
    return torch.nan_to_num(u, nan=0., posinf=CLIP_U, neginf=-CLIP_U).clamp(-CLIP_U, CLIP_U), ok


class ErrorContext:
    """What is known about each sensor's local forecast error from labels released so far (in the sensor's rolling units)."""

    def __init__(self, n, device, alpha):
        self.ema, self.mag, self.last = (torch.zeros(n, device=device) for _ in range(3))
        self.seen = torch.full((n,), -10 ** 9, dtype=torch.long, device=device)
        self.alpha = alpha

    def release(self, residual, valid, newest):
        """residual, valid [W, N, 12] of released windows; newest = stream index of the latest of them."""
        cnt = valid.sum((0, 2))
        has = cnt > 0
        w = valid.to(residual.dtype)
        mean = (residual * w).sum((0, 2)) / cnt.clamp_min(1)
        mag = (residual.abs() * w).sum((0, 2)) / cnt.clamp_min(1)
        first = has & (self.seen < 0)
        blend = lambda old, new: torch.where(first, new, torch.where(has, (1 - self.alpha) * old + self.alpha * new, old))
        self.ema, self.mag = blend(self.ema, mean), blend(self.mag, mag)
        self.last = torch.where(has, mean, self.last)
        self.seen = torch.where(has, torch.full_like(self.seen, newest), self.seen)

    def features(self, windows):
        """[W, N, DE] for forecasts issued at stream indices `windows` [W]."""
        age = ((windows[:, None] - self.seen[None]).to(self.ema.dtype) / CADENCE).clamp(0, MAX_AGE)
        available = (age < MAX_AGE).to(self.ema.dtype)
        stats = torch.stack([self.ema, self.mag, self.last], -1).clamp(-CLIP_E, CLIP_E)[None] * available[..., None]
        return torch.cat([stats, torch.where(available > 0, age / MAX_AGE, torch.ones_like(age))[..., None], available[..., None]], -1)


def hide_errors(c, hidden):
    """Copy of c in which the sensors marked in `hidden` [N] look as if no label had been released for them."""
    c = c.clone()
    c[:, hidden, DU:DU + 3] = 0.
    c[:, hidden, DU + 3] = 1.
    c[:, hidden, DU + 4] = 0.
    return c


# --------------------------------------------------------------------------
# Arms: modules, optimisers, forecasts and training steps
# --------------------------------------------------------------------------

class Arms:
    def __init__(self, names, n, device, opt, seed, has_geo):
        torch.manual_seed(seed)
        attn = Adapter('attn')
        template = {k: v.clone() for k, v in attn.state_dict().items()}               # identical start for all attention arms
        self.n, self.device, self.opt = n, device, opt
        self.gen = torch.Generator().manual_seed(7919 * seed + 3)
        self.arms, self.dropped = {}, []
        for name in names:
            kind, key = ARMS[name]
            if key == 'geo' and not has_geo:
                self.dropped.append(name)
                continue
            module = Adapter(kind, matched_hidden(parameters_of(attn)) if kind == 'self' else None)
            if kind == 'attn':
                module.load_state_dict(template)
            elif kind == 'fixed':
                module.load_state_dict({k: v for k, v in template.items() if k.split('.')[0] in ('phi', 'out', 'gate')})
            module.to(device)
            self.arms[name] = dict(module=module, kind=kind, key=key,
                                   optim=torch.optim.Adam(module.parameters(), lr=opt.lr), steps=0)
        self.seconds = dict(forward=0., update=0.)
        self.losses = []

    def correction(self, name, rec, stats=None):
        """Correction of one arm for every sensor of a forecast chunk, in flow units; [W, N, 12], no gradient."""
        arm, G = self.arms[name], rec['graph']
        nbr = None if arm['key'] is None else G.candidates(arm['key'])
        w = rec['u'].shape[0]
        sub = max(1, int(3e7 // max(self.n * max(G.k, 1) * DM, 1)))
        out = []
        with torch.no_grad():
            for i in range(0, w, sub):
                out.append(arm['module'](rec['u'][i:i + sub], rec['c'][i:i + sub], None, nbr, rec['ok'][i:i + sub],
                                         G.beh_w if arm['kind'] == 'fixed' else None, stats if i == 0 else None))
        return torch.cat(out) * rec['s'][None, :, None]

    def step(self, slices, hide=0.):
        """One optimiser step per arm on released windows. slices: [(record, lo, hi)] with windows lo:hi of the record."""
        began = time.time()
        m = min(self.opt.nodes_per_step, self.n)
        idx = torch.randperm(self.n, generator=self.gen)[:m].to(self.device)
        hidden = (torch.rand(self.n, generator=self.gen) < hide).to(self.device) if hide > 0 else None
        total = sum(hi - lo for _, lo, hi in slices)
        row = {}
        for name, arm in self.arms.items():
            module = arm['module']
            loss = 0.
            with torch.enable_grad():
                for rec, lo, hi in slices:
                    G = rec['graph']
                    c = rec['c'][lo:hi]
                    if hidden is not None:
                        c = hide_errors(c, hidden)
                    nbr = None if arm['key'] is None else G.candidates(arm['key'])[idx]
                    d = module(rec['u'][lo:hi], c, idx, nbr, rec['ok'][lo:hi], G.beh_w[idx] if arm['kind'] == 'fixed' else None)
                    s = rec['s'][idx][None, :, None]
                    target = (rec['y'][lo:hi][:, idx] - rec['yl'][lo:hi][:, idx]) / s
                    valid = rec['tv'][lo:hi][:, idx].to(d.dtype)
                    fit = (F.huber_loss(d, target, reduction='none', delta=self.opt.huber) * valid).sum() / valid.sum().clamp_min(1)
                    loss = loss + (fit + self.opt.l2 * (d ** 2).mean()) * ((hi - lo) / total)
                arm['optim'].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(module.parameters(), 1.)
                arm['optim'].step()
            arm['steps'] += 1
            row[name] = float(loss.detach())
        self.seconds['update'] += time.time() - began
        return row


# --------------------------------------------------------------------------
# One causal pass over a stream of windows (driven by g.rls_stream through on_window)
# --------------------------------------------------------------------------

class Session:
    def __init__(self, arms, timeline, pieces, static, opt, first_portion, train=True, keep=False, dump=0):
        self.arms, self.tl, self.pieces, self.static, self.opt = arms, timeline, pieces, static, opt
        self.n, self.device = arms.n, arms.device
        self.offsets = np.cumsum([0] + [c for _, _, c in pieces])
        self.T, self.first = int(self.offsets[-1]), first_portion
        self.train, self.keep = train, keep
        self.ctx = ErrorContext(self.n, self.device, opt.ema)
        self.records, self.kept, self.buf = [], [], []
        self.released, self.graph, self.graph_at = -1, None, None
        self.names = ['base'] + list(arms.arms)
        self.acc = {a: {p: g.ErrAcc(self.n, self.device) for p in ('val', 'test')} for a in self.names}
        self.curve, self.diag, self.refresh_seconds, self.refreshes = [], [], 0., 0
        self.dump = dump
        self.dumped = [] if dump else None
        self.state_bytes = 0

    def locate(self, t):
        k = int(np.searchsorted(self.offsets, t, side='right') - 1)
        raw, first, _ = self.pieces[k]
        return raw, first + t - int(self.offsets[k]), k

    def inputs(self, t0, t1):
        xs, t = [], t0
        while t < t1:
            raw, w, k = self.locate(t)
            take = min(t1, int(self.offsets[k + 1])) - t
            xs.append(np.array(g.get_windows(raw, w, take)[0], dtype=np.float32))
            t += take
        return torch.as_tensor(np.concatenate(xs), device=self.device)

    def on_window(self, t, pred, y, count):
        if t % CADENCE == 0:
            self.boundary(t)
        self.buf.append((pred, y.clone()))
        if (t + 1) % CADENCE == 0 or t + 1 == self.T:
            self.forecast(t + 1 - len(self.buf), t + 1)

    def take(self, lo, hi):
        """Cached forecast-time records of stream windows lo:hi."""
        out = []
        for rec in self.records:
            a, b = max(lo, rec['t0']), min(hi, rec['t1'])
            if a < b:
                out.append((rec, a - rec['t0'], b - rec['t0']))
        return out

    def boundary(self, t):
        upto = t - DELAY                                               # labels of windows <= upto have matured
        if upto > self.released:
            slices = self.take(self.released + 1, upto + 1)
            if slices:
                if self.train:
                    for _ in range(self.opt.online_steps):
                        self.arms.losses.append(dict(t=int(t), phase='online', **self.arms.step(slices)))
                for rec, lo, hi in slices:
                    s = rec['s'][None, :, None]
                    self.ctx.release((rec['y'][lo:hi] - rec['yl'][lo:hi]) / s, rec['tv'][lo:hi], rec['t0'] + hi - 1)
            self.released = upto
            self.records = [r for r in self.records if r['t1'] - 1 > upto]
        if self.graph is None or t - self.graph_at >= self.opt.refresh:
            raw, w, _ = self.locate(t)
            self.graph = Graph(self.tl, self.tl.visible(raw, w), self.opt.history_rows, self.opt.neighbours, self.static, self.device)
            self.graph_at = t
            self.refresh_seconds += self.graph.seconds
            self.refreshes += 1

    def forecast(self, t0, t1):
        began = time.time()
        yl = torch.stack([p for p, _ in self.buf])                     # [W, N, 12] float64, as issued by the local path
        y = torch.stack([v for _, v in self.buf])
        self.buf = []
        G = self.graph
        u, ok = summarise(self.inputs(t0, t1), yl.to(torch.float32), G.mean, G.std)
        c = torch.cat([u, self.ctx.features(torch.arange(t0, t1, device=self.device))], -1)
        rec = dict(t0=t0, t1=t1, u=u, c=c, ok=ok, yl=yl.to(torch.float32), y=y.to(torch.float32), tv=g.target_valid(y), s=G.std, graph=G)
        cut = max(0, min(self.first - t0, t1 - t0))
        row, diag, finals = dict(t0=int(t0)), dict(t0=int(t0)), []
        for name in self.names:
            stats = {}
            final = yl if name == 'base' else yl + self.arms.correction(name, rec, stats).to(torch.float64)
            if cut:
                self.acc[name]['val'].add(final[:cut], y[:cut])
            if cut < t1 - t0:
                self.acc[name]['test'].add(final[cut:], y[cut:])
            valid = g.target_valid(y)
            row[name] = float(torch.where(valid, (final - y).abs(), torch.zeros_like(y)).sum())
            if name != 'base':
                stats['mean_abs_correction'] = float((final - yl).abs().mean())
                diag[name] = stats
            if self.dumped is not None and t0 < self.dump:
                finals.append(final[:max(0, self.dump - t0)].to(torch.float32).cpu())
        row['valid'] = int(g.target_valid(y).sum())
        self.curve.append(row)
        self.diag.append(diag)
        if finals:
            self.dumped.append(torch.stack(finals, 1))                 # [W, arms, N, 12]
        self.records.append(rec)
        if self.keep:
            self.kept.append(rec)
        self.state_bytes = max(self.state_bytes, sum(v.numel() * v.element_size() for r in self.records for v in r.values() if torch.is_tensor(v)))
        self.arms.seconds['forward'] += time.time() - began


def warm_up(arms, records, opt):
    """Passes over the replayed chunks in time order; in every step a random share of sensors has its error context hidden."""
    for epoch in range(opt.warm_epochs):
        for rec in records:
            arms.losses.append(dict(t=int(rec['t0']), phase='warm%d' % epoch, **arms.step([(rec, 0, rec['t1'] - rec['t0'])], opt.warm_hide)))


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def steps_per_day(B, opt):
    return opt.steps_per_day or (24 if B.meta['dataset'].upper().startswith('TFNSW') else 288)


def run(B, seed, device, opt, out):
    n, z = B.n, B.z
    arm = next(a for a in e3.arms_of(B, seed) if a['name'] == opt.base_arm)
    adj = B.adjacency(device)
    weights = {arm['ckpt']: B.weights(seed, arm['ckpt'], device)} if arm['base'] == 'gcn' else {}
    with torch.no_grad():
        reference = e3.check_reference(B, seed, [arm], weights, adj, device)
    train, val, test = z['raw_train'], z['raw_val'], z['raw_test']
    fit_n = min(e3.FIT_WINDOWS, g.n_windows(train))
    fit_first = g.n_windows(train) - fit_n
    n_val = g.n_windows(val) if not opt.max_windows else min(opt.max_windows, g.n_windows(val))
    n_test = g.n_windows(test) if not opt.max_windows else min(opt.max_windows, g.n_windows(test))
    select_n = min(e3.SELECT_WINDOWS, n_val)
    y_fit = g.clean_y(g.get_windows(train, fit_first, fit_n)[1])
    scale = float(y_fit[g.target_valid(y_fit)].std())
    del y_fit

    def fitted(arm_list, ridge, forget, first, count):
        _, fn = e3.make_chunks(B, weights, adj, device, arm_list, [(train, first, count)])
        model = g.NodeRLS(n * len(arm_list), g.Y_LEN, scale, ridge, forget, device)
        step = 256 if n * len(arm_list) <= 24000 else 64
        for s in range(0, count, step):
            x, b, y, valid, _ = fn(slice(s, min(s + step, count)))
            model.fit(x, b, y, valid)
        return model

    # ---- hyper-parameters of the local path: exactly the g29 selection ----
    began = time.time()
    grid = [(r, f) for r in e3.RIDGES for f in e3.FORGETS]
    with torch.no_grad():
        ridge = torch.tensor([r for r, _ in grid], dtype=torch.float64).repeat_interleave(n)
        forget = torch.tensor([f for _, f in grid], dtype=torch.float64).repeat_interleave(n)
        model = fitted([arm] * len(grid), ridge, forget, fit_first, fit_n)
        T, fn = e3.make_chunks(B, weights, adj, device, [arm] * len(grid), [(val, 0, select_n)])
        acc = g.ErrAcc(n * len(grid), device)
        g.rls_stream(model, T, fn, DELAY, CADENCE, lambda t, pred, y, count: acc.add(pred, y), chunk=e3.chunk_of(n * len(grid)))
        arrays = acc.arrays()
    scores = [dict(ridge=r, forget=f, validation_H12_MAE=g.summarize(*(arrays[k][i * n:(i + 1) * n] for k in ('abs', 'sq', 'cnt')))['H12_MAE'])
              for i, (r, f) in enumerate(grid)]
    best = min(scores, key=lambda v: v['validation_H12_MAE'])
    if opt.fixed_selection:                                            # self-tests only
        best = dict(zip(('ridge', 'forget'), map(float, opt.fixed_selection.split(','))), validation_H12_MAE=None, fixed=True)
    del model, acc
    select_seconds = time.time() - began

    # ---- adapter arms ----
    period = steps_per_day(B, opt)
    opt.refresh = opt.refresh or period
    timeline = Timeline(train, val, test, period)
    static = static_candidates(n, opt.neighbours, adj, seed, device)
    arms = Arms([a for a in opt.arms.split(',') if a], n, device, opt, seed, static['geo'] is not None)

    # ---- warm-up: causal replay on the second half of the fit windows ----
    began = time.time()
    replay = None
    if opt.warm_epochs > 0:
        half = fit_n // 2
        with torch.no_grad():
            local = fitted([arm], best['ridge'], best['forget'], fit_first, half)
            T, fn = e3.make_chunks(B, weights, adj, device, [arm], [(train, fit_first + half, fit_n - half)])
            replay = Session(arms, timeline, [(train, fit_first + half, fit_n - half)], static, opt, T, train=False, keep=True)
            g.rls_stream(local, T, fn, DELAY, CADENCE, replay.on_window, chunk=e3.chunk_of(n))
        warm_up(arms, replay.kept, opt)
        del local
    warm_seconds = time.time() - began
    warm_steps = {name: a['steps'] for name, a in arms.arms.items()}

    # ---- the stream: local path as in g29, corrections on top ----
    began = time.time()
    with torch.no_grad():
        model = fitted([arm], best['ridge'], best['forget'], fit_first, fit_n)
        pieces = [(val, 0, n_val), (test, 0, n_test)]
        T, fn = e3.make_chunks(B, weights, adj, device, [arm], pieces)
        session = Session(arms, timeline, pieces, static, opt, n_val, train=opt.online_steps > 0, dump=opt.dump_forecasts)
        g.rls_stream(model, T, fn, DELAY, CADENCE, session.on_window, chunk=e3.chunk_of(n))
        raw_acc = {p: g.ErrAcc(n, device) for p in ('val', 'test')}       # backbone alone on the same windows
        for start in range(0, T, 512):
            x, b, y, _, _ = fn(slice(start, min(start + 512, T)))
            cut = max(0, min(n_val - start, len(x)))
            if cut:
                raw_acc['val'].add(b[:cut], y[:cut])
            if cut < len(x):
                raw_acc['test'].add(b[cut:], y[cut:])
    stream_seconds = time.time() - began

    ref_year = B.year - 5 if (B.year - 5) in B.ckpt_years[1:] else None
    mask = B.new_mask(ref_year) if ref_year is not None else None
    cohorts = dict(all=np.ones(n, bool))
    if mask is not None and 0 < mask.sum() < n:
        cohorts.update(old=~mask, new=mask)
    rows = []
    for portion in ('val', 'test'):
        raw = raw_acc[portion].arrays()
        for name in session.names:
            cal = session.acc[name][portion].arrays()
            for cohort, m in cohorts.items():
                rows.append(dict(dataset=B.meta['dataset'], year=B.year, seed=seed, portion=portion, arm=name, base=arm['base'],
                                 base_arm=arm['name'], checkpoint_year=arm['ckpt'], norm_year=arm['norm'], cadence=CADENCE,
                                 cohort=cohort, nodes=int(m.sum()), ridge=best['ridge'], forget=best['forget'],
                                 windows=int(n_val if portion == 'val' else n_test),
                                 validation_portion_contains_selection_windows=bool(portion == 'val'),
                                 calibrated=g.summarize(cal['abs'], cal['sq'], cal['cnt'], m),
                                 uncalibrated=g.summarize(raw['abs'], raw['sq'], raw['cnt'], m)))
    curve = {name: np.asarray([r[name] for r in session.curve]) for name in session.names}
    np.savez_compressed(out / 'e9_curves.npz', chunk_start=np.asarray([r['t0'] for r in session.curve]),
                        valid=np.asarray([r['valid'] for r in session.curve]), n_val=n_val,
                        **{'abs/' + k: v for k, v in curve.items()},
                        **({'forecasts': torch.cat(session.dumped).numpy(), 'forecast_arms': np.asarray(session.names)} if session.dumped else {}))
    mean_of = lambda key, name: float(np.mean([d[name][key] for d in session.diag if name in d and key in d[name]] or [float('nan')]))
    report = dict(
        selection=dict(grid=scores, chosen=best, selection_windows=int(select_n)), scale=scale, fit_windows=int(fit_n),
        n_val=int(n_val), n_test=int(n_test), steps_per_day=period, refresh_windows=int(opt.refresh),
        neighbours=int(session.graph.k), dropped_arms=arms.dropped, cohort_reference_year=ref_year if 'new' in cohorts else None,
        arms={name: dict(kind=a['kind'], candidates=a['key'], parameters=parameters_of(a['module']), warm_steps=warm_steps[name],
                         online_steps=a['steps'] - warm_steps[name], mean_abs_correction=mean_of('mean_abs_correction', name),
                         mean_gate=mean_of('gate', name),
                         attention_entropy=None if a['kind'] == 'self' else mean_of('entropy', name),
                         share_without_usable_neighbour=None if a['kind'] == 'self' else mean_of('isolated', name))
              for name, a in arms.arms.items()},
        seconds=dict(selection=select_seconds, warm_up=warm_seconds, stream=stream_seconds, adapter_forward=arms.seconds['forward'],
                     adapter_update=arms.seconds['update'], candidate_refresh=session.refresh_seconds + (replay.refresh_seconds if replay else 0.)),
        candidate_refreshes=int(session.refreshes), pending_cache_bytes_peak=int(session.state_bytes),
        error_context_bytes=int(n * 4 * 3 + n * 8), candidate_bytes=int(n * session.graph.k * 8 * 2),
        losses=arms.losses[-400:], loss_first=arms.losses[:5])
    return rows, arm, reference, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--backbone', choices=['gcn', 'stae', 'refresh'], default='gcn',
                    help="refresh (g34): a five-year-old STAEFormer whose embedding table is refreshed (arms of e12_online)")
    ap.add_argument('--base-arm', default='same', help='which frozen backbone the local path uses (an arm name of e3 / e5)')
    ap.add_argument('--arms', default='rra,self,rand,fixed,geo')
    ap.add_argument('--neighbours', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--l2', type=float, default=1e-2, help='penalty on the squared correction (rolling units)')
    ap.add_argument('--huber', type=float, default=1.)
    ap.add_argument('--nodes-per-step', type=int, default=256)
    ap.add_argument('--online-steps', type=int, default=1, help='optimiser steps per release; 0 freezes the adapter after warm-up')
    ap.add_argument('--warm-epochs', type=int, default=20, help='passes over the replayed fit windows; 0 = no warm-up')
    ap.add_argument('--warm-hide', type=float, default=.2, help='share of sensors whose error context is hidden in a warm-up step')
    ap.add_argument('--ema', type=float, default=.3)
    ap.add_argument('--history-rows', type=int, default=2016, help='rows of the rolling window behind candidates and sensor statistics')
    ap.add_argument('--refresh', type=int, default=0, help='windows between candidate refreshes; 0 = one day')
    ap.add_argument('--steps-per-day', type=int, default=0, help='0 = 288, or 24 for TFNSW')
    ap.add_argument('--max-windows', type=int, default=0, help='bounded smoke only; 0 = complete split')
    ap.add_argument('--dump-forecasts', type=int, default=0, help='self-tests only: store the forecasts of the first windows')
    ap.add_argument('--fixed-selection', default='', help='self-tests only: "ridge,forget" instead of the chosen pair')
    ap.add_argument('--role', default='unspecified')
    opt = ap.parse_args()
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if opt.backbone == 'stae':
        import e5_stae                                                 # noqa: F401  (replaces the bundle and forecast provider of e3)
    elif opt.backbone == 'refresh':
        import e12_online                                              # noqa: F401  (the same, with refreshed embedding tables)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if opt.device == 'auto' else opt.device))
    # Gradients of the neighbour gather are accumulated in a fixed order, so that a run can be repeated bit for bit.
    torch.use_deterministic_algorithms(True, warn_only=True)
    start = time.time()
    B = e3.Bundle(opt.bundle)
    assert opt.seed in B.meta['seeds']
    rows, arm, reference, report = run(B, opt.seed, device, opt, out)
    with (out / 'e3_metrics.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
    options = {k: v for k, v in vars(opt).items() if k not in ('bundle', 'out')}
    Path(out / 'e9_setup.json').write_text(json.dumps(dict(
        dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n, backbone=opt.backbone, base_arm=arm, reference_check=reference,
        options=options, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        torch=torch.__version__, numpy=np.__version__, peak_gpu_gib=peak, **report), indent=2) + '\n')
    if opt.backbone == 'refresh':
        (out / 'e10_info.json').write_text(json.dumps(e12_online.info_of(B), indent=2) + '\n')
    g.save_json(out / 'completed.json', dict(status='completed', rows=len(rows), seconds=time.time() - start,
                                             bounded_smoke=bool(opt.max_windows), role=opt.role))
    print(json.dumps(dict(status='e9_completed', dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n,
                          arms=list(report['arms']), seconds=round(time.time() - start, 1), peak_gpu_gib=peak)), flush=True)


if __name__ == '__main__':
    main()
