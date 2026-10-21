"""E17 (g38 B): sensors that have just arrived. Short-history identity estimation with and without anchors, g29 protocol.

In g33 the sensors a five-year-old checkpoint does not know ("new") had the full 2,048 fit windows of the target year
before the stream started. Here they are treated as having arrived only h windows before the stream (h in HISTORIES):
their rows before that are never read — not as inputs, not as labels, not for behaviour matching, not by the calibrator.
The sensors the old checkpoint knows ("mature", K) have the full history. Shared weights of the old checkpoint stay frozen.

  1. Mature identities: the table rows of K are refitted on the 2,048 fit windows with the model run on K alone (the
     network as it was before the arrivals). This is E_K.
  2. New identities, per scheme, K rows fixed at E_K, loss on the new sensors only, the last h fit windows (h/4 holdout):
       mean     global mean of E_K, no fit                         prof0 / rand0 / geo0   the anchor below, no fit
       ind      fit from the global mean, no anchor term           pinit                  fit from the behaviour anchor, no anchor term
       gmean    fit from the global mean, pulled towards it
       rand     fit from the mean of K random mature rows of E_K, pulled towards it
       prof     fit from the mean of the K mature sensors whose series over the new sensor's observed rows is closest, pulled towards it
       geo      the same with the K mature sensors of largest adjacency weight
     Anchor term: BETA * L0 * mean_i ||e_i - a_i||^2 / ||a_i||^2 (L0 = forecast loss at the start, a constant), the same
     BETA for every anchor. Fits follow e16_budget (g33's rate, epochs, patience, batches; the full budget's windows per epoch).
  3. Stream: unchanged protocol on all sensors. Base forecasts come from the full network with the scheme's table, except
     in fit windows before the arrival, where the network is K alone. The calibrator of a new sensor is fitted on its last
     h fit windows only. Arm c<h>_same gives the same-year model's forecasts with that calibrator budget (what the short
     calibrator history alone costs); arm konly runs K alone for the whole stream (are mature sensors harmed by arrivals?).

Reference arms: same, stale5o, fit_emb (g33: everybody has the full history). e3_confirm reports every arm for the
cohorts old / new and the portions val (the first half of the stream, right after the arrival) / test.
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
import e16_budget as e16
import g28_common as g

HISTORIES = (64, 256)
if os.environ.get('G38_HISTORIES'):                                      # bounded smoke only
    HISTORIES = tuple(int(v) for v in os.environ['G38_HISTORIES'].split(','))
BETA = float(os.environ.get('G38_BETA', '0.1'))                          # fixed in PRESPEC.md from the development runs; the override is for those runs only
FREE = ('mean', 'prof0', 'rand0', 'geo0')                                # no fit
FITTED = dict(ind=('mean', None), pinit=('prof', None), gmean=('mean', 'mean'), rand=('rand', 'rand'), prof=('prof', 'prof'), geo=('geo', 'geo'))   # scheme -> (start, anchor)
MIN_NEW = 10                                                             # fewer new sensors than this: the cold arms are not run
if os.environ.get('G38_SCHEMES'):                                        # development probe of BETA only: a subset of the schemes
    FREE = tuple(s for s in FREE if s in os.environ['G38_SCHEMES'].split(','))
    FITTED = {k: v for k, v in FITTED.items() if k in os.environ['G38_SCHEMES'].split(',')}


def parse(variant):
    """'c64:prof' -> (64, 'prof'); None for other variants."""
    if isinstance(variant, str) and variant.startswith('c') and ':' in variant:
        h, scheme = variant[1:].split(':')
        return int(h), scheme
    return None


class ColdBundle(e16.BudgetBundle):
    def __init__(self, path):
        super().__init__(path)                                              # raw splits are single objects (e16_budget.Memo)
        self._cold, self._ek, self._anchor = {}, None, {}

    # ---- who is new ----
    def split(self, device):
        known = self.known(self.old, device)
        return torch.nonzero(known)[:, 0], torch.nonzero(~known)[:, 0]

    def arrival(self, h):
        """First fit window (index into raw_train) at which a new sensor exists for history h; its first visible row is the same number."""
        return g.n_windows(self.z['raw_train']) - h

    # ---- one fit of some rows of a table on a sub-network ----
    def fit_rows(self, device, nodes, start, trainable, first, n_fit, loss_nodes, anchor=None, beta=0., tag=None):
        """nodes: indices of the sensors that form the network; start [12, len(nodes), 80]; trainable / loss_nodes: bool over nodes.
        Windows: the n_fit fit windows from `first` (the last quarter as holdout). anchor [12, trainable.sum(), 80] or None."""
        began, raw = time.time(), self.z['raw_train']
        stride = e10.fit_stride(self.n)
        hold_n = n_fit // 4
        train = np.arange(first, first + n_fit - hold_n, stride)
        hold = np.arange(first + n_fit - hold_n, first + n_fit, stride)
        per_epoch = len(np.arange(0, e16.FULL - e16.FULL // 4, stride))
        model = self.model(self.old, device)
        for p in model.parameters():
            p.requires_grad_(False)
        mean, std = self.norm[self.old]
        nodes_cpu = nodes.cpu().numpy()
        weight = loss_nodes.to(device)[None, :, None]

        def batch(idx):
            x = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in idx])[:, nodes_cpu]
            y = g.clean_y(np.stack([np.array(g.get_windows(raw, int(i), 1)[1][0]) for i in idx])[:, nodes_cpu]).to(device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y) & weight

        def predict(x, table):
            model.backbone.adaptive_embedding = table if isinstance(table, nn.Parameter) else nn.Parameter(table, requires_grad=False)
            return model.fc(model.activation(model.backbone(x, None) + x))

        size = [e10.fit_batch(len(nodes_cpu))]

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
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)
        history = [holdout(start)]
        best = (history[0], start.clone(), 0)
        pull = None
        if anchor is not None and beta > 0:
            norm = (anchor ** 2).sum((0, 2)).clamp_min(1e-12)                    # per trainable sensor
            pull = lambda: (((table[:, trainable.to(device)] - anchor) ** 2).sum((0, 2)) / norm).mean() * (beta * history[0])
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
                        if pull is not None:
                            loss = loss + pull()
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                    table.grad[:, frozen] = 0.                                   # rows that are not trained never move (Adam: zero gradient, zero step)
                    optim.step()
                    return b
                s += guarded(step)
            history.append(holdout(table.detach()))
            if history[-1] < best[0]:
                best = (history[-1], table.detach().clone(), epoch + 1)
            elif epoch + 1 - best[2] >= e10.FIT['patience']:
                break
        model.backbone.adaptive_embedding = nn.Parameter(e5.StaeBundle.table(self, self.old, device).clone(), requires_grad=False)
        assert torch.equal(best[1][:, frozen], start[:, frozen])
        if tag:
            self.info[tag] = dict(network_sensors=int(len(nodes_cpu)), trained_sensors=int(trainable.sum()), loss_sensors=int(loss_nodes.sum()), fit_windows=int(n_fit),
                                  train_windows=int(len(train)), holdout_windows=int(len(hold)), windows_per_epoch=int(per_epoch), batch_windows=size[0], lr=e10.FIT['lr'],
                                  beta=float(beta) if anchor is not None else None, holdout_mae=history, best_epoch=best[2],
                                  mean_abs_change=float((best[1][:, trainable.to(device)] - start[:, trainable.to(device)]).abs().mean()), seconds=time.time() - began)
        return best[1]

    # ---- step 1: the mature sensors, on the network without the arrivals ----
    def mature(self, device):
        if self._ek is None:
            K, _ = self.split(device)
            raw = self.z['raw_train']
            stale = e5.StaeBundle.table(self, self.old, device)[:, K]
            every = torch.ones(len(K), dtype=torch.bool)
            self._ek = self.fit_rows(device, K, stale, every, g.n_windows(raw) - e16.FULL, e16.FULL, every, tag='mature').contiguous()
        return self._ek

    # ---- anchors of the new sensors: positions inside K, [new sensors, K_ANCHORS] ----
    def anchors(self, h, device):
        if h not in self._anchor:
            K, U = (v.cpu().numpy() for v in self.split(device))
            raw = np.asarray(self.z['raw_train'], np.float64)[self.arrival(h):]          # only rows from the arrival on are read
            fin = np.isfinite(raw)
            v = np.where(fin, raw, 0.)
            a, ma, b, mb = v[:, U].T, fin[:, U].T.astype(np.float64), v[:, K].T, fin[:, K].T.astype(np.float64)
            both = ma @ mb.T
            dist = ((a ** 2) @ mb.T + ma @ (b ** 2).T - 2 * a @ b.T) / np.maximum(both, 1.)
            dist[both < .2 * raw.shape[0]] = np.inf                                       # too few common observations: not a donor
            dist[:, mb.mean(1) < .2] = np.inf
            k = min(e10.K, len(K))
            rng = np.random.default_rng(1000003 * self.seed + self.year + h)
            out = dict(rand=np.stack([rng.choice(len(K), k, replace=False) for _ in range(len(U))]))
            blind = np.isfinite(dist).sum(1) < k                                        # a new sensor without usable observations: random anchors
            dist[blind] = 0.
            out['prof'] = e10.nearest(dist, k)
            out['prof'][blind] = out['rand'][blind]
            gcn = self.path.parent.parent / 'gcn' / self.path.name
            out['geo'] = None
            if gcn.exists():
                zg, _ = g.load_bundle(gcn)
                assert [str(s) for s in zg['sensor_ids']] == [str(s) for s in self.z['sensor_ids']]
                out['geo'] = e10.nearest(-np.asarray(zg['adj_norm'], np.float64)[np.ix_(U, K)], k)
            self.info['anchors@%d' % h] = dict(rows_read=int(raw.shape[0]), neighbours=int(k), geo=bool(out['geo'] is not None),
                                               new_sensors_without_behaviour_anchor=int(blind.sum()))
            self._anchor[h] = out
        return self._anchor[h]

    # ---- step 2: the table of one scheme ----
    def cold_table(self, h, scheme, device):
        key = (h, scheme)
        if key not in self._cold:
            K, U = self.split(device)
            ek = self.mature(device)
            A = self.anchors(h, device)
            gmean = ek.mean(1, keepdim=True).expand(-1, len(U), -1)
            rows = dict(mean=gmean)
            for kind in ('prof', 'rand', 'geo'):
                if A[kind] is not None:
                    rows[kind] = ek[:, torch.as_tensor(A[kind], device=device)].mean(2)
            full = torch.empty(ek.shape[0], self.n, ek.shape[2], device=device)
            full[:, K] = ek
            if scheme in FREE:
                full[:, U] = rows[scheme.rstrip('0')]
            else:
                start, anchor = FITTED[scheme]
                full[:, U] = rows[start]
                trainable = torch.zeros(self.n, dtype=torch.bool)
                trainable[U.cpu()] = True
                full = self.fit_rows(device, torch.arange(self.n, device=device), full, trainable, self.arrival(h), h, trainable,
                                     anchor=None if anchor is None else rows[anchor].contiguous(), beta=BETA, tag='c%d:%s' % (h, scheme))
            assert torch.equal(full[:, K], ek) and bool(torch.isfinite(full).all())
            self._cold[key] = full.contiguous()
        return self._cold[key]

    # ---- forecasts ----
    def variant(self, ckpt, name, device):
        if name == 'fit':                                                        # g33's arm on the full history (budget 2048)
            e16.set_budget(e16.FULL)
        return super().variant(ckpt, name, device)

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        cold = parse(variant)
        if cold is None and variant != 'konly':
            return super()._forward(key, x_raw, device, sub)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model = self.model(ckpt, device)
        if variant == 'konly':                                                   # the network without the arrivals; they get the last observed value
            K, _ = self.split(device)
            last = torch.nan_to_num(torch.from_numpy(x_raw[..., -1:]), nan=0., posinf=0., neginf=0.).to(device)
            out = last.expand(-1, -1, g.Y_LEN).clone()
            ek = self.mature(device)
            for i in range(0, len(x), sub):
                out[i:i + sub, K] = model(x[i:i + sub][:, K], ek)
        else:
            table = self.cold_table(cold[0], cold[1], device)
            out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))

    def base(self, key, raw, first, count, device):
        """Cold arms: fit windows before the arrival come from the network without the new sensors (the forecasts of arm konly)."""
        cold = parse(key[3])
        if cold is None:
            return super().base(key, raw, first, count, device)
        h, scheme = cold
        if scheme == 'same':                                                     # the same-year model's forecasts; only its calibrator differs
            return super().base(key[:3] + ('full',), raw, first, count, device)
        cut = self.arrival(h)
        if raw is not self.z['raw_train'] or first >= cut:
            return super().base(key, raw, first, count, device)
        before = key[:3] + ('konly',)
        if first + count <= cut:
            return super().base(before, raw, first, count, device)
        return torch.cat([super().base(before, raw, first, cut - first, device), super().base(key, raw, cut, first + count - cut, device)])


def arms_of(B, seed):
    B.seed = seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arm = lambda name, ckpt, variant, **kw: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant, **kw)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit')]
    n_new = int((~B.known(B.old, device)).sum())
    B.info.update(sensors=B.n, new_sensors=n_new, cold_arms=bool(n_new >= MIN_NEW and n_new < B.n), beta=BETA, histories=list(HISTORIES))
    if not B.info['cold_arms']:
        return arms
    arms.append(arm('konly', B.old, 'konly'))
    geo = (B.path.parent.parent / 'gcn' / B.path.name).exists()
    for h in HISTORIES:
        arms.append(arm('c%d_same' % h, B.year, 'c%d:same' % h, history=h))
        for scheme in FREE + tuple(FITTED):
            if geo or not scheme.startswith('geo'):
                arms.append(arm('c%d_%s' % (h, scheme), B.old, 'c%d:%s' % (h, scheme), history=h))
    return arms


def make_chunks(B, weights, adj, device, arms, pieces):
    """e5_stae.make_chunks; in the fit piece a new sensor of a cold arm is not valid before its arrival (the calibrator never reads it)."""
    T, fn = e5.make_chunks(B, weights, adj, device, arms, pieces)
    histories = [a.get('history') for a in arms]
    if not any(histories) or len(pieces) != 1 or pieces[0][0] is not B.z['raw_train']:
        return T, fn
    new = torch.nonzero(~B.known(B.old, device))[:, 0]
    first = pieces[0][1]

    def chunk_fn(s):
        x, b, y, valid, extra = fn(s)
        index = first + torch.arange(s.start, s.stop, device=device)
        valid = valid.clone()
        for i, h in enumerate(histories):
            if h:
                before = torch.nonzero(index < B.arrival(h))[:, 0]
                if len(before):
                    valid[before[:, None], (i * B.n + new)[None, :]] = False
        return x, b, y, valid, extra
    return T, chunk_fn


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


e3.Bundle = ColdBundle
e3.arms_of = arms_of
e3.make_chunks = make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e16.set_budget(e16.FULL)                                             # the calibrator and its selection use the standard windows
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(e10.RefreshBundle.last.info, indent=2) + '\n')
