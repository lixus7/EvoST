"""E12 (g34 C): "borrow an identity by behaviour" and "refit the identities" as one method, with online updating.

g33: with the shared weights of a five-year-old STAEFormer frozen, refitting only the per-sensor embedding table on the
fit windows of the target year recovers 85% of the loss; borrowing embeddings from behaviour-matched sensors without
training recovers 15%, and helps the sensors the old checkpoint does not know. Here the two are combined, and the table
keeps learning from the labels that the stream releases. The shared weights stay frozen in every arm; the stream, the
calibrator and the metrics are the unchanged e3_confirm.

Arms (base forecast; each uncalibrated and with the calibrator):

  same       same-year checkpoint (reference)
  stale5o    five-year-old checkpoint, its own embeddings (mean embedding for sensors it does not know)      (g30, g33)
  new_prof   as stale5o, sensors it does not know take the mean old embedding of 5 behaviour-matched donors  (g33)
  fit_emb    table fitted on the fit windows, starting from the stale5o table                                (g33)
  fit_prof   table fitted on the fit windows, starting from the new_prof table       (borrow, then refit)
  on_stale   the stale5o table, updated online                                       (no offline fit)
  on_prof    the new_prof table, updated online                                      (borrow, no offline fit)
  on_emb     the fit_emb table, updated online
  on_fit     the fit_prof table, updated online                                      (borrow, refit, online: the full method)
  on_same    the same-year checkpoint, its own table updated online                  (the like-for-like reference of the online arms)
  on_fit_lo, on_fit_hi, on_same_lo, on_same_hi   on_fit and on_same with a third of / three times the online learning rate

Online updating. Labels are released as for the calibrator: at every 64th window the labels of all windows issued at
least 12 windows earlier become available. At such a boundary the table makes one pass over the newly released windows
(every 2nd, every 4th above 1600 sensors; the stride, batch size, loss and optimiser of the offline fit) and the windows
of the next 64 positions are forecast with the updated table (Adam, learning rate 1e-3, fixed on development data). A
base forecast is therefore computed from labels of
windows at least 12 positions older, and is never recomputed: the calibrator is updated with, and the uncalibrated error
is measured on, the forecast as it was issued. On the fit windows the calibrator of an online arm sees the forecasts of
the table the arm starts from.
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
import e10_refresh as e10
import g28_common as g

DELAY, CADENCE = e3.DELAY, e3.CADENCE
ONLINE = dict(lr=float(os.environ.get('G34_ONLINE_LR', '1e-3')))      # fixed on development data (PRESPEC.md); the override was for that sweep
LOWER, HIGHER = 3e-4, 3e-3                                            # the two other rates of the sweep, run as sensitivity arms
ARMS = [a for a in os.environ.get('G34_ARMS', '').split(',') if a]    # development only: a subset of the arms
LIVE = 'on:'                                                          # variant 'on:<start>', or 'on@<rate>:<start>' for another learning rate


def is_live(arm):
    return arm['base'] == 'stae' and str(arm.get('variant') or '').startswith(('on:', 'on@'))


def parse(variant):
    """(learning rate, variant of the table the arm starts from) of an online variant."""
    kind, start = variant.split(':', 1)
    return (float(kind[3:]) if kind.startswith('on@') else ONLINE['lr']), start


def at(rate, start):
    return 'on@%g:%s' % (rate, start)


def segments(pieces, lo, hi):
    """[(raw, first window, count)] covering stream positions lo:hi of a concatenation of pieces."""
    offsets = np.cumsum([0] + [c for _, _, c in pieces])
    out, t = [], lo
    while t < hi:
        k = int(np.searchsorted(offsets, t, side='right') - 1)
        raw, first, _ = pieces[k]
        take = min(hi, int(offsets[k + 1])) - t
        out.append((raw, first + t - int(offsets[k]), take))
        t += take
    return out


class Memo:
    """A bundle archive whose arrays are read once, so that a raw split is one object wherever it is asked for."""

    def __init__(self, z):
        self._z, self.files, self._seen = z, z.files, {}

    def __getitem__(self, key):
        if key not in self._seen:
            self._seen[key] = self._z[key]
        return self._seen[key]


class OnlineTable:
    """Embedding table of a frozen checkpoint that learns from released labels while the stream runs."""

    def __init__(self, B, ckpt, start, name, device, label=None, lr=None):
        self.B, self.ckpt, self.name, self.device, self.label = B, ckpt, name, device, label or name
        self.lr = ONLINE['lr'] if lr is None else float(lr)
        self.model = B.model(ckpt, device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.first = start.clone()
        self.table = nn.Parameter(start.clone())
        self.optim = torch.optim.Adam([self.table], lr=self.lr)
        self.stride, self.size = e10.fit_stride(B.n), [e10.fit_batch(B.n)]
        self.seq, self.out = [], []                        # per stream position: (raw split, window) and the forecast as issued
        self.block, self.absorbed = 0, -1
        self.steps, self.seconds, self.trace = 0, 0., []

    # ---- forecasts as issued ----
    def serve(self, pieces, lo, hi):
        """Base forecasts of stream positions lo:hi, [hi - lo, N, 12]. A stream is asked for in order; positions that
        were issued before (the selection windows are the head of the final stream) are returned as they were issued."""
        assert lo <= len(self.seq), 'the stream must be requested in order'
        p = lo
        for raw, first, take in segments(pieces, lo, min(hi, len(self.seq))):
            for j in range(take):
                assert self.seq[p][0] is raw and self.seq[p][1] == first + j, 'a different stream was issued at this position'
                p += 1
        total = int(sum(c for _, _, c in pieces))
        while len(self.seq) < hi:
            p = len(self.seq)
            block = p // CADENCE
            if block > self.block:
                self.advance(block)
            self.issue(segments(pieces, p, min((block + 1) * CADENCE, total)))
        return torch.stack(self.out[lo:hi])

    def issue(self, segs):
        B = self.B
        B._variant[(self.ckpt, self.name)] = self.table.detach()       # read by RefreshBundle._forward
        for raw, first, take in segs:
            x_raw = np.array(g.get_windows(raw, first, take)[0], dtype=np.float32)
            out = B.forward(('stae', self.ckpt, self.ckpt, self.name), x_raw, self.device)
            for j in range(take):
                self.seq.append((raw, first + j))
                self.out.append(out[j])

    # ---- learning from released labels ----
    def advance(self, block):
        """Absorb what is released at every boundary up to the one that opens `block`."""
        for b in range(self.block + 1, block + 1):
            upto = b * CADENCE - DELAY                                 # labels of positions <= upto have matured
            assert upto < len(self.seq)
            fresh = [p for p in range(self.absorbed + 1, upto + 1) if p % self.stride == 0]
            if fresh:
                self.learn(fresh, b)
            self.absorbed = upto
        self.block = block

    def learn(self, positions, block):
        began, model, mean, std = time.time(), self.model, *self.B.norm[self.ckpt]
        s, losses = 0, []
        while s < len(positions):
            def step(b):
                part = [self.seq[p] for p in positions[s:s + b]]
                x = np.stack([np.array(g.get_windows(raw, w, 1)[0][0], dtype=np.float32) for raw, w in part])
                y = g.clean_y(np.stack([np.array(g.get_windows(raw, w, 1)[1][0]) for raw, w in part])).to(self.device)
                x, valid = g.normalize_x(x, mean, std).to(self.device), g.target_valid(y)
                with torch.enable_grad():
                    model.backbone.adaptive_embedding = self.table
                    loss = ((model.fc(model.activation(model.backbone(x, None) + x)) - y).abs() * valid).sum() / valid.sum().clamp_min(1)
                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                self.optim.step()
                losses.append(float(loss.detach()))
                return b
            while True:                                                # on an out-of-memory error halve the batch and try again
                try:
                    s += step(self.size[0])
                    break
                except torch.cuda.OutOfMemoryError:
                    assert self.size[0] > 1, 'out of GPU memory with one window per step'
                    self.size[0] = max(1, self.size[0] // 2)
                    torch.cuda.empty_cache()
        model.backbone.adaptive_embedding = nn.Parameter(self.first.clone(), requires_grad=False)
        self.steps += len(losses)
        self.seconds += time.time() - began
        self.trace.append(dict(block=int(block), windows=len(positions), steps=len(losses), loss_before_step=float(np.mean(losses)),
                               mean_abs_change=float((self.table.detach() - self.first).abs().mean())))

    def report(self):
        return dict(checkpoint_year=int(self.ckpt), lr=self.lr, stride=self.stride, batch_windows=self.size[0], positions_issued=len(self.seq),
                    releases=len(self.trace), steps=self.steps, seconds_learning=self.seconds, trace=self.trace)


class OnlineBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.z = Memo(self.z)
        self._live = {}

    def live(self, arm, device):
        key = (arm['ckpt'], arm['variant'])
        if key not in self._live:
            rate, start = parse(arm['variant'])
            table = e5.StaeBundle.table(self, arm['ckpt'], device) if start == 'full' else self.variant(arm['ckpt'], start, device)
            self._live[key] = OnlineTable(self, arm['ckpt'], table, arm['variant'], device, arm['name'], rate)
        return self._live[key]


def started_from(arm):
    """The arm whose table an online arm starts from (its forecasts on the fit windows)."""
    return dict(arm, variant=parse(arm['variant'])[1])


def make_chunks(B, weights, adj, device, arms, pieces):
    """Contract of e5_stae.make_chunks. Online arms: on train windows the forecast of the table they start from, on the
    evaluation stream the forecast as issued."""
    if not any(is_live(a) for a in arms):
        return e5.make_chunks(B, weights, adj, device, arms, pieces)
    train = B.z['raw_train']
    on_train = [raw is train for raw, _, _ in pieces]
    assert all(on_train) or not any(on_train), 'a stream does not mix train and evaluation windows'
    if on_train[0]:
        return e5.make_chunks(B, weights, adj, device, [started_from(a) if is_live(a) else a for a in arms], pieces)
    T = int(sum(c for _, _, c in pieces))

    def chunk_fn(s):
        segs = segments(pieces, s.start, s.stop)
        x_raw = np.concatenate([np.array(g.get_windows(raw, first, take)[0], dtype=np.float32) for raw, first, take in segs])
        y = torch.cat([g.clean_y(g.get_windows(raw, first, take)[1]) for raw, first, take in segs]).to(device)
        memo, xs, bs = {}, [], []
        for arm in arms:
            key = e5.key_of(arm)
            if key not in memo:
                x = g.normalize_x(x_raw, *B.norm[arm['norm']]).to(device)
                if is_live(arm):
                    b = B.live(arm, device).serve(pieces, s.start, s.stop).to(device)
                elif arm['base'] == 'stae':
                    b = torch.cat([B.base(key, raw, first, take, device) for raw, first, take in segs]).to(device)
                elif arm['base'] == 'zero':
                    b = torch.zeros_like(y)
                else:
                    last = torch.nan_to_num(torch.from_numpy(x_raw[..., -1:]), nan=0., posinf=0., neginf=0.)
                    b = last.to(device).expand_as(y)
                memo[key] = (x.to(torch.float64), b.to(torch.float64))
            xs.append(memo[key][0])
            bs.append(memo[key][1])
        r = len(arms)
        return (torch.cat(xs, 1), torch.cat(bs, 1), y.repeat(1, r, 1).to(torch.float64), g.window_valid(y).repeat(1, r), None)
    return T, chunk_fn


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, variant, ckpt=None: dict(name=name, base='stae', ckpt=ckpt or B.old, norm=ckpt or B.old, age=B.year - (ckpt or B.old), variant=variant)
    arms = [arm('same', 'full', B.year), arm('stale5o', 'full'), arm('new_prof', 'new_prof'), arm('fit_emb', 'fit'), arm('fit_prof', 'fit_prof'),
            arm('on_stale', LIVE + 'full'), arm('on_prof', LIVE + 'new_prof'), arm('on_emb', LIVE + 'fit'), arm('on_fit', LIVE + 'fit_prof'),
            arm('on_same', LIVE + 'full', B.year),
            arm('on_fit_lo', at(LOWER, 'fit_prof')), arm('on_fit_hi', at(HIGHER, 'fit_prof')),
            arm('on_same_lo', at(LOWER, 'full'), B.year), arm('on_same_hi', at(HIGHER, 'full'), B.year)]
    if ARMS:
        assert set(ARMS) <= {a['name'] for a in arms}, ARMS
        arms = [a for a in arms if a['name'] in ARMS]
    return arms


def info_of(B):
    B.info['online'] = {live.label: dict(live.report(), start=parse(live.name)[1]) for live in B._live.values()}
    return B.info


e3.Bundle = OnlineBundle
e3.arms_of = arms_of
e3.make_chunks = make_chunks
e3.check_reference = e10.check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)          # so that a run can be repeated on the same machine
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(info_of(e10.RefreshBundle.last), indent=2) + '\n')
