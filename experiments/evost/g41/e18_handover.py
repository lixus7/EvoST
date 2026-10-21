"""E18 (g40): borrow at arrival, refit when the history suffices. The hand-over from g38 B to g38 A, g29 protocol.

g38 found two things on a frozen five-year-old STAEFormer: (A) refitting the per-sensor embedding needs about 1,024 labelled
windows, less than a day is of no use; (B) a sensor that has just arrived is best given, without training, the mean row of the
five mature sensors whose recent series look most like its own (arm c64_prof0) - and that row was then never updated. Here
the two are joined: a new sensor starts with a borrowed row, and when the stream reaches window EVENT (1,024) its row is
refitted on the EVENT most recent windows whose labels have been released by then.

Setting of e17_coldstart with history H = 64: the sensors the old checkpoint does not know arrive 64 windows before the
stream; nothing of theirs from before the arrival enters a forecast, a fit, a matching or a calibrator. Mature sensors keep
the rows E_K fitted on the network without the arrivals; shared weights stay frozen.

  the event   Forecasts of stream windows t < EVENT use the table of the arrival (e17's c64:<start>); forecasts of t >= EVENT
              use the table in which the rows of the new sensors were refitted (g33's rate, epochs, patience and batches;
              e16_budget's windows per epoch; the last quarter of the windows as holdout; loss on the new sensors only). The
              fit windows are the EVENT most recent windows whose labels are released when window EVENT is forecast: stream
              windows 0 .. EVENT - 12 and the last 11 windows before the stream (rls_stream releases window i at t >= i + 12).
  calibrator  _k: untouched by the event (it goes on absorbing labels, now of other forecasts).
              _r: the calibrators of the new sensors are restarted at the event: exact ridge on the same fit windows with
                  the forecasts of the refitted table, the arm's ridge and forgetting factor unchanged; then the stream goes on.
                  This is what the protocol does for every sensor before the stream (g33: the calibrator is fitted on the
                  windows the embeddings were fitted on).

Arms: same, stale5o, fit_emb, konly, c64_same, c64_mean, c64_prof0 (all as in e17) and s_<start>_<k|r> for start in mean,
prof0. The ridge and the forgetting factor of an arm are chosen on the first 1,024 stream windows as always; these are
exactly the windows before the event, so a scheduled arm chooses what its static twin chooses.

The stream driver (g28_common.rls_stream) and the protocol (e3_confirm.run) are the unchanged files. The event is carried
by a subclass of the calibrator that counts its own calls: the restart happens inside the forecast call of stream window
EVENT, after the driver has absorbed every label released by then (asserted: exactly EVENT - 11 windows absorbed).
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
import e17_coldstart as c17
import g28_common as g

H = 64                                                                   # history at the arrival: the one g38's rules were read at
EVENT = int(os.environ.get('G40_EVENT', '1024'))                         # the override is for bounded smokes only
RELEASED = EVENT - e3.DELAY + 1                                          # stream windows 0 .. EVENT - DELAY: labels released when window EVENT is forecast
TAIL = EVENT - RELEASED                                                  # windows before the stream that complete the EVENT fit windows
STARTS = ('mean', 'prof0')
assert 0 <= TAIL <= H and EVENT % e5.BLOCK_WINDOWS == 0, (EVENT, TAIL)
PlainRLS = g.NodeRLS


class EventRLS(PlainRLS):
    """The protocol's calibrator, counting its calls; an event attached to it runs once, inside the forecast of stream window EVENT."""
    last = None

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.updates, self.predicts, self.event = 0, 0, None
        EventRLS.last = self

    def update(self, x, b, y, eligible):
        super().update(x, b, y, eligible)
        self.updates += 1                                                 # one call per absorbed stream window, in order

    def predict(self, x, b):
        if self.event is not None and self.predicts == EVENT:
            self.event(self)
        self.predicts += 1                                                # one call per forecast stream window, in order
        return super().predict(x, b)


class HandoverBundle(c17.ColdBundle):
    def __init__(self, path):
        super().__init__(path)
        self._post = {}

    def event_pieces(self):
        """The EVENT most recent windows with released labels when stream window EVENT is forecast."""
        train, val = self.z['raw_train'], self.z['raw_val']
        assert g.n_windows(val) >= EVENT, 'the validation portion is shorter than the event'
        return ([(train, g.n_windows(train) - TAIL, TAIL)] if TAIL else []) + [(val, 0, RELEASED)]

    def fit_windows(self, device, start, trainable, windows, tag):
        """e17_coldstart.fit_rows on the whole network for an explicit list of windows [(raw, index)], in time order; no anchor term."""
        began = time.time()
        n_fit, stride = len(windows), e10.fit_stride(self.n)
        hold_n = n_fit // 4
        train = np.arange(0, n_fit - hold_n, stride)
        hold = np.arange(n_fit - hold_n, n_fit, stride)
        per_epoch = len(np.arange(0, e16.FULL - e16.FULL // 4, stride))
        model = self.model(self.old, device)
        for p in model.parameters():
            p.requires_grad_(False)
        mean, std = self.norm[self.old]
        weight = trainable.to(device)[None, :, None]

        def batch(idx):
            x = np.stack([np.array(g.get_windows(windows[i][0], int(windows[i][1]), 1)[0][0], dtype=np.float32) for i in idx])
            y = g.clean_y(np.stack([np.array(g.get_windows(windows[i][0], int(windows[i][1]), 1)[1][0]) for i in idx])).to(device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y) & weight

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

        frozen = (~trainable).to(device)
        table = nn.Parameter(start.clone())
        optim = torch.optim.Adam([table], lr=e10.FIT['lr'])
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)
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
                    table.grad[:, frozen] = 0.                                   # rows that are not trained never move
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
        self.info[tag] = dict(network_sensors=int(self.n), trained_sensors=int(trainable.sum()), loss_sensors=int(trainable.sum()), fit_windows=int(n_fit),
                              train_windows=int(len(train)), holdout_windows=int(len(hold)), windows_per_epoch=int(per_epoch), batch_windows=size[0], lr=e10.FIT['lr'],
                              holdout_mae=history, best_epoch=best[2], epochs_run=len(history) - 1,
                              mean_abs_change=float((best[1][:, trainable.to(device)] - start[:, trainable.to(device)]).abs().mean()), seconds=time.time() - began)
        return best[1]

    def refit(self, start, device):
        """The table after the event: the rows of the new sensors refitted from the table of the arrival."""
        if start not in self._post:
            K, U = self.split(device)
            windows = [(raw, first + i) for raw, first, count in self.event_pieces() for i in range(count)]
            assert len(windows) == EVENT
            trainable = torch.zeros(self.n, dtype=torch.bool)
            trainable[U.cpu()] = True
            full = self.fit_windows(device, self.cold_table(H, start, device), trainable, windows, 'event:%s' % start)
            assert torch.equal(full[:, K], self.mature(device)) and bool(torch.isfinite(full).all())
            self._post[start] = full.contiguous()
        return self._post[start]

    # ---- forecasts ----
    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if not (isinstance(variant, str) and variant.startswith('p:')):
            return super()._forward(key, x_raw, device, sub)
        table = self.refit(variant[2:], device)                                  # the fit happens when the first forecast after the event is asked for
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model = self.model(ckpt, device)
        out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))

    def base(self, key, raw, first, count, device):
        """A scheduled arm: the table of the arrival before stream window EVENT, the refitted table from it on."""
        variant = key[3]
        if not (isinstance(variant, str) and variant.startswith('s:')):
            return super().base(key, raw, first, count, device)
        before, after = key[:3] + ('c%d:%s' % (H, variant[2:]),), key[:3] + ('p:' + variant[2:],)
        if raw is self.z['raw_train']:
            return super().base(before, raw, first, count, device)               # as the cold arm: the network without the arrivals before the arrival
        if raw is not self.z['raw_val'] or first >= EVENT:
            return super().base(after, raw, first, count, device)
        if first + count <= EVENT:
            return super().base(before, raw, first, count, device)
        return torch.cat([super().base(before, raw, first, EVENT - first, device), super().base(after, raw, EVENT, first + count - EVENT, device)])


def arms_of(B, seed):
    B.seed = seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arm = lambda name, ckpt, variant, **kw: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant, **kw)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit')]
    n_new = int((~B.known(B.old, device)).sum())
    B.info.update(sensors=B.n, new_sensors=n_new, cold_arms=bool(n_new >= c17.MIN_NEW and n_new < B.n), history=H, event=EVENT, released=RELEASED, tail=TAIL,
                  select_windows=int(e3.SELECT_WINDOWS))
    if B.info['cold_arms']:
        arms += [arm('konly', B.old, 'konly'), arm('c%d_same' % H, B.year, 'c%d:same' % H, history=H)]
        arms += [arm('c%d_%s' % (H, s), B.old, 'c%d:%s' % (H, s), history=H) for s in STARTS]
        arms += [arm('s_%s_%s' % (s, c), B.old, 's:' + s, history=H, restart=bool(c == 'r')) for s in STARTS for c in ('k', 'r')]
    B.arm_names = [a['name'] for a in arms]
    return arms


def make_chunks(B, weights, adj, device, arms, pieces):
    """e17_coldstart.make_chunks (a new sensor is not valid before its arrival). For the final run of the protocol, which is the
    only call with the complete list of arms, the masked targets are counted in the fit piece and the restart is attached to
    the stream's calibrator."""
    T, fn = c17.make_chunks(B, weights, adj, device, arms, pieces)
    if [a['name'] for a in arms] != B.arm_names or not B.info['cold_arms']:
        return T, fn
    n, new = B.n, torch.nonzero(~B.known(B.old, device))[:, 0]
    if len(pieces) == 1 and pieces[0][0] is B.z['raw_train']:                    # the calibrator's fit piece
        first, counts = pieces[0][1], B.info.setdefault('mask_counts', {})
        for a in arms:
            counts[a['name']] = dict(new_before_arrival=0, new_after_arrival=0, unmasked_new_before_arrival=0)

        def counted(s):
            out = fn(s)
            valid, valid0 = out[3], g.window_valid(out[2])                       # with the arrival mask / as the protocol would have it without
            before = (first + torch.arange(s.start, s.stop, device=device)) < B.arrival(H)
            for i, a in enumerate(arms):
                cols, c = i * n + new, counts[a['name']]
                c['new_before_arrival'] += int(valid[before][:, cols].sum())
                c['new_after_arrival'] += int(valid[~before][:, cols].sum())
                c['unmasked_new_before_arrival'] += int(valid0[before][:, cols].sum())
            return out
        return T, counted
    if len(pieces) == 2:                                                         # the stream over the validation and the test portion
        model = EventRLS.last
        assert model is not None and model.theta.shape[0] == n * len(arms) and model.event is None and model.predicts == 0 and model.updates == 0

        def event(m):
            assert m.updates == RELEASED, ('the event must follow the absorption of exactly the released windows', m.updates, RELEASED)
            for i, a in enumerate(arms):
                if not a.get('restart'):
                    continue
                rows = i * n + new
                sub = PlainRLS(len(new), g.Y_LEN, m.scale, m.ridge[rows], m.forget[rows], device)
                _, post = e5.make_chunks(B, weights, adj, device, [dict(a, variant='p:' + a['variant'][2:])], B.event_pieces())
                used = 0
                for s in range(0, EVENT, 256):
                    x, b, y, valid, _ = post(slice(s, min(s + 256, EVENT)))
                    sub.fit(x[:, new], b[:, new], y[:, new], valid[:, new])
                    used += int(valid[:, new].sum())
                m.theta[rows], m.P[rows], m.count[rows] = sub.theta, sub.P, sub.count
                B.info['restart:' + a['name']] = dict(calibrators=int(len(new)), windows=EVENT, absorbed_before=int(m.updates), forecast_calls_before=int(m.predicts),
                                                      valid_node_windows=used)
        model.event = event
    return T, fn


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


g.NodeRLS = EventRLS
e3.Bundle = HandoverBundle
e3.arms_of = arms_of
e3.make_chunks = make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e16.set_budget(e16.FULL)                                             # the calibrator and its selection use the standard windows
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    if B.info['cold_arms'] and '--max-windows' not in sys.argv:
        assert EVENT >= e3.SELECT_WINDOWS, 'the selection windows must end before the event'
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
