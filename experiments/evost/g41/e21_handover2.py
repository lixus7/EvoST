"""E21 (g41 family hnd): two repairs of g40's hand-over, read on the same runs.

g40 (e18_handover, unchanged here): a sensor the old checkpoint does not know arrives 64 windows before the stream and gets the
mean row of its five behaviour-nearest mature sensors; at stream window EVENT (1,024) its row is refitted on the 1,024 most
recent windows with released labels and its calibrator is restarted on them (arm s_prof0_r). Two things were left open:
the mature sensors stay 0.8% worse than without the arrivals (g40's H2 failed), and the restarted calibrators are fitted on
the windows the new rows were just fitted on, i.e. on in-sample forecasts.

  a_prof0_r   at the event every row is refitted - mature and new, shared weights frozen - from the arrival table, on the same
              1,024 windows, with the same fit; every sensor's calibrator is restarted on those windows (exact ridge, the arm's
              ridge and forgetting factor unchanged).
  x_prof0_r   the table of s_prof0_r (new rows refitted, every forecast of the stream identical to s_prof0_r's); the new
              sensors' calibrators are restarted on out-of-fold forecasts: the 1,024 event windows are cut in two halves, the
              new rows are refitted on each half (from the arrival table; the windows of a half that could share a row with the
              other half left out: a gap of 24 windows at the cut), and each event window gets the forecast of the table fitted
              on the other half.

Arms: same, stale5o, fit_emb, konly, c64_prof0, s_prof0_r (= g40's), a_prof0_r, x_prof0_r. The driver, the protocol and the
event machinery are g40's (e18_handover: the calibrator subclass that counts its calls, the restart inside the forecast of
stream window EVENT after exactly EVENT - 11 absorbed windows); this file only adds tables, routes and the two restart kinds.
Rules A1, A2 and X2 in ../PRESPEC.md.
"""
import json
from pathlib import Path
import sys
import torch
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import e16_budget as e16
import e17_coldstart as c17
import e18_handover as e18
import g28_common as g

GAP = g.SPAN                                                             # 24: windows i and j share a row when |i - j| < 24
HALF = e18.EVENT // 2
H = e18.H


class Handover2Bundle(e18.HandoverBundle):
    def __init__(self, path):
        super().__init__(path)
        self._post_all, self._fold = {}, {}

    def event_windows(self):
        windows = [(raw, first + i) for raw, first, count in self.event_pieces() for i in range(count)]
        assert len(windows) == e18.EVENT
        return windows

    def event_index(self, raw, first):
        """Position of window `first` of `raw` among the EVENT event windows (the train tail first, then the stream)."""
        if raw is self.z['raw_train']:
            return first - (g.n_windows(raw) - e18.TAIL)
        assert raw is self.z['raw_val'], 'event windows lie in the train tail and the validation portion only'
        return e18.TAIL + first

    def refit_all(self, start, device):
        """a_: every row refitted at the event, from the arrival table, on the event windows."""
        if start not in self._post_all:
            full = self.fit_windows(device, self.cold_table(H, start, device), torch.ones(self.n, dtype=torch.bool), self.event_windows(),
                                    'event_all:%s' % start)
            assert bool(torch.isfinite(full).all())
            self._post_all[start] = full.contiguous()
        return self._post_all[start]

    def fold(self, start, half, device):
        """x_: the new rows refitted on one half of the event windows (gap of GAP windows at the cut)."""
        key = (start, half)
        if key not in self._fold:
            K, U = self.split(device)
            windows = self.event_windows()
            span = windows[:HALF - GAP] if half == 'A' else windows[HALF + GAP:]
            trainable = torch.zeros(self.n, dtype=torch.bool)
            trainable[U.cpu()] = True
            t = self.fit_windows(device, self.cold_table(H, start, device), trainable, span, 'event_fold%s:%s' % (half, start))
            assert torch.equal(t[:, K], self.mature(device)) and bool(torch.isfinite(t).all())
            self._fold[key] = t.contiguous()
        return self._fold[key]

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if not (isinstance(variant, str) and (variant.startswith('pa:') or variant.startswith('ofA:') or variant.startswith('ofB:'))):
            return super()._forward(key, x_raw, device, sub)
        table = self.refit_all(variant[3:], device) if variant.startswith('pa:') else self.fold(variant[4:], variant[2], device)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model = self.model(ckpt, device)
        out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))

    def base(self, key, raw, first, count, device):
        variant = key[3]
        if not isinstance(variant, str):
            return super().base(key, raw, first, count, device)
        if variant.startswith('sx:'):                                            # x_: every forecast is s_'s
            return super().base(key[:3] + ('s:' + variant[3:],), raw, first, count, device)
        if variant.startswith('sa:'):                                            # a_: e18's schedule with the all-rows table after the event
            before, after = key[:3] + ('c%d:%s' % (H, variant[3:]),), key[:3] + ('pa:' + variant[3:],)
            if raw is self.z['raw_train']:
                return super().base(before, raw, first, count, device)
            if raw is not self.z['raw_val'] or first >= e18.EVENT:
                return super().base(after, raw, first, count, device)
            if first + count <= e18.EVENT:
                return super().base(before, raw, first, count, device)
            return torch.cat([super().base(before, raw, first, e18.EVENT - first, device), super().base(after, raw, e18.EVENT, first + count - e18.EVENT, device)])
        if variant.startswith('o:'):                                             # out-of-fold forecasts of the event windows (calibrator restart of x_)
            e0 = self.event_index(raw, first)
            assert 0 <= e0 and e0 + count <= e18.EVENT, (e0, count)
            parts = []
            if e0 < HALF:
                parts.append(super().base(key[:3] + ('ofB:' + variant[2:],), raw, first, min(e0 + count, HALF) - e0, device))
            if e0 + count > HALF:
                s = max(e0, HALF)
                parts.append(super().base(key[:3] + ('ofA:' + variant[2:],), raw, first + (s - e0), e0 + count - s, device))
            return torch.cat(parts)
        return super().base(key, raw, first, count, device)


RESTART = {'s:': ('new', 'p:'), 'sa:': ('all', 'pa:'), 'sx:': ('new', 'o:')}     # schedule prefix -> (calibrators restarted, forecasts they are fitted on)


def schedule(variant):
    for prefix, (rows, post) in RESTART.items():
        if variant.startswith(prefix):
            return rows, post + variant[len(prefix):]
    raise ValueError(variant)


def arms_of(B, seed):
    B.seed = seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arm = lambda name, ckpt, variant, **kw: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant, **kw)
    arms = [arm('same', B.year, 'full'), arm('stale5o', B.old, 'full'), arm('fit_emb', B.old, 'fit')]
    n_new = int((~B.known(B.old, device)).sum())
    B.info.update(sensors=B.n, new_sensors=n_new, cold_arms=bool(n_new >= c17.MIN_NEW and n_new < B.n), history=H, event=e18.EVENT, released=e18.RELEASED,
                  tail=e18.TAIL, select_windows=int(e3.SELECT_WINDOWS), half=HALF, gap=GAP)
    if B.info['cold_arms']:
        arms += [arm('konly', B.old, 'konly'), arm('c%d_prof0' % H, B.old, 'c%d:prof0' % H, history=H),
                 arm('s_prof0_r', B.old, 's:prof0', history=H, restart=True), arm('a_prof0_r', B.old, 'sa:prof0', history=H, restart=True),
                 arm('x_prof0_r', B.old, 'sx:prof0', history=H, restart=True)]
    B.arm_names = [a['name'] for a in arms]
    return arms


def make_chunks(B, weights, adj, device, arms, pieces):
    """e18_handover.make_chunks with the restart generalised: per arm, which calibrators are restarted (the new sensors' or
    every sensor's) and on which forecasts (the post-event table's, or out-of-fold)."""
    T, fn = c17.make_chunks(B, weights, adj, device, arms, pieces)
    if [a['name'] for a in arms] != B.arm_names or not B.info['cold_arms']:
        return T, fn
    n, new = B.n, torch.nonzero(~B.known(B.old, device))[:, 0]
    if len(pieces) == 1 and pieces[0][0] is B.z['raw_train']:                    # the calibrator's fit piece: count masked targets (as g40)
        first, counts = pieces[0][1], B.info.setdefault('mask_counts', {})
        for a in arms:
            counts[a['name']] = dict(new_before_arrival=0, new_after_arrival=0, unmasked_new_before_arrival=0)

        def counted(s):
            out = fn(s)
            valid, valid0 = out[3], g.window_valid(out[2])
            before = (first + torch.arange(s.start, s.stop, device=device)) < B.arrival(H)
            for i, a in enumerate(arms):
                cols, c = i * n + new, counts[a['name']]
                c['new_before_arrival'] += int(valid[before][:, cols].sum())
                c['new_after_arrival'] += int(valid[~before][:, cols].sum())
                c['unmasked_new_before_arrival'] += int(valid0[before][:, cols].sum())
            return out
        return T, counted
    if len(pieces) == 2:                                                         # the stream over the validation and the test portion
        model = e18.EventRLS.last
        assert model is not None and model.theta.shape[0] == n * len(arms) and model.event is None and model.predicts == 0 and model.updates == 0
        everyone = torch.arange(n, device=device)

        def event(m):
            assert m.updates == e18.RELEASED, ('the event must follow the absorption of exactly the released windows', m.updates, e18.RELEASED)
            for i, a in enumerate(arms):
                if not a.get('restart'):
                    continue
                which, post_variant = schedule(a['variant'])
                cal = new if which == 'new' else everyone
                rows = i * n + cal
                sub = e18.PlainRLS(len(cal), g.Y_LEN, m.scale, m.ridge[rows], m.forget[rows], device)
                _, post = e5.make_chunks(B, weights, adj, device, [dict(a, variant=post_variant)], B.event_pieces())
                used = 0
                for s in range(0, e18.EVENT, 256):
                    x, b, y, valid, _ = post(slice(s, min(s + 256, e18.EVENT)))
                    sub.fit(x[:, cal], b[:, cal], y[:, cal], valid[:, cal])
                    used += int(valid[:, cal].sum())
                m.theta[rows], m.P[rows], m.count[rows] = sub.theta, sub.P, sub.count
                B.info['restart:' + a['name']] = dict(calibrators=int(len(cal)), which=which, forecasts=post_variant, windows=e18.EVENT,
                                                      absorbed_before=int(m.updates), forecast_calls_before=int(m.predicts), valid_node_windows=used)
        model.event = event
    return T, fn


e3.Bundle = Handover2Bundle
e3.arms_of = arms_of
e3.make_chunks = make_chunks

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e16.set_budget(e16.FULL)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    B = e10.RefreshBundle.last
    if B.info['cold_arms'] and '--max-windows' not in sys.argv:
        assert e18.EVENT >= e3.SELECT_WINDOWS, 'the selection windows must end before the event'
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
