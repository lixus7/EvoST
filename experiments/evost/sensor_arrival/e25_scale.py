"""E25 (2026-10-25, Sydney): g41's hand-over (e21_handover2.py, unchanged) with the calibrators' output scale s taken from the
targets that the arrival masks allow.

Why. e3_confirm.run computes the shared output scale
    s = standard deviation of every finite non-zero target in the last min(2,048, n) training windows, all sensors,
before the arrival masks act (e17_coldstart.make_chunks removes a new sensor's windows from before its arrival only from the
calibrators' fitting records). s therefore includes targets that new sensors record before their arrival.

Rules (--scale-rule):
  full      e21_handover2.py as it is. The scale that e3_confirm.run computes is checked against the one computed here and
            recorded, never changed.
  arrival   the same statistic over the targets the masks allow: every established sensor's windows and each new sensor's
            windows from its arrival on (window index >= n_windows(train) - H, H = 64). This one value replaces s in every
            calibrator of every arm: the selection grid, the final stream and the restarts at stream window 1,024.

Everything else is e21_handover2.py: arms, fits, seeds, protocol and outputs. s enters a calibrator only through b/s, (y - b)/s
and b + s * theta'z, so the two rules differ only in the effective ridge penalty on the twelve rows of theta that multiply b/s.
Extra output: e25_scale.json. e3_selection.json keeps the statistic e3_confirm.run computed ('scale'); under 'arrival' every
calibrator used e25_scale.json's scale_used instead.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import e3_confirm as e3
import e10_refresh as e10
import e16_budget as e16
import e18_handover as e18
import e21_handover2 as e21
import g28_common as g

VERSION = 'e25-scale-2026-10-25a'
RULES = ('full', 'arrival')
H = e21.H
STATE = dict(constructed=0, incoming=set(), used=set())


def scales(path):
    """(all, arrival, counts): e3_confirm.run's statistic, and the same statistic with the arrival masks."""
    z, meta = g.load_bundle(path)
    raw = z['raw_train']
    nw = g.n_windows(raw)
    fit_n = min(e3.FIT_WINDOWS, nw)
    first = nw - fit_n
    _, y = g.get_windows(raw, first, fit_n)
    y = g.clean_y(y)
    valid = g.target_valid(y)
    s_all = float(y[valid].std())                                           # as e3_confirm.run
    year = int(meta['year'])
    new = torch.from_numpy(~np.asarray(z['known/y%d' % (year - 5)], bool))   # e10_refresh: the old checkpoint is five years older
    arrival = nw - H                                                        # e17_coldstart.ColdBundle.arrival(H)
    before = torch.arange(first, first + fit_n) < arrival
    allowed = valid.clone()
    allowed[before[:, None] & new[None, :]] = False
    s_arr = float(y[allowed].std())
    counts = dict(fit_first=int(first), fit_windows=int(fit_n), arrival=int(arrival), history=H, sensors=int(len(new)),
                  new_sensors=int(new.sum()), valid_targets=int(valid.sum()), allowed_targets=int(allowed.sum()))
    return s_all, s_arr, counts


def install(rule, s_all, s_arr):
    """Every calibrator is a g28_common.NodeRLS: e18 binds the class as PlainRLS, and its EventRLS (the stream calibrator)
    subclasses it. The constructor receives e3_confirm.run's statistic (selection grid, final stream) or the stream
    calibrator's own scale (restarts at the event); under 'arrival' both become s_arr."""
    plain_init = e18.PlainRLS.__init__
    target = s_arr if rule == 'arrival' else s_all

    def init(self, n, h, scale, ridge, forget, device):
        scale = float(scale)
        assert scale in (s_all, target), ('a calibrator received an unexpected scale', scale, s_all, target)
        STATE['constructed'] += 1
        STATE['incoming'].add(scale)
        STATE['used'].add(target)
        plain_init(self, n, h, target, ridge, forget, device)

    e18.PlainRLS.__init__ = init
    return target


def take_rule():
    """--scale-rule is ours; every other argument goes to e3_confirm.main unchanged."""
    argv = sys.argv[1:]
    assert argv.count('--scale-rule') == 1, 'give --scale-rule once: ' + ' | '.join(RULES)
    i = argv.index('--scale-rule')
    rule = argv[i + 1]
    assert rule in RULES, rule
    del argv[i:i + 2]
    sys.argv = sys.argv[:1] + argv
    return rule, argv[argv.index('--bundle') + 1], Path(argv[argv.index('--out') + 1]).resolve()


if __name__ == '__main__':
    rule, bundle, out = take_rule()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e16.set_budget(e16.FULL)
    s_all, s_arr, counts = scales(bundle)
    used = install(rule, s_all, s_arr)
    e3.main()                                                               # e21_handover2.py's main from here on
    B = e10.RefreshBundle.last
    if B.info['cold_arms'] and '--max-windows' not in sys.argv:
        assert e18.EVENT >= e3.SELECT_WINDOWS, 'the selection windows must end before the event'
    (out / 'e10_info.json').write_text(json.dumps(B.info, indent=2) + '\n')
    selection_scale = json.loads((out / 'e3_selection.json').read_text())['scale']
    assert selection_scale == s_all, ('e3_confirm.run computed another statistic', selection_scale, s_all)
    assert (counts['sensors'], counts['new_sensors']) == (B.n, B.info['new_sensors']), (counts, B.n, B.info['new_sensors'])
    assert STATE['used'] == {used} and STATE['constructed'] > 0, STATE
    g.save_json(out / 'e25_scale.json', dict(
        version=VERSION, rule=rule, scale_all=s_all, scale_arrival=s_arr, arrival_over_all=s_arr / s_all, scale_used=used,
        calibrators_constructed=STATE['constructed'], scales_received=sorted(STATE['incoming']), scales_used=sorted(STATE['used']),
        mask='g28_common.target_valid(clean_y(y)): finite and non-zero; arrival: new sensors (unknown to the checkpoint five years older) '
             'only in windows with index >= arrival',
        note="e3_selection.json 'scale' is e3_confirm.run's statistic (scale_all); every calibrator used scale_used", **counts))
