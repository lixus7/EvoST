"""Checks of the online embedding updates of e12_online.py that must pass before any run is read. Fails closed.

  python selftest_e12.py --bundle STAE_BUNDLE.npz --work EMPTY_DIR [--seed 51] [--device cuda:0]

1. frozen: with learning rate 0 an online table issues the forecasts of the table it starts from.
2. learning: with the learning rate of the runs the forecasts change after the first release, not before.
3. causality: replacing every raw value from validation row C on leaves every forecast issued before that row is
   observed unchanged, and changes later ones. C is placed so that a release that came up to 12 windows too early
   (labels not yet matured) would change forecasts that must stay unchanged.
4. prefix: a stream that is first asked for its head (as the selection of the calibrator does) and then in full issues
   the forecasts of a stream asked in full at once, bit for bit; when the head ends inside a block (bounded smokes only),
   the forecasts of that one block agree to rounding. A split may begin inside a block.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import e12_online as e12
import e5_stae as e5
import g28_common as g

CADENCE, DELAY = e12.CADENCE, e12.DELAY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--seed', type=int, default=51)
    ap.add_argument('--device', default='auto')
    opt = ap.parse_args()
    work = Path(opt.work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else ('cpu' if opt.device == 'auto' else opt.device))
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    B = e12.OnlineBundle(opt.bundle)
    B.seed = opt.seed
    val, test = B.z['raw_val'], B.z['raw_test']
    arm = dict(name='on_stale', base='stae', ckpt=B.old, norm=B.old, age=5, variant=e12.LIVE + 'full')
    start = e5.StaeBundle.table(B, B.old, device)
    lr = e12.ONLINE['lr']
    assert lr > 0
    P = 3 * CADENCE + 8                                                # three complete blocks and the head of a fourth

    def issued(pieces, requests, rate=lr):
        e12.ONLINE['lr'] = rate
        try:
            live = e12.OnlineTable(B, B.old, start, arm['variant'], device)
            with torch.no_grad():
                out = [live.serve(pieces, lo, hi) for lo, hi in requests][-1]
        finally:
            e12.ONLINE['lr'] = lr
        return out.cpu(), live

    report = dict(bundle=Path(opt.bundle).name, seed=opt.seed, device=str(device), positions=P, learning_rate=lr)
    whole = [(val, 0, P)]
    # 1. frozen
    frozen, _ = issued(whole, [(0, P)], 0.)
    with torch.no_grad():
        static = B.forward(('stae', B.old, B.old, 'full'), np.array(g.get_windows(val, 0, P)[0], dtype=np.float32), device).cpu()
    scale = float(static.abs().max())
    report['frozen_max_abs_difference'] = float((frozen - static).abs().max())
    assert report['frozen_max_abs_difference'] <= 1e-4 * (1 + scale), ('a table that does not learn issues other forecasts than its start', report)
    # 2. learning
    learned, live = issued(whole, [(0, P)])
    assert torch.equal(learned[:CADENCE], frozen[:CADENCE]), 'forecasts changed before the first release'
    report['learning_max_abs_change_after_first_release'] = float((learned[CADENCE:] - frozen[CADENCE:]).abs().max())
    assert report['learning_max_abs_change_after_first_release'] > 1e-6 * (1 + scale), 'the table did not learn'
    assert [t['block'] for t in live.trace] == [1, 2, 3] and live.absorbed == 3 * CADENCE - DELAY
    assert live.trace[0]['windows'] == len(range(0, CADENCE - DELAY + 1, live.stride))
    report['releases'] = live.trace
    # 3. causality
    C = CADENCE + 16                                                   # row 80: observed by the forecasts of positions >= 69
    assert CADENCE + g.X_LEN + g.Y_LEN - 1 >= C > CADENCE + g.X_LEN - 1, 'C must be reachable by a release that is 12 windows early'
    changed = np.array(val, dtype=np.float32, copy=True)
    changed[C:] = changed[C:] * 1.5 + 10.
    other, _ = issued([(changed, 0, P)], [(0, P)])
    first_seen = C - g.X_LEN + 1                                       # first position whose input window contains row C
    assert first_seen > CADENCE
    assert torch.equal(other[:first_seen], learned[:first_seen]), 'a forecast depends on a value that was not yet observed'
    report['causality_positions_unchanged'] = int(first_seen)
    report['causality_max_abs_change_later'] = float((other[first_seen:] - learned[first_seen:]).abs().max())
    assert report['causality_max_abs_change_later'] > 0
    # 4. prefix
    full = [(val, 0, P), (test, 0, CADENCE)]
    at_once, _ = issued(full, [(0, P + CADENCE)])
    for head in (2 * CADENCE, CADENCE + 36):                           # the head ends on a boundary / inside a block
        staged, _ = issued([(val, 0, head)], [(0, head)])
        e12.ONLINE['lr'] = lr
        live2 = e12.OnlineTable(B, B.old, start, arm['variant'], device)
        with torch.no_grad():
            live2.serve([(val, 0, head)], 0, head)
            parts = [live2.serve(full, lo, min(lo + 50, P + CADENCE)).cpu() for lo in range(0, P + CADENCE, 50)]
        again = torch.cat(parts)
        assert torch.equal(again[:head], staged), 'the head of the stream was issued differently'
        # A head that ends inside a block splits the forward pass of that block in two: those forecasts agree to
        # rounding, every other one bit for bit (the table is the same; only the batch of the forward pass differs).
        split = torch.zeros(len(again), dtype=torch.bool)
        if head % CADENCE:
            split[head // CADENCE * CADENCE:(head // CADENCE + 1) * CADENCE] = True
        worst = float((again - at_once).abs().max())
        report['prefix_%d_max_abs_difference' % head] = worst
        assert torch.equal(again[~split], at_once[~split]), ('a stream asked in stages differs from the stream asked at once', head, worst)
        assert worst <= 1e-5 * (1 + scale), ('the block that was issued in two parts differs by more than rounding', head, worst)
    report['status'] = 'passed'
    (work / 'selftest_e12.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='selftest_e12_passed', frozen=report['frozen_max_abs_difference'],
                          learned=report['learning_max_abs_change_after_first_release'], unchanged_positions=report['causality_positions_unchanged'])), flush=True)


if __name__ == '__main__':
    main()
