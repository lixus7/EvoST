"""Pre-specified summary of g42 (PRESPEC.md): the identity refit against EAC's node prompt, and a model trained anew. Read-only.

  python analyze_g42.py DELIVERED_ROOT --g37 G37_ROOT [--out summary.json]

DELIVERED_ROOT/results_V holds eac-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/{e3_metrics.jsonl, e10_info.json, ...}
(e23_eac.py). --g37: g37's delivered root (checks of the shared arms; fit_all / fit_shared / fit_head for comparison).
Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95% CI by the g29 district bootstrap.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A
import analyze_g33 as G33

SIX = ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
PRIMARY = [(d, y) for d in SIX for y in (2020, 2025)]
ARMS = ('same', 'stale5o', 'fit_emb', 'fit_emb_sel', 'fit_prompt', 'fit_emb_prompt', 'fit_scratch')
PAIRS = collections.OrderedDict([
    ('fit_emb_sel / fit_prompt', ('fit_emb_sel', 'fit_prompt')), ('fit_emb_prompt / fit_emb_sel', ('fit_emb_prompt', 'fit_emb_sel')),
    ('fit_emb_sel / fit_scratch', ('fit_emb_sel', 'fit_scratch')), ('fit_prompt / stale5o', ('fit_prompt', 'stale5o')),
    ('fit_emb_sel / stale5o', ('fit_emb_sel', 'stale5o')), ('fit_scratch / stale5o', ('fit_scratch', 'stale5o'))] +
    [('%s / same' % a, (a, 'same')) for a in ('stale5o', 'fit_emb', 'fit_emb_sel', 'fit_prompt', 'fit_emb_prompt', 'fit_scratch')])


def load(root):
    data = collections.defaultdict(dict)
    for d in sorted((Path(root) / 'results_V').glob('eac-*')):
        for k, v in G33.load(d).items():
            for seed, rows in v.items():
                assert seed not in data[k], ('two runs of one dataset-year and seed', k, seed)
                data[k][seed] = rows
    return data


def complete(data, arms, **kw):
    ok, missing = [], []
    for k in PRIMARY:
        if k in data and sorted(data[k]) == list(A.SEEDS) and all(A.value(data[k], a, 'calibrated', **kw) is not None for a in arms):
            ok.append(k)
        else:
            missing.append('%s/%d' % k)
    return ok, missing


def view(data, label, pairs=PAIRS, arms=ARMS, kind='calibrated', **kw):
    ok, missing = complete(data, arms, **kw)
    print('\n### %s: %d of %d dataset-years%s' % (label, len(ok), len(PRIMARY), ('; missing ' + ', '.join(missing)) if missing else ''))
    if not ok:
        return dict(n=0, missing=missing)
    items = []
    for k in ok:
        v = lambda arm: A.value(data[k], arm, kind, **kw)
        items.append((k, {name: (v(a) / v(b) if v(a) is not None and v(b) is not None else None) for name, (a, b) in pairs.items()}))
    table = {}
    for name in pairs:
        s = table[name] = A.geometric(items, name)
        print('- %s: **%.3f** (95%% CI %.3f-%.3f; range %.3f-%.3f; below 1 in %d of %d)' % (name, s['gm'], s['lo'], s['hi'], s['min'], s['max'],
                                                                                         sum(1 for _, r in items if r[name] is not None and r[name] < 1), s['n']))
    if 'stale5o / same' in table:
        for arm in ('fit_emb_sel', 'fit_prompt', 'fit_emb_prompt', 'fit_scratch'):
            st, fi = table['stale5o / same']['gm'], table['%s / same' % arm]['gm']
            table['recovery_' + arm] = (st - fi) / (st - 1.) if st > 1 else None
        print('- share of the five-year gap recovered: ' + ', '.join('%s %s' % (a, '%.0f%%' % (100 * table['recovery_' + a]) if table['recovery_' + a] is not None else '-')
                                                                   for a in ('fit_emb_sel', 'fit_prompt', 'fit_emb_prompt', 'fit_scratch')))
    return dict(n=len(ok), missing=missing, **table)


def agreement(label, mine, ref, arms, tolerance):
    worst, n, where = 0., 0, None
    for k, runs in mine.items():
        for seed, rows in runs.items():
            for (arm, portion, cohort), row in rows.items():
                other = ref.get(k, {}).get(seed, {}).get((arm, portion, cohort)) if arm in arms else None
                if other is None:
                    continue
                for kind in ('calibrated', 'uncalibrated'):
                    d = abs(row[kind]['H12_MAE'] - other[kind]['H12_MAE']) / other[kind]['H12_MAE']
                    n += 1
                    if d > worst:
                        worst, where = d, '%s/%d seed %d %s/%s/%s %s' % (k[0], k[1], seed, arm, portion, cohort, kind)
    ok = bool(n and worst <= tolerance)
    print('- check, %s: max relative difference %.2e over %d comparisons%s (tolerance %.0e: %s)' % (
        label, worst, n, ' (%s)' % where if where else '', tolerance, 'OK' if ok else ('no comparison' if not n else 'DIFFERS')))
    return dict(worst=worst, comparisons=n, where=where, ok=ok)


def fits(root):
    """Selected rates, epochs, trained values and time per arm (primary runs)."""
    rows = collections.defaultdict(list)
    for f in sorted((Path(root) / 'results_V').glob('eac-*/*/seed*/e10_info.json')):
        info = json.loads(f.read_text())
        for arm in ('fit_emb_sel', 'fit_prompt', 'fit_emb_prompt', 'fit_scratch'):
            if arm in info:
                r = info[arm]
                rows[arm].append((r['selected_lr'], r['epochs_run'], r['best_epoch'], r['trained_values'], r['seconds'], info.get('sensors')))
    print('\n| arm | runs | selected rates (count) | epochs run (median, max) | trained values (median) | seconds of fitting, three rates (median, max) |\n|---|---:|---|---|---:|---|')
    out = {}
    for arm, r in rows.items():
        rates = collections.Counter('%g' % x[0] for x in r)
        e = np.array([x[1] for x in r]); t = np.array([x[4] for x in r]); v = np.array([x[3] for x in r])
        out[arm] = dict(runs=len(r), rates=dict(rates), epochs_median=float(np.median(e)), epochs_max=int(e.max()), trained_values_median=float(np.median(v)),
                        seconds_median=float(np.median(t)), seconds_max=float(t.max()))
        print('| %s | %d | %s | %d, %d | %d | %.0f, %.0f |' % (arm, len(r), ', '.join('%s: %d' % kv for kv in sorted(rates.items())), np.median(e), e.max(), np.median(v),
                                                            np.median(t), t.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g37')
    ap.add_argument('--out')
    a = ap.parse_args()
    result = {}
    data = load(a.root)
    print('# g42: refreshing the identity against EAC\'s node prompt, and a model trained anew (PRESPEC.md)')
    print('Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95% CI by the g29 bootstrap over districts.')
    g37 = None
    if a.g37:
        g37 = collections.defaultdict(dict)
        for d in sorted((Path(a.g37) / 'results_V').glob('mnt-*')):
            for k, v in G33.load(d).items():
                g37[k].update(v)
        result['check_g37_exact'] = agreement("same / stale5o against g37's", data, g37, ('same', 'stale5o'), 1e-5)
        result['check_g37_fit'] = agreement("fit_emb / fit_emb_sel against g37's", data, g37, ('fit_emb', 'fit_emb_sel'), 2e-3)
    cal = result['test'] = view(data, 'Primary, test portion, with the calibrator, all sensors')
    result['test_new'] = view(data, 'Primary, sensors the old model does not know, test portion, with the calibrator', cohort='new')
    result['test_known'] = view(data, 'Primary, sensors the old model knows, test portion, with the calibrator', cohort='old')
    result['test_uncalibrated'] = view(data, 'Primary, test portion, without the calibrator', kind='uncalibrated')
    result['validation'] = view(data, 'Primary, validation portion, with the calibrator', portion='val')
    if g37:
        merged = {k: {s: {**rows, **{key: r for key, r in g37.get(k, {}).get(s, {}).items() if key[0] in ('fit_all', 'fit_shared', 'fit_head')}}
                      for s, rows in v.items()} for k, v in data.items()}
        pairs = collections.OrderedDict([('fit_prompt / fit_all', ('fit_prompt', 'fit_all')), ('fit_emb_sel / fit_all', ('fit_emb_sel', 'fit_all')),
                                         ('fit_prompt / fit_shared', ('fit_prompt', 'fit_shared')), ('fit_prompt / fit_head', ('fit_prompt', 'fit_head'))])
        print('\n(descriptive; g37\'s fit_all / fit_shared / fit_head are from its delivered runs of the same dataset-years, seeds and fit windows)')
        result['with_g37'] = view(merged, 'Primary, test portion, with the calibrator, against g37\'s arms', pairs=pairs,
                                  arms=('fit_prompt', 'fit_emb_sel', 'fit_all', 'fit_shared', 'fit_head', 'same', 'stale5o', 'fit_emb_prompt', 'fit_scratch'))
    result['fits'] = fits(a.root)
    print('\n### Rules fixed in PRESPEC.md (primary set, test portion, with the calibrator, all sensors)')
    if cal.get('n') != len(PRIMARY):
        print('- not evaluated: incomplete (%s)' % ', '.join(cal.get('missing', [])))
    else:
        f = lambda n: '%.3f, CI %.3f-%.3f' % (cal[n]['gm'], cal[n]['lo'], cal[n]['hi'])
        r = dict(E1=cal['fit_emb_sel / fit_prompt']['hi'] <= 1.01, E2=cal['fit_emb_sel / fit_prompt']['hi'] < 1,
                 E3=cal['fit_emb_prompt / fit_emb_sel']['gm'] >= .99, S1=cal['fit_emb_sel / fit_scratch']['hi'] < 1)
        v = lambda k: 'holds' if r[k] else 'does not hold'
        print('- **E1**, refreshing the existing identity is at least as good as EAC\'s prompt (fit_emb_sel / fit_prompt, CI at most 1.01): **%s** (%s)' % (v('E1'), f('fit_emb_sel / fit_prompt')))
        print('- **E2**, … and better (CI below 1): **%s**' % v('E2'))
        print('- **E3**, a prompt on top of the refreshed identity adds less than 1%% (fit_emb_prompt / fit_emb_sel, G at least 0.99): **%s** (%s)' % (v('E3'), f('fit_emb_prompt / fit_emb_sel')))
        print('- **S1**, reusing the old model beats training a new one on the same windows (fit_emb_sel / fit_scratch, CI below 1): **%s** (%s)' % (v('S1'), f('fit_emb_sel / fit_scratch')))
        reading = ('refreshing the per-node parameters the model already has is better than adding EAC\'s prompt' if r['E1'] and r['E2'] else
                   ('as good as EAC\'s prompt, with no new parameters' if r['E1'] else 'EAC\'s prompt is better by more than 1%'))
        print('- reading fixed in PRESPEC.md: **%s**' % reading)
        result['rules'] = dict({k: bool(x) for k, x in r.items()}, reading=reading)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
