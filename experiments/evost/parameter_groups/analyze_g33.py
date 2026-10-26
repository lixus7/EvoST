"""Pre-specified summary of g33 (PRESPEC.md): identity refresh of a five-year-old frozen STAEFormer (read-only).

  python analyze_g33.py RESULT_ROOT [--g30 ROOT ...] [--c2 ROOT ...] [--out summary.json]

RESULT_ROOT holds .../<dataset>_<year>/seed<seed>/e3_metrics.jsonl. Endpoint: test-portion H12 MAE with the calibrator,
mean of seeds 51-53; ratios are summarised by geometric means with the g29 bootstrap over districts.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A

PRIMARY = [(d, y) for d in ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12') for y in (2020, 2025)]
SECONDARY = [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2020)]
DEVELOPMENT = [('TFNSW', 2025)]
ARMS = ('same_mean', 'same_prof', 'same_geo', 'same_rand', 'stale5o', 'stale5o_mean', 'don_prof', 'don_geo', 'don_rand', 'new_prof', 'new_geo', 'fit_emb')
PAIRS = collections.OrderedDict([
    ('fit_emb / stale5o', ('fit_emb', 'stale5o')), ('new_prof / stale5o', ('new_prof', 'stale5o')), ('new_geo / stale5o', ('new_geo', 'stale5o')),
    ('don_prof / stale5o', ('don_prof', 'stale5o')), ('new_prof / new_geo', ('new_prof', 'new_geo')), ('don_prof / don_geo', ('don_prof', 'don_geo')),
    ('don_prof / don_rand', ('don_prof', 'don_rand')), ('don_prof / stale5o_mean', ('don_prof', 'stale5o_mean')),
    ('same_prof / same_geo', ('same_prof', 'same_geo')), ('same_prof / same_mean', ('same_prof', 'same_mean')), ('same_geo / same_rand', ('same_geo', 'same_rand'))])


def load(root):
    data = collections.defaultdict(dict)
    for f in sorted(Path(root).rglob('e3_metrics.jsonl')):
        if not (f.parent / 'completed.json').exists() or json.loads((f.parent / 'completed.json').read_text()).get('bounded_smoke'):
            continue
        rows = [json.loads(line) for line in f.open()]
        key, seed = (rows[0]['dataset'], int(rows[0]['year'])), int(rows[0]['seed'])
        assert seed not in data[key], ('two runs of one dataset-year and seed', key, seed)
        data[key][seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
    return data


def items_of(data, keys, kind='calibrated', cohort='all'):
    items, missing = [], []
    for k in keys:
        if k not in data or sorted(data[k]) != list(A.SEEDS):
            missing.append('%s/%d' % k); continue
        v = lambda arm: A.value(data[k], arm, kind, cohort=cohort)
        same = v('same')
        if same is None:
            missing.append('%s/%d' % k); continue
        row = {arm + ' / same': (None if v(arm) is None else v(arm) / same) for arm in ARMS}
        for name, (a, b) in PAIRS.items():
            row[name] = None if None in (v(a), v(b)) else v(a) / v(b)
        items.append((k, row))
    return items, missing


def geo(items, name):
    part = [(k, r) for k, r in items if r.get(name) is not None]
    return A.geometric(part, name) if part else None


def show(label, items, missing, expected):
    print('\n### %s: %d of %d dataset-years%s' % (label, len(items), expected, ('; missing ' + ', '.join(missing)) if missing else ''))
    out = {}
    for name in [a + ' / same' for a in ARMS] + list(PAIRS):
        s = geo(items, name)
        if s:
            out[name] = s
            print('- %s: **%.3f** (95%% CI %.3f-%.3f; range %.3f-%.3f; %d dataset-years, %d districts)' % (name, s['gm'], s['lo'], s['hi'], s['min'], s['max'], s['n'], s['districts']))
    return out


def recovery(summary, arm):
    gap, got = summary['stale5o / same']['gm'], summary[arm + ' / same']['gm']
    return (gap - got) / (gap - 1.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g30', nargs='*', default=[])
    ap.add_argument('--c2', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    data = load(a.root)
    print('# g33: identity refresh of a five-year-old frozen STAEFormer')
    print('Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53. Runs found: %d dataset-years, %d with all three seeds.' % (
        len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    for label, roots, arms in (('g30', a.g30, ('same', 'stale5o', 'nobb_zero', 'nobb_last')), ('g31 C2', a.c2, ('same', 'same_mean', 'stale5o', 'stale5o_mean'))):
        if roots:
            ref, worst, n = A.load(roots), 0., 0
            for k, runs in data.items():
                for seed, rows in runs.items():
                    for arm in arms:
                        if k in ref and seed in ref[k] and (arm, 'test', 'all') in ref[k][seed]:
                            for kind in ('calibrated', 'uncalibrated'):
                                x, y = rows[(arm, 'test', 'all')][kind]['H12_MAE'], ref[k][seed][(arm, 'test', 'all')][kind]['H12_MAE']
                                worst, n = max(worst, abs(x - y) / y), n + 1
            print('Shared arms against %s: max relative difference %.2e over %d comparisons (%s)' % (label, worst, n, 'OK' if worst <= 1e-5 else 'DIFFERS: do not read the numbers below'))
    result = {}
    prim, miss = items_of(data, PRIMARY)
    result['primary'] = show('Primary (six districts x 2020, 2025), with the calibrator', prim, miss, len(PRIMARY))
    raw, _ = items_of(data, PRIMARY, 'uncalibrated')
    result['primary_uncalibrated'] = show('Primary, without the calibrator', raw, miss, len(PRIMARY))
    for cohort, text in (('old', 'sensors the old checkpoint knows'), ('new', 'sensors the old checkpoint does not know')):
        part, m2 = items_of(data, PRIMARY, 'calibrated', cohort)
        result['primary_' + cohort] = show('Primary, %s, with the calibrator' % text, part, m2, len(PRIMARY))
    sec, miss_s = items_of(data, SECONDARY)
    result['secondary'] = show('Secondary (PEMS05 2020 / 2025, TFNSW 2020), with the calibrator', sec, miss_s, len(SECONDARY))
    dev, miss_d = items_of(data, DEVELOPMENT)
    result['development'] = show('Development (TFNSW 2025; never part of a decision), with the calibrator', dev, miss_d, len(DEVELOPMENT))
    if prim:
        print('\n| dataset/year | same | stale5o | stale5o_mean | don_prof | new_prof | new_geo | fit_emb | fit_emb / same |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for (d, y), r in prim + sec + dev:
            v = lambda arm: A.value(data[(d, y)], arm, 'calibrated')
            cell = lambda arm: '-' if v(arm) is None else '%.2f' % v(arm)
            print('| %s/%d | %.2f | %s | %s | %s | %s | %s | %s | %.3f |' % (d, y, v('same'), cell('stale5o'), cell('stale5o_mean'), cell('don_prof'), cell('new_prof'), cell('new_geo'),
                                                                          cell('fit_emb'), r['fit_emb / same']))
    print('\n### Rules fixed in PRESPEC.md (primary set)')
    if len(prim) != len(PRIMARY):
        print('- not evaluated: the primary set is incomplete')
    else:
        p, new = result['primary'], result['primary_new']
        rho_f, rho_n = recovery(p, 'fit_emb'), recovery(p, 'new_prof')
        f1 = p['fit_emb / stale5o']['hi'] < 1 and rho_f >= .5
        f2 = p['fit_emb / same']['hi'] <= A.MARGIN
        n1 = p['new_prof / stale5o']['hi'] < 1 and rho_n >= 1 / 3 and p['new_prof / same']['gm'] < p['stale5o_mean / same']['gm']
        n2 = new['new_prof / stale5o']['hi'] < 1
        b1 = p['same_prof / same_geo']['hi'] < 1 and p['don_prof / don_geo']['hi'] < 1
        print('- F1, refitting only the identities repairs most of the damage (fit_emb / stale5o CI below 1, recovery at least 1/2): **%s** (ratio %.3f, CI %.3f-%.3f; recovery %.0f%%)' % (
            'holds' if f1 else 'does not hold', p['fit_emb / stale5o']['gm'], p['fit_emb / stale5o']['lo'], p['fit_emb / stale5o']['hi'], 100 * rho_f))
        print('- F2, ... to within the g29 margin of the same-year model (fit_emb / same CI at most %.2f): **%s** (%.3f, CI %.3f-%.3f)' % (
            A.MARGIN, 'holds' if f2 else 'does not hold', p['fit_emb / same']['gm'], p['fit_emb / same']['lo'], p['fit_emb / same']['hi']))
        print('- N1, training-free refresh works (new_prof / stale5o CI below 1, recovery at least 1/3, better than the mean embedding): **%s** (ratio %.3f, CI %.3f-%.3f; recovery %.0f%%)' % (
            'holds' if n1 else 'does not hold', p['new_prof / stale5o']['gm'], p['new_prof / stale5o']['lo'], p['new_prof / stale5o']['hi'], 100 * rho_n))
        print('- N2, on sensors the old checkpoint does not know, behaviour-matched embeddings beat the mean embedding: **%s** (new_prof / stale5o on that cohort %.3f, CI %.3f-%.3f)' % (
            'holds' if n2 else 'does not hold', new['new_prof / stale5o']['gm'], new['new_prof / stale5o']['lo'], new['new_prof / stale5o']['hi']))
        print('- B1, identity follows behaviour rather than location (same_prof / same_geo and don_prof / don_geo, both CIs below 1): **%s** (%.3f, CI %.3f-%.3f; %.3f, CI %.3f-%.3f)' % (
            'holds' if b1 else 'does not hold', p['same_prof / same_geo']['gm'], p['same_prof / same_geo']['lo'], p['same_prof / same_geo']['hi'],
            p['don_prof / don_geo']['gm'], p['don_prof / don_geo']['lo'], p['don_prof / don_geo']['hi']))
        result['rules'] = dict(F1=bool(f1), F2=bool(f2), N1=bool(n1), N2=bool(n2), B1=bool(b1), recovery_fit_emb=rho_f, recovery_new_prof=rho_n)
    fits = []
    for f in sorted(Path(a.root).rglob('e10_info.json')):
        info = json.loads(f.read_text())
        if 'fit' in info:
            h = info['fit']['holdout_mae']
            fits.append((h[-1] if info['fit']['best_epoch'] == len(h) - 1 else min(h)) / h[0])
    if fits:
        print('\nEmbedding fit, holdout MAE inside the fit windows after / before: median %.2f (range %.2f-%.2f, %d runs)' % (np.median(fits), min(fits), max(fits), len(fits)))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
