"""Pre-specified summary of g38 (PRESPEC.md): label budget of the identity refresh (family bud) and sensors that have just
arrived (family cold). Read-only.

  python analyze_g38.py DELIVERED_ROOT [--g33 ROOT ...] [--out summary.json]

DELIVERED_ROOT/results_V holds
  bud-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/b<budget>/e3_metrics.jsonl      (e16_budget.py)
  cold-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/e3_metrics.jsonl               (e17_coldstart.py)
Endpoint: H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap over districts.
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
SECONDARY = [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2020)]
DEVELOPMENT = [('TFNSW', 2025)]
BUDGETS, HISTORIES = (64, 256, 1024, 2048), (64, 256)
SCHEMES = ('mean', 'prof0', 'rand0', 'geo0', 'ind', 'pinit', 'gmean', 'rand', 'prof', 'geo')
EXACT, REFIT = 1e-5, 2e-3


def load_runs(files):
    data = collections.defaultdict(dict)
    for f in files:
        done = f.parent / 'completed.json'
        if not done.exists() or json.loads(done.read_text()).get('bounded_smoke'):
            continue
        rows = [json.loads(line) for line in f.open()]
        key, seed = (rows[0]['dataset'], int(rows[0]['year'])), int(rows[0]['seed'])
        assert seed not in data[key], ('two runs of one dataset-year and seed', key, seed)
        data[key][seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
    return data


def complete(data, keys, arms, **kw):
    ok, missing = [], []
    for k in keys:
        if k in data and sorted(data[k]) == list(A.SEEDS) and all(A.value(data[k], a, 'calibrated', **kw) is not None for a in arms):
            ok.append(k)
        else:
            missing.append('%s/%d' % k)
    return ok, missing


def ratio_items(data, keys, pairs, kind='calibrated', **kw):
    """[(key, {name: ratio})] for the dataset-years of `keys` (already complete)."""
    items = []
    for k in keys:
        v = lambda arm: A.value(data[k], arm, kind, **kw)
        items.append((k, {name: (v(a) / v(b) if v(a) is not None and v(b) is not None else None) for name, (a, b) in pairs.items()}))
    return items


def line(name, s, items=None):
    below = '' if items is None else '; below 1 in %d of %d' % (sum(1 for _, r in items if r[name] is not None and r[name] < 1), s['n'])
    print('- %s: **%.3f** (95%% CI %.3f-%.3f; range %.3f-%.3f%s)' % (name, s['gm'], s['lo'], s['hi'], s['min'], s['max'], below))


def agreement(label, mine, ref, arms, tolerance):
    worst, n = 0., 0
    for k, runs in mine.items():
        for seed, rows in runs.items():
            for arm in arms:
                if k in ref and seed in ref[k] and (arm, 'test', 'all') in ref[k][seed] and (arm, 'test', 'all') in rows:
                    for kind in ('calibrated', 'uncalibrated'):
                        x, y = rows[(arm, 'test', 'all')][kind]['H12_MAE'], ref[k][seed][(arm, 'test', 'all')][kind]['H12_MAE']
                        worst, n = max(worst, abs(x - y) / y), n + 1
    if n:
        print('Check, %s: max relative difference %.2e over %d comparisons (tolerance %.0e: %s)' % (
            label, worst, n, tolerance, 'OK' if worst <= tolerance else 'DIFFERS: do not read the numbers that depend on it'))
    return dict(worst=worst, comparisons=n, ok=bool(worst <= tolerance)) if n else None


# ---------------------------------------------------------------- family bud
def family_bud(root, ref, result):
    per = {b: load_runs(sorted((Path(root) / 'results_V').glob('bud-*/*/seed*/b%d/e3_metrics.jsonl' % b))) for b in BUDGETS}
    print('\n## A. Label budget of the identity refresh (family bud): %s dataset-years with all three seeds per budget' % ', '.join(
        '%d: %d' % (b, sum(sorted(v) == list(A.SEEDS) for v in per[b].values())) for b in BUDGETS))
    if not any(per.values()):
        return
    if ref:
        result['check_bud_g33_exact'] = agreement('budget 2048, same and stale5o against g33', per[2048], ref, ('same', 'stale5o'), EXACT)
        result['check_bud_g33_fit'] = agreement('budget 2048, fit_emb against g33 (a refit)', per[2048], ref, ('fit_emb',), REFIT)
    pairs = collections.OrderedDict([('stale5o / same', ('stale5o', 'same')), ('fit_emb / same', ('fit_emb', 'same')), ('fit_emb / stale5o', ('fit_emb', 'stale5o'))])
    out = result['bud'] = {}
    for label, keys, kw in (('Primary (six districts x 2020, 2025), test portion, with the calibrator', PRIMARY, {}),
                            ('Primary, sensors the old checkpoint does not know', PRIMARY, dict(cohort='new')),
                            ('Primary, sensors the old checkpoint knows', PRIMARY, dict(cohort='old')),
                            ('Primary, validation portion (the first half of the stream; contains the windows on which ridge and forgetting factor were chosen)', PRIMARY, dict(portion='val')),
                            ('Secondary (PEMS05 2020 / 2025, TFNSW 2020)', SECONDARY, {}), ('Development (TFNSW 2025; never part of a decision)', DEVELOPMENT, {})):
        print('\n### %s' % label)
        print('| budget (windows) | dataset-years | stale5o / same | fit_emb / same | fit_emb / stale5o | recovery | fit_emb at this budget / fit_emb at 2048 |\n|---:|---:|---:|---:|---:|---:|---:|')
        table = {}
        for b in BUDGETS:
            ok, missing = complete(per[b], keys, ('same', 'stale5o', 'fit_emb'), **kw)
            if not ok:
                print('| %d | 0 of %d | - | - | - | - | - |' % (b, len(keys))); continue
            items = ratio_items(per[b], ok, pairs, **kw)
            s = {n: A.geometric(items, n) for n in pairs}
            rho = (s['stale5o / same']['gm'] - s['fit_emb / same']['gm']) / (s['stale5o / same']['gm'] - 1.)
            both = [k for k in ok if k in per[2048] and sorted(per[2048][k]) == list(A.SEEDS)]
            cross = A.geometric([(k, dict(r=A.value(per[b][k], 'fit_emb', 'calibrated', **kw) / A.value(per[2048][k], 'fit_emb', 'calibrated', **kw))) for k in both], 'r') if both else None
            c = lambda v: '%.3f (%.3f-%.3f)' % (v['gm'], v['lo'], v['hi'])
            print('| %d | %d of %d | %s | %s | %s | %.0f%% | %s |' % (b, len(ok), len(keys), c(s['stale5o / same']), c(s['fit_emb / same']), c(s['fit_emb / stale5o']), 100 * rho,
                                                              c(cross) if cross else '-'))
            table[b] = dict(n=len(ok), missing=missing, recovery=rho, cross=cross, **s)
        out[label] = table
    prim = out['Primary (six districts x 2020, 2025), test portion, with the calibrator']
    print('\n### Rules fixed in PRESPEC.md (primary set, test portion)')
    rules, usable = {}, True
    for b in BUDGETS:
        t = prim.get(b)
        if not t or t['missing']:
            print('- budget %d: not evaluated, incomplete' % b); usable = False; continue
        r = dict(U1=t['fit_emb / stale5o']['hi'] < 1, U2=t['recovery'] >= .5, U3=t['fit_emb / same']['hi'] <= A.MARGIN)
        rules[b] = {k: bool(v) for k, v in r.items()}
        print('- budget %d: U1 (the refit helps) **%s**; U2 (recovery at least 1/2) **%s** (%.0f%%); U3 (within %.2f of the same-year model) **%s** (%.3f, CI %.3f-%.3f)' % (
            b, 'holds' if r['U1'] else 'does not hold', 'holds' if r['U2'] else 'does not hold', 100 * t['recovery'], A.MARGIN,
            'holds' if r['U3'] else 'does not hold', t['fit_emb / same']['gm'], t['fit_emb / same']['lo'], t['fit_emb / same']['hi']))
    if usable:
        good = [b for b in BUDGETS if all(all(rules[c].values()) for c in BUDGETS if c >= b)]
        print('- the smallest budget at which U1, U2 and U3 hold, and at every larger one: **%s**' % ('%d windows' % min(good) if good else 'none'))
        result['bud_rules'] = dict(per_budget=rules, smallest=min(good) if good else None)


# ---------------------------------------------------------------- family cold
def family_cold(root, ref, result):
    data = collections.defaultdict(dict)
    for d in sorted((Path(root) / 'results_V').glob('cold-*')):
        for k, v in G33.load(d).items():
            data[k].update(v)
    print('\n## B. Sensors that have just arrived (family cold): %d dataset-years, %d with all three seeds' % (len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if not data:
        return
    if ref:
        result['check_cold_g33_exact'] = agreement('same, stale5o against g33', data, ref, ('same', 'stale5o'), EXACT)
        result['check_cold_g33_fit'] = agreement('fit_emb against g33 (a refit)', data, ref, ('fit_emb',), REFIT)
    out = result['cold'] = {}
    for h in HISTORIES:
        arms = ['c%d_%s' % (h, s) for s in SCHEMES]
        c = lambda s: 'c%d_%s' % (h, s)
        pairs = collections.OrderedDict([(a + ' / same', (a, 'same')) for a in ['stale5o', 'fit_emb', c('same')] + arms])
        pairs.update([('prof / ind', (c('prof'), c('ind'))), ('prof / gmean', (c('prof'), c('gmean'))), ('prof / rand', (c('prof'), c('rand'))),
                      ('prof / geo', (c('prof'), c('geo'))), ('prof / pinit', (c('prof'), c('pinit'))), ('prof0 / mean', (c('prof0'), c('mean'))),
                      ('ind / mean', (c('ind'), c('mean'))), ('prof / fit_emb', (c('prof'), 'fit_emb')), ('prof / konly', (c('prof'), 'konly')),
                      ('ind / konly', (c('ind'), 'konly')), ('fit_emb / konly', ('fit_emb', 'konly'))])
        for label, keys, kw, kind in (('new sensors, validation portion (right after the arrival), with the calibrator', PRIMARY, dict(cohort='new', portion='val'), 'calibrated'),
                                      ('new sensors, test portion, with the calibrator', PRIMARY, dict(cohort='new'), 'calibrated'),
                                      ('new sensors, validation portion, without the calibrator', PRIMARY, dict(cohort='new', portion='val'), 'uncalibrated'),
                                      ('known sensors, test portion, with the calibrator', PRIMARY, dict(cohort='old'), 'calibrated'),
                                      ('all sensors, test portion, with the calibrator', PRIMARY, {}, 'calibrated'),
                                      ('secondary (PEMS05 2020 / 2025, TFNSW 2020), new sensors, validation portion, with the calibrator', SECONDARY, dict(cohort='new', portion='val'), 'calibrated'),
                                      ('development (TFNSW 2025; never part of a decision), new sensors, validation portion, with the calibrator', DEVELOPMENT, dict(cohort='new', portion='val'), 'calibrated')):
            ok, missing = complete(data, keys, ['same', 'stale5o', 'fit_emb', 'konly', c('same'), c('mean'), c('ind'), c('gmean'), c('rand'), c('prof'), c('pinit')], **kw)
            print('\n### History %d windows; %s: %d of %d dataset-years%s' % (h, label, len(ok), len(keys), ('; missing ' + ', '.join(missing)) if missing else ''))
            if not ok:
                continue
            items = ratio_items(data, ok, pairs, kind, **kw)
            table = {}
            for name in pairs:
                part = [(k, r) for k, r in items if r[name] is not None]
                if part:
                    table[name] = A.geometric(part, name)
                    line(name, table[name], part)
            out['h%d: %s' % (h, label)] = dict(n=len(ok), missing=missing, **table)
        early = out.get('h%d: new sensors, validation portion (right after the arrival), with the calibrator' % h)
        old = out.get('h%d: known sensors, test portion, with the calibrator' % h)
        print('\n### Rules fixed in PRESPEC.md, history %d windows%s' % (h, '' if h == 64 else ' (reported; the decision is read at 64)'))
        if not early or early['missing'] or not old or old['missing']:
            print('- not evaluated: incomplete'); continue
        r = dict(A1=early['prof / ind']['hi'] < 1, A2=early['prof / gmean']['hi'] < 1 and early['prof / rand']['hi'] < 1,
                 A3=max(early[n]['gm'] for n in ('prof / ind', 'prof / gmean', 'prof / rand')) <= .98, H1=old['prof / konly']['hi'] <= 1.01)
        f = lambda t, n: '%.3f, CI %.3f-%.3f' % (t[n]['gm'], t[n]['lo'], t[n]['hi'])
        v = lambda k: 'holds' if r[k] else 'does not hold'
        print('- A1, a behaviour anchor helps a sensor with a short history (prof / ind CI below 1): **%s** (%s)' % (v('A1'), f(early, 'prof / ind')))
        print('- A2, because it is the right relation (prof / gmean and prof / rand CIs below 1): **%s** (%s; %s)' % (v('A2'), f(early, 'prof / gmean'), f(early, 'prof / rand')))
        print('- A3, by an amount worth a method (all three geometric means at most 0.98): **%s**' % v('A3'))
        print('- H1, the arrivals do not harm the mature sensors (known sensors, test portion, prof / konly CI at most 1.01): **%s** (%s)' % (v('H1'), f(old, 'prof / konly')))
        print('- a relational anchor for short-history identities (needs A1, A2, A3): **%s**' % ('enters the method' if r['A1'] and r['A2'] and r['A3'] else 'does not enter the method'))
        result.setdefault('cold_rules', {})[h] = {k: bool(x) for k, x in r.items()}
    fits = collections.defaultdict(list)
    for f in sorted((Path(root) / 'results_V').glob('cold-*/*/seed*/e10_info.json')):
        info = json.loads(f.read_text())
        for tag, rec in info.items():
            if isinstance(rec, dict) and 'holdout_mae' in rec:
                fits[tag].append((min(rec['holdout_mae']) / rec['holdout_mae'][0], rec['best_epoch'], rec['seconds']))
    if fits:
        print('\n| fit | runs | holdout after / before (median) | best epoch (median) | seconds (median, max) |\n|---|---:|---:|---:|---:|')
        for tag in sorted(fits):
            r = np.asarray(fits[tag], float)
            print('| %s | %d | %.3f | %d | %.0f, %.0f |' % (tag, len(r), np.median(r[:, 0]), np.median(r[:, 1]), np.median(r[:, 2]), r[:, 2].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    ref = collections.defaultdict(dict)
    for root in a.g33:
        for k, v in G33.load(root).items():
            ref[k].update(v)
    result = {}
    print('# g38: label budget of the identity refresh, and sensors that have just arrived')
    print('Endpoint: H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95% CI by the g29 bootstrap over districts.')
    family_bud(a.root, ref, result)
    family_cold(a.root, ref, result)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
