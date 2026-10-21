"""Pre-specified summary of g32 (PRESPEC.md): relational residual branch against the local path and its controls (read-only).

  python analyze_g32.py RESULT_ROOT [--family gcn|stae|gcno] [--g29 G29_RESULT_ROOT]

RESULT_ROOT holds .../<dataset>_<year>/seed<seed>/e3_metrics.jsonl (the virga layout results_V/rra-<family>-<dataset>-<year>/...
or the development layout). Endpoint: test-portion H12 MAE, mean of seeds 51-53; geometric means with the g29 bootstrap over
districts. Rules: G = rra/base, S = rra/self, R = rra/rand, F = rra/fixed, P = geo/rand, each "holds" if the upper CI bound is
below 1; a graph component can be claimed only if S and R hold and the geometric mean of S is at most 0.99.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A

RATIOS = collections.OrderedDict([
    ('G = rra / base', ('rra', 'base')), ('S = rra / self', ('rra', 'self')), ('R = rra / rand', ('rra', 'rand')),
    ('F = rra / fixed', ('rra', 'fixed')), ('P = geo / rand', ('geo', 'rand')),
    ('self / base', ('self', 'base')), ('rand / base', ('rand', 'base')), ('fixed / base', ('fixed', 'base')), ('geo / base', ('geo', 'base'))])
RULES = ('G = rra / base', 'S = rra / self', 'R = rra / rand', 'F = rra / fixed', 'P = geo / rand')


def load(root, family):
    data = collections.defaultdict(dict)
    for f in sorted(Path(root).rglob('e3_metrics.jsonl')):
        if not (f.parent / 'completed.json').exists() or (family and ('rra-%s-' % family) not in str(f)):
            continue
        if json.loads((f.parent / 'completed.json').read_text()).get('bounded_smoke'):
            continue
        rows = [json.loads(line) for line in f.open()]
        key, seed = (rows[0]['dataset'], int(rows[0]['year'])), int(rows[0]['seed'])
        assert seed not in data[key], ('two runs of the same dataset-year and seed under this root; pass --family', key, seed)
        data[key][seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
    return data


def items_of(data, keys, cohort='all', metric='H12_MAE', portion='test'):
    items, missing = [], []
    for k in keys:
        if k not in data or sorted(data[k]) != list(A.SEEDS):
            missing.append('%s/%d' % k); continue
        row = {}
        for name, (num, den) in RATIOS.items():
            a, b = (A.value(data[k], arm, 'calibrated', metric, portion, cohort) for arm in (num, den))
            row[name] = None if None in (a, b) else a / b
        if row['G = rra / base'] is None:
            missing.append('%s/%d' % k); continue
        items.append((k, row))
    return items, missing


def summary(items, names=RATIOS):
    out = {}
    for name in names:
        part = [(k, r) for k, r in items if r.get(name) is not None]
        if part:
            out[name] = A.geometric(part, name)
    return out


def show(label, items, missing, expected):
    print('\n### %s: %d of %d dataset-years%s' % (label, len(items), expected, ('; missing ' + ', '.join(missing)) if missing else ''))
    out = summary(items)
    for name, s in out.items():
        print('- %s: geometric mean **%.4f** (95%% CI %.4f-%.4f; range %.4f-%.4f; below 1 in %d of %d dataset-years, %d districts)' % (
            name, s['gm'], s['lo'], s['hi'], s['min'], s['max'], sum(r[name] < 1 for _, r in items if r.get(name) is not None), s['n'], s['districts']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--family', default='', help='gcn (Online-AN, same year), stae (STAEFormer, same year), gcno (Online-AN, five years old)')
    ap.add_argument('--g29', help='g29 result root: checks that the base arm equals the g29 same-year arm')
    ap.add_argument('--out')
    a = ap.parse_args()
    data = load(a.root, a.family)
    print('# g32: relational residual branch%s' % (' (%s)' % a.family if a.family else ''))
    print('Endpoint: test-portion H12 MAE, mean of seeds 51-53. Runs found: %d dataset-years, %d with all three seeds.' % (
        len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if a.g29:
        ref, worst, n = A.load([a.g29]), 0., 0
        for k, runs in data.items():
            for seed, rows in runs.items():
                if k in ref and seed in ref[k]:
                    for kind in ('calibrated', 'uncalibrated'):
                        x, y = rows[('base', 'test', 'all')][kind]['H12_MAE'], ref[k][seed][('same', 'test', 'all')][kind]['H12_MAE']
                        worst, n = max(worst, abs(x - y) / y), n + 1
        print('Base arm against the g29 same-year arm: max relative difference %.2e over %d comparisons (%s)' % (
            worst, n, 'OK' if worst <= 1e-6 else 'DIFFERS: do not read the numbers below'))
    result = {}
    tier_a, miss = items_of(data, A.TIER_A)
    result['tier_a'] = show('Tier A (eight held-out districts)', tier_a, miss, len(A.TIER_A))
    tier_b, miss_b = items_of(data, A.TIER_B)
    result['tier_b'] = show('Tier B (PEMS05, TFNSW; other years)', tier_b, miss_b, len(A.TIER_B))
    dev, miss_d = items_of(data, sorted(A.DEVELOPMENT))
    result['development'] = show('Development dataset-years (never part of a decision)', dev, miss_d, len(A.DEVELOPMENT))
    if tier_a:
        print('\n| dataset/year | base | rra | self | rand | fixed | geo | G | S | R | F |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for (d, y), r in tier_a + tier_b + dev:
            v = lambda arm: A.value(data[(d, y)], arm, 'calibrated')
            geo = v('geo')
            print('| %s/%d | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %.4f | %.4f | %.4f | %.4f |' % (
                d, y, v('base'), v('rra'), v('self'), v('rand'), v('fixed'), '%.3f' % geo if geo else '-',
                r['G = rra / base'], r['S = rra / self'], r['R = rra / rand'], r['F = rra / fixed']))
        print('\n### Tier A, descriptive')
        for cohort in ('old', 'new'):
            part, _ = items_of(data, A.TIER_A, cohort)
            s = summary(part, ('G = rra / base', 'S = rra / self'))
            if s:
                print('- sensors %s five years earlier (%d dataset-years): G %.4f (%.4f-%.4f), S %.4f (%.4f-%.4f)' % (
                    'present' if cohort == 'old' else 'absent', len(part), s['G = rra / base']['gm'], s['G = rra / base']['lo'], s['G = rra / base']['hi'],
                    s['S = rra / self']['gm'], s['S = rra / self']['lo'], s['S = rra / self']['hi']))
        for metric, portion in (('Avg_MAE', 'test'), ('H3_MAE', 'test'), ('H12_MAE', 'val')):
            part, _ = items_of(data, A.TIER_A, 'all', metric, portion)
            s = summary(part, ('G = rra / base', 'S = rra / self', 'R = rra / rand'))
            print('- %s, %s portion: G %.4f, S %.4f, R %.4f' % (metric, portion, s['G = rra / base']['gm'], s['S = rra / self']['gm'], s['R = rra / rand']['gm']))
    print('\n### Rules fixed in PRESPEC.md (Tier A)')
    if len(tier_a) != len(A.TIER_A):
        print('- not evaluated: Tier A is incomplete')
    else:
        t = result['tier_a']
        holds = {name: (t[name]['hi'] < 1) for name in RULES if name in t}
        for name in RULES:
            if name in t:
                print('- %s: **%s** (%.4f, CI %.4f-%.4f)' % (name, 'holds' if holds[name] else 'does not hold', t[name]['gm'], t[name]['lo'], t[name]['hi']))
        s, r = holds['S = rra / self'], holds['R = rra / rand']
        if s and r and t['S = rra / self']['gm'] <= .99:
            verdict = 'a graph component can be claimed (S and R hold, S at least 1%)'
        elif s and r:
            verdict = 'detectable, below 1%: S and R hold but the gain over the same-size adapter without neighbours is under 1%'
        elif holds['G = rra / base'] and not s:
            verdict = 'no graph component: the gain over the local path is not larger than that of the same-size adapter without neighbours'
        elif s and not r:
            verdict = 'other sensors help, but any set of them serves as context (random candidates do as well)'
        else:
            verdict = 'no graph component: the branch does not improve the local path'
        print('- reading: **%s**' % verdict)
        result['verdict'] = verdict
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
