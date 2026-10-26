"""Pre-registered analysis of the g29 stale-backbone confirmation (read-only).

  python analyze_g29.py RESULT_ROOT [RESULT_ROOT ...] [--out summary.json]

A result root is searched recursively for <DATASET>_<year>/seed<seed>/e3_metrics.jsonl.
Endpoint: test-portion H12 MAE, mean over the three seeds. Ratios are ratios of those means;
a tier is summarised by the geometric mean over its dataset-years with a percentile
bootstrap over districts (10,000 draws). Decision rules are in PREREGISTRATION.md.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np

SEEDS = (51, 52, 53)
DEVELOPMENT = {('PEMS05', 2015), ('TFNSW', 2025)}
TIER_A = [(d, y) for d in ('PEMS03', 'PEMS04', 'PEMS06', 'PEMS07', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
          for y in (2015, 2020, 2025)]
TIER_B = [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2015), ('TFNSW', 2020)]
MARGIN, DRAWS, BOOT_SEED = 1.10, 10000, 2029
METRIC = 'H12_MAE'


def load(roots):
    """{(dataset, year): {seed: {(arm, portion, cohort): row}}}; a later root never overrides an earlier one."""
    data = collections.defaultdict(dict)
    for root in roots:
        for f in sorted(Path(root).rglob('e3_metrics.jsonl')):
            if not (f.parent / 'completed.json').exists():
                continue
            rows = [json.loads(line) for line in f.open()]
            key, seed = (rows[0]['dataset'], int(rows[0]['year'])), int(rows[0]['seed'])
            if seed not in data[key]:
                data[key][seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
    return data


def value(runs, arm, kind, metric=METRIC, portion='test', cohort='all'):
    """Mean over seeds; None unless every seed has the arm."""
    vals = []
    for seed in SEEDS:
        row = runs.get(seed, {}).get((arm, portion, cohort))
        if row is None or row[kind] is None:
            return None
        vals.append(row[kind][metric])
    return float(np.mean(vals))


def ratios(runs, stale='stale5', metric=METRIC, cohort='all'):
    v = lambda arm, kind: value(runs, arm, kind, metric, 'test', cohort)
    same, same_raw, st, st_raw = v('same', 'calibrated'), v('same', 'uncalibrated'), v(stale, 'calibrated'), v(stale, 'uncalibrated')
    nobb = [x for x in (v('nobb_zero', 'calibrated'), v('nobb_last', 'calibrated')) if x is not None]
    if None in (same, same_raw, st, st_raw) or not nobb:
        return None
    own = v(stale + 'o', 'calibrated')
    return dict(same=same_raw, same_rls=same, stale=st_raw, stale_rls=st, stale_own_rls=own, stale_own=v(stale + 'o', 'uncalibrated'),
                nobb_rls=min(nobb), naive=v('nobb_last', 'uncalibrated'),
                R1=st / same, R2=st / same_raw, R3=st / min(nobb), R0=st_raw / same_raw,
                R1_own=None if own is None else own / same,
                gap_closed=None if st_raw == same else (st_raw - st) / (st_raw - same))


def geometric(items, key):
    """Geometric mean of a ratio over (dataset, year) items with a district-clustered bootstrap CI."""
    logs = collections.defaultdict(list)
    for (dataset, _), r in items:
        if r[key] is not None:
            logs[dataset].append(np.log(r[key]))
    if not logs:
        return None
    everything = np.concatenate([np.asarray(v) for v in logs.values()])
    districts = sorted(logs)
    rng = np.random.RandomState(BOOT_SEED)
    draws = []
    for _ in range(DRAWS):
        pick = rng.randint(len(districts), size=len(districts))
        draws.append(np.concatenate([logs[districts[i]] for i in pick]).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    by_district = {d: float(np.exp(np.mean(v))) for d, v in logs.items()}
    return dict(gm=float(np.exp(everything.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)), n=int(len(everything)),
                districts=len(districts), min=float(np.exp(everything.min())), max=float(np.exp(everything.max())),
                by_district=by_district)


def tier_summary(data, tier, label, lines):
    items = [((d, y), ratios(data[(d, y)])) for d, y in tier if (d, y) in data]
    missing = [k for k in tier if k not in data] + [k for k, r in items if r is None]
    items = [(k, r) for k, r in items if r is not None]
    out = dict(label=label, expected=len(tier), available=len(items), missing=['%s/%d' % k for k in missing])
    lines.append('\n### %s: %d of %d dataset-years available%s' % (label, len(items), len(tier),
                 ('; MISSING ' + ', '.join(out['missing'])) if missing else ''))
    if not items:
        return out
    lines.append('| dataset/year | same | same+RLS | stale5 | stale5+RLS | stale5(own)+RLS | no-backbone+RLS | R1 | R2 | R3 | gap closed |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    f = lambda v, d=2: '--' if v is None else '%.*f' % (d, v)
    for (d, y), r in items:
        lines.append('| %s/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            d, y, f(r['same']), f(r['same_rls']), f(r['stale']), f(r['stale_rls']), f(r['stale_own_rls']), f(r['nobb_rls']),
            f(r['R1'], 3), f(r['R2'], 3), f(r['R3'], 3), '--' if r['gap_closed'] is None else '%.0f%%' % (100 * r['gap_closed'])))
    for key, text, threshold in (('R2', 'R2 = stale5+RLS / same (uncalibrated)', 1.), ('R1', 'R1 = stale5+RLS / same+RLS', MARGIN),
                                 ('R3', 'R3 = stale5+RLS / best no-backbone RLS', 1.), ('R0', 'R0 = stale5 / same (both uncalibrated)', None),
                                 ('R1_own', 'R1 with the checkpoint\'s own input statistics', None)):
        s = geometric(items, key)
        if s is None:
            continue
        s['threshold'] = threshold
        if threshold is not None:
            s['dataset_years_below'] = int(sum(r[key] < threshold for _, r in items if r[key] is not None))
            s['districts_below'] = int(sum(v < threshold for v in s['by_district'].values()))
            s['ci_upper_below_threshold'] = bool(s['hi'] < threshold if threshold == 1. else s['hi'] <= threshold)
        out[key] = s
        lines.append('- %s: geometric mean **%.3f** (95%% CI %.3f-%.3f, %d dataset-years, %d districts; range %.3f-%.3f)%s' % (
            text, s['gm'], s['lo'], s['hi'], s['n'], s['districts'], s['min'], s['max'],
            '' if threshold is None else '; below %.2f in %d/%d dataset-years and %d/%d districts; CI upper bound %s the threshold' % (
                threshold, s['dataset_years_below'], s['n'], s['districts_below'], s['districts'],
                'meets' if s['ci_upper_below_threshold'] else 'DOES NOT meet')))
    return out


def ages(data, lines):
    """Calibrated and uncalibrated cost of an a-year-old backbone relative to the same-year one."""
    out = {}
    for label, keep in (('yearly districts, targets 2010-2015 (PEMS06, PEMS11, PEMS12)',
                         lambda d, y: d in ('PEMS06', 'PEMS11', 'PEMS12') and 2010 <= y <= 2015),
                        ('panel targets 2015/2020/2025, PeMS districts except PEMS05/2015',
                         lambda d, y: d.startswith('PEMS') and y in (2015, 2020, 2025) and (d, y) not in DEVELOPMENT),
                        ('TFNSW 2015/2020/2026', lambda d, y: d == 'TFNSW' and (d, y) not in DEVELOPMENT)):
        table = collections.defaultdict(list)
        for (d, y), runs in sorted(data.items()):
            if not keep(d, y):
                continue
            arms = sorted({k[0] for k in runs.get(SEEDS[0], {}) if k[0].startswith('stale') and not k[0].endswith('o')},
                          key=lambda a: int(a[5:]))
            for arm in arms:
                r = ratios(runs, arm)
                if r is not None:
                    table[int(arm[5:])].append(((d, y), r))
        if not table:
            continue
        lines.append('\n### Backbone age: %s' % label)
        lines.append('| age (years) | pairs | districts | stale+RLS / same+RLS | own-statistics variant | stale / same (uncalibrated) | stale+RLS / same (uncal.) | stale+RLS / no-backbone RLS |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
        out[label] = {}
        for age in sorted(table):
            s = {k: geometric(table[age], k) for k in ('R1', 'R1_own', 'R0', 'R2', 'R3')}
            out[label][age] = s
            cell = lambda k: '--' if s[k] is None else '%.3f (%.3f-%.3f)' % (s[k]['gm'], s[k]['min'], s[k]['max'])
            lines.append('| %d | %d | %d | %s | %s | %s | %s | %s |' % (age, s['R1']['n'], s['R1']['districts'], cell('R1'),
                                                                   cell('R1_own'), cell('R0'), cell('R2'), cell('R3')))
        lines.append('(geometric mean over target-year/checkpoint pairs; range in parentheses)')
    return out


def secondary(data, tier, lines):
    items = {k: data[k] for k in tier if k in data}
    out = {}
    lines.append('\n### Tier A, other metrics and sensor cohorts (descriptive)')
    for metric in ('H3_MAE', 'H6_MAE', 'H12_MAE', 'Avg_MAE', 'H12_RMSE'):
        rs = [(k, ratios(runs, 'stale5', metric)) for k, runs in items.items()]
        rs = [(k, r) for k, r in rs if r is not None]
        if rs:
            s1, s2 = geometric(rs, 'R1'), geometric(rs, 'R2')
            out[metric] = dict(R1=s1, R2=s2)
            lines.append('- %s: R1 %.3f (%.3f-%.3f), R2 %.3f (%.3f-%.3f)' % (metric, s1['gm'], s1['lo'], s1['hi'], s2['gm'], s2['lo'], s2['hi']))
    for cohort, text in (('old', 'sensors that existed five years earlier'), ('new', 'sensors absent five years earlier')):
        rs = [(k, ratios(runs, 'stale5', METRIC, cohort)) for k, runs in items.items()
              if (('stale5', 'test', cohort) in runs.get(SEEDS[0], {}))]
        rs = [(k, r) for k, r in rs if r is not None]
        if rs:
            s1, s0 = geometric(rs, 'R1'), geometric(rs, 'R0')
            out[cohort] = dict(R1=s1, R0=s0)
            lines.append('- %s (%d dataset-years): R1 %.3f (%.3f-%.3f); uncalibrated R0 %.3f' % (text, s1['n'], s1['gm'], s1['lo'], s1['hi'], s0['gm']))
    spread = []
    for k, runs in items.items():
        per_seed = [runs[s][('stale5', 'test', 'all')]['calibrated'][METRIC] / runs[s][('same', 'test', 'all')]['calibrated'][METRIC]
                    for s in SEEDS if s in runs and ('stale5', 'test', 'all') in runs[s]]
        if len(per_seed) == len(SEEDS):
            spread.append(max(per_seed) - min(per_seed))
    if spread:
        out['seed_range_R1'] = dict(median=float(np.median(spread)), max=float(np.max(spread)))
        lines.append('- per-seed R1 range within a dataset-year: median %.3f, max %.3f' % (np.median(spread), np.max(spread)))
    chosen = collections.Counter()
    for k, runs in items.items():
        for s in SEEDS:
            for arm in ('same', 'stale5'):
                row = runs.get(s, {}).get((arm, 'test', 'all'))
                if row:
                    chosen[(arm, row['ridge'], row['forget'])] += 1
    out['selected_hyperparameters'] = {'%s ridge=%g forget=%g' % k: v for k, v in sorted(chosen.items())}
    lines.append('- selected (ridge, forget) counts: ' + ', '.join('%s: %d' % kv for kv in out['selected_hyperparameters'].items()))
    return out


def sanity(data, lines):
    problems = []
    for key, runs in sorted(data.items()):
        if sorted(runs) != list(SEEDS):
            problems.append('%s/%d: seeds %s' % (key + (sorted(runs),)))
            continue
        for arm in ('nobb_zero', 'nobb_last'):       # no-backbone arms cannot depend on the backbone seed
            vals = [runs[s][(arm, 'test', 'all')]['calibrated'][METRIC] for s in SEEDS]
            if max(vals) - min(vals) > 1e-6 * max(vals):
                problems.append('%s/%d: %s differs across seeds %s' % (key + (arm, vals)))
    lines.append('\n### Sanity\n- dataset-years found: %d; problems: %s' % (len(data), problems or 'none'))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--out')
    opt = ap.parse_args()
    data = load(opt.roots)
    lines = ['# g29 stale-backbone confirmation: pre-registered analysis',
             'Endpoint: test-portion H12 MAE, mean of seeds 51-53. R1/R2/R3 as defined in PREREGISTRATION.md.']
    summary = dict(problems=sanity(data, lines))
    summary['tier_a'] = tier_summary(data, TIER_A, 'Tier A (held-out regions, primary)', lines)
    summary['tier_b'] = tier_summary(data, TIER_B, 'Tier B (development regions, other years)', lines)
    summary['development'] = tier_summary(data, sorted(DEVELOPMENT), 'Development dataset-years (not confirmation)', lines)
    summary['secondary'] = secondary(data, TIER_A, lines)
    summary['ages'] = ages(data, lines)
    a = summary['tier_a']
    if all(k in a for k in ('R1', 'R2', 'R3')) and a['available'] == a['expected']:
        verdict = dict(H1=a['R2']['ci_upper_below_threshold'], H2=a['R1']['ci_upper_below_threshold'], H3=a['R3']['ci_upper_below_threshold'])
        lines.append('\n### Pre-registered decisions (Tier A)\n- H1 (R2 < 1): %s\n- H2 (R1 <= %.2f): %s\n- H3 (R3 < 1): %s' % (
            'CONFIRMED' if verdict['H1'] else 'NOT CONFIRMED', MARGIN, 'CONFIRMED' if verdict['H2'] else 'NOT CONFIRMED',
            'CONFIRMED' if verdict['H3'] else 'NOT CONFIRMED'))
        summary['verdict'] = verdict
    else:
        lines.append('\n### Pre-registered decisions (Tier A)\n- not evaluated: Tier A is incomplete')
    print('\n'.join(lines))
    if opt.out:
        Path(opt.out).write_text(json.dumps(summary, indent=1, default=lambda o: {str(k): v for k, v in o.items()}) + '\n')


if __name__ == '__main__':
    main()
