"""Pre-specified summary of g39 (PRESPEC.md): identity ageing of STAEFormer with and without time inputs. Read-only.

  python analyze_g39.py DELIVERED_ROOT [--g33 ROOT ...] [--out summary.json]

DELIVERED_ROOT holds results_V/evl-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/{e3_metrics.jsonl, e10_info.json, ...}
(e19_timeinput.py) and ckpt_V/trn-<dataset>-<year>-<variant>-s<seed>/train_report.json (train_stae.py).
--g33: g33's delivered runs; their arms same / stale5o must equal this package's same_rp / stale_rp.
Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap over districts.
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
ARMS = ('same_rp', 'stale_rp', 'same_nt', 'stale_nt', 'fit_nt', 'same_tm', 'stale_tm', 'fit_tm')
PAIRS = collections.OrderedDict()
for v in ('nt', 'tm', 'rp'):
    PAIRS['stale_%s / same_%s' % (v, v)] = ('stale_' + v, 'same_' + v)
for v in ('nt', 'tm'):
    PAIRS['fit_%s / same_%s' % (v, v)] = ('fit_' + v, 'same_' + v)
    PAIRS['fit_%s / stale_%s' % (v, v)] = ('fit_' + v, 'stale_' + v)
for a in ('same', 'stale', 'fit'):
    PAIRS['%s_tm / %s_nt' % (a, a)] = (a + '_tm', a + '_nt')
PAIRS['same_nt / same_rp'] = ('same_nt', 'same_rp')
PAIRS['stale_nt / stale_rp'] = ('stale_nt', 'stale_rp')
EXACT = 1e-5


def agreement(mine, ref):
    """same_rp / stale_rp of this package against g33's same / stale5o, every portion, cohort and both kinds."""
    worst, n, where = 0., 0, None
    for k, runs in mine.items():
        for seed, rows in runs.items():
            for (arm, portion, cohort), row in rows.items():
                theirs = {'same_rp': 'same', 'stale_rp': 'stale5o'}.get(arm)
                other = ref.get(k, {}).get(seed, {}).get((theirs, portion, cohort)) if theirs else None
                if other is None:
                    continue
                for kind in ('calibrated', 'uncalibrated'):
                    d = abs(row[kind]['H12_MAE'] - other[kind]['H12_MAE']) / other[kind]['H12_MAE']
                    n += 1
                    if d > worst:
                        worst, where = d, '%s/%d seed %d %s/%s/%s' % (k[0], k[1], seed, arm, portion, cohort)
    if n:
        print('Check, same_rp / stale_rp against g33\'s same / stale5o: max relative difference %.2e over %d comparisons%s (tolerance %.0e: %s)' % (
            worst, n, ' (%s)' % where if where else '', EXACT, 'OK' if worst <= EXACT else 'DIFFERS: do not read the numbers that depend on it'))
    return dict(worst=worst, comparisons=n, ok=bool(worst <= EXACT)) if n else None


def view(data, label, kind='calibrated', **kw):
    ok, missing = [], []
    for k in PRIMARY:
        if k in data and sorted(data[k]) == list(A.SEEDS) and all(A.value(data[k], a, 'calibrated', **kw) is not None for a in ARMS):
            ok.append(k)
        else:
            missing.append('%s/%d' % k)
    print('\n### %s: %d of %d dataset-years%s' % (label, len(ok), len(PRIMARY), ('; missing ' + ', '.join(missing)) if missing else ''))
    if not ok:
        return dict(n=0, missing=missing)
    items = []
    for k in ok:
        v = lambda arm: A.value(data[k], arm, kind, **kw)
        items.append((k, {name: v(a) / v(b) for name, (a, b) in PAIRS.items()}))
    table = {}
    for name in PAIRS:
        s = table[name] = A.geometric(items, name)
        print('- %s: **%.3f** (95%% CI %.3f-%.3f; range %.3f-%.3f; below 1 in %d of %d)' % (name, s['gm'], s['lo'], s['hi'], s['min'], s['max'],
                                                                                         sum(1 for _, r in items if r[name] < 1), s['n']))
    for v in ('nt', 'tm', 'rp'):
        if v != 'rp':
            st_, fi = table['stale_%s / same_%s' % (v, v)]['gm'], table['fit_%s / same_%s' % (v, v)]['gm']
            table['recovery_' + v] = (st_ - fi) / (st_ - 1.) if st_ > 1 else None
            print('- recovery with the refit, %s: %s' % (v, '%.0f%%' % (100 * table['recovery_' + v]) if table['recovery_' + v] is not None else 'not defined (no ageing)'))
    return dict(n=len(ok), missing=missing, **table)


def training(root):
    rows = collections.defaultdict(list)
    for f in sorted((Path(root) / 'ckpt_V').glob('trn-*/train_report.json')):
        rep = json.loads(f.read_text())
        for name, r in rep['runs'].items():
            if r['dataset'] in SIX:
                rows[r['variant']].append(r)
    if not rows:
        return None
    print('\n| variant | models | epochs (median, max) | best epoch (median) | test H12 MAE with train statistics, median | pieces per batch of 16 (max) | hours of training (sum) |\n|---|---:|---:|---:|---:|---:|---:|')
    out = {}
    for v in ('nt', 'tm'):
        r = rows.get(v, [])
        if not r:
            continue
        e = np.array([x['epochs_run'] for x in r]); b = np.array([x['best_epoch'] for x in r])
        out[v] = dict(models=len(r), epochs_median=float(np.median(e)), epochs_max=int(e.max()), best_median=float(np.median(b)),
                      test_T12_median=float(np.median([x['metrics']['test_train_stats']['T12_MAE'] for x in r])), hours=float(sum(x['seconds'] for x in r) / 3600),
                      hit_epoch_limit=int(sum(1 for x in r if x['epochs_run'] >= int(x['conf']['epoch']))))
        print('| %s | %d | %d, %d | %d | %.2f | %d | %.1f |' % (v, len(r), np.median(e), e.max(), np.median(b), out[v]['test_T12_median'],
                                                             max(16 // max(x.get('micro_batch', 16), 1) for x in r), out[v]['hours']))
    ref = [(x['metrics']['test_train_stats']['T12_MAE'], x['repository_checkpoint']['metrics']['test_train_stats']['T12_MAE']) for x in rows.get('nt', []) if 'repository_checkpoint' in x]
    if ref:
        r = np.array(ref)
        out['nt_over_repository_test'] = float(np.exp(np.log(r[:, 0] / r[:, 1]).mean()))
        print('- retrained without time inputs / the repository\'s checkpoint of the same year and seed, test H12 MAE (train statistics), geometric mean over %d models: %.3f' % (
            len(r), out['nt_over_repository_test']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    result = {}
    data = G33.load(Path(a.root) / 'results_V')
    print('# g39: identity ageing of STAEFormer with and without time-of-day / day-of-week inputs')
    print('Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95%% CI by the g29 bootstrap over districts. Runs found: %d dataset-years, %d with all three seeds.' % (
        len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if a.g33:
        ref = collections.defaultdict(dict)
        for root in a.g33:
            for k, v in G33.load(root).items():
                ref[k].update(v)
        result['check_g33'] = agreement(data, ref)
    print('\n### Training (primary districts, all panel years)')
    result['training'] = training(a.root)
    cal = result['test'] = view(data, 'Primary, test portion, with the calibrator')
    result['test_uncalibrated'] = view(data, 'Primary, test portion, without the calibrator', 'uncalibrated')
    result['new'] = view(data, 'Primary, sensors the old checkpoint does not know, test portion, with the calibrator', cohort='new')
    result['old'] = view(data, 'Primary, sensors the old checkpoint knows, test portion, with the calibrator', cohort='old')
    print('\n### Rules fixed in PRESPEC.md (primary set, test portion, with the calibrator)')
    if not cal.get('n') or cal['missing']:
        print('- not evaluated: incomplete (%s)' % ', '.join(cal.get('missing', [])))
    else:
        f = lambda n: '%.3f, CI %.3f-%.3f' % (cal[n]['gm'], cal[n]['lo'], cal[n]['hi'])
        gap_tm, gap_nt = cal['stale_tm / same_tm']['gm'] - 1., cal['stale_nt / same_nt']['gm'] - 1.
        share = gap_tm / gap_nt if gap_nt > 0 else None
        tr = result.get('training') or {}
        q = tr.get('nt_over_repository_test')
        r = dict(Q=bool(q is not None and q <= 1.05), T1=cal['stale_tm / same_tm']['lo'] > 1,
                 T2=cal['fit_tm / stale_tm']['hi'] < 1 and (cal['recovery_tm'] or 0.) >= .5, T3=cal['fit_tm / same_tm']['hi'] <= A.MARGIN,
                 T4=bool(share is not None and share >= .5))
        v = lambda k: 'holds' if r[k] else 'does not hold'
        print('- Q, the retraining reproduces the repository\'s model (retrained without time inputs / repository, test, at most 1.05): **%s** (%s)' % (v('Q'), '%.3f' % q if q is not None else 'no reference'))
        print('- T1, with time inputs the five-year-old model is still worse (stale_tm / same_tm, CI above 1): **%s** (%s)' % (v('T1'), f('stale_tm / same_tm')))
        print('- T2, refitting its identities repairs most of it (fit_tm / stale_tm CI below 1, recovery at least 1/2): **%s** (%s; recovery %s)' % (
            v('T2'), f('fit_tm / stale_tm'), '%.0f%%' % (100 * cal['recovery_tm']) if cal['recovery_tm'] is not None else 'not defined'))
        print('- T3, to within %.2f of the same-year model (fit_tm / same_tm CI): **%s** (%s)' % (A.MARGIN, v('T3'), f('fit_tm / same_tm')))
        print('- T4, time inputs do not explain most of the ageing (excess of stale_tm / same_tm at least half that of stale_nt / same_nt): **%s** (%s)' % (
            v('T4'), 'with time inputs %.1f%%, without %.1f%%, share %s' % (100 * gap_tm, 100 * gap_nt, '%.2f' % share if share is not None else 'not defined')))
        print('- "the ageing of the identity is not an artefact of the missing time inputs" (needs T1, T2, T4): **%s**' % ('holds' if r['T1'] and r['T2'] and r['T4'] else 'does not hold'))
        result['rules'] = dict({k: bool(x) for k, x in r.items()}, share=share, recovery_tm=cal['recovery_tm'], recovery_nt=cal['recovery_nt'])
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
