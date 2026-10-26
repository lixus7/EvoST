"""Pre-specified summary of g40 (PRESPEC.md): borrow at arrival, refit at stream window 1,024. Read-only.

  python analyze_g40.py DELIVERED_ROOT [--ref ROOT ...] [--out summary.json]

DELIVERED_ROOT/results_V holds hnd-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/{e3_metrics.jsonl, e3_sums.npz, e10_info.json, ...}
(e18_handover.py). --ref: runs that hold the static arms of the same dataset-years and seeds: g38's delivered root (its cold-*
tasks are read) or any directory of runs; a later one replaces an earlier one.
Endpoint: H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap over districts.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A
import analyze_g33 as G33
import analyze_g38 as G38

PRIMARY, SECONDARY, DEVELOPMENT = G38.PRIMARY, G38.SECONDARY, G38.DEVELOPMENT
METHOD = 'r'                                                             # the calibrator hand-over of the method arm, fixed in PRESPEC.md from development data
OTHER = dict(k='r', r='k')[METHOD]
EVENT, RELEASED = 1024, 1013
STATIC = ('same', 'stale5o', 'fit_emb', 'konly', 'c64_same', 'c64_mean', 'c64_prof0')
SCHEDULED = ('s_mean_k', 's_mean_r', 's_prof0_k', 's_prof0_r')
EXACT, REFIT = (('same', 'stale5o', 'c64_same'), 1e-5), (('fit_emb', 'konly', 'c64_mean', 'c64_prof0'), 2e-3)
S, M = 's_prof0_' + METHOD, 's_mean_' + METHOD
PAIRS = collections.OrderedDict([(a + ' / same', (a, 'same')) for a in STATIC[1:] + SCHEDULED] + [
    ('%s / c64_prof0' % S, (S, 'c64_prof0')), ('%s / %s' % (S, M), (S, M)), ('%s / c64_same' % S, (S, 'c64_same')), ('%s / fit_emb' % S, (S, 'fit_emb')),
    ('%s / konly' % S, (S, 'konly')), ('%s / c64_mean' % M, (M, 'c64_mean')), ('s_prof0_%s / s_prof0_%s' % (METHOD, OTHER), (S, 's_prof0_' + OTHER)),
    ('s_prof0_%s / c64_prof0' % OTHER, ('s_prof0_' + OTHER, 'c64_prof0')), ('s_prof0_%s / s_mean_%s' % (OTHER, OTHER), ('s_prof0_' + OTHER, 's_mean_' + OTHER)),
    ('s_prof0_%s / konly' % OTHER, ('s_prof0_' + OTHER, 'konly')), ('c64_prof0 / c64_mean', ('c64_prof0', 'c64_mean')), ('c64_prof0 / konly', ('c64_prof0', 'konly')),
    ('c64_prof0 / fit_emb', ('c64_prof0', 'fit_emb')), ('fit_emb / konly', ('fit_emb', 'konly'))])


def load_reference(roots):
    ref = collections.defaultdict(dict)
    for root in roots:
        root = Path(root)
        parts = sorted((root / 'results_V').glob('cold-*')) if (root / 'results_V').is_dir() else [root]
        for d in parts:
            for k, v in G33.load(d).items():
                ref[k].update(v)
    return ref


def agreement(label, mine, ref, arms, tolerance):
    """Largest relative difference of H12 MAE over every portion, cohort and both kinds, for the given arms."""
    worst, n, where = 0., 0, None
    for k, runs in mine.items():
        for seed, rows in runs.items():
            for key, row in rows.items():
                other = ref.get(k, {}).get(seed, {}).get(key)
                if key[0] in arms and other is not None:
                    for kind in ('calibrated', 'uncalibrated'):
                        if row[kind] is not None and other[kind] is not None:
                            d = abs(row[kind]['H12_MAE'] - other[kind]['H12_MAE']) / other[kind]['H12_MAE']
                            n += 1
                            if d > worst:
                                worst, where = d, '%s/%d seed %d %s' % (k[0], k[1], seed, '/'.join(key))
    if n:
        print('Check, %s: max relative difference %.2e over %d comparisons%s (tolerance %.0e: %s)' % (
            label, worst, n, ' (%s)' % where if where else '', tolerance, 'OK' if worst <= tolerance else 'DIFFERS: do not read the numbers that depend on it'))
    else:
        print('Check, %s: no reference run found' % label)
    return dict(worst=worst, comparisons=n, where=where, ok=bool(worst <= tolerance)) if n else None


def integrity(root, data):
    """What every run stored about its own event, its masks and its selection."""
    bad, runs, twins = [], 0, 0
    for f in sorted((Path(root) / 'results_V').glob('hnd-*/*/seed*/e10_info.json')):
        info = json.loads(f.read_text())
        if not (f.parent / 'completed.json').exists() or json.loads((f.parent / 'completed.json').read_text()).get('bounded_smoke') or not info.get('cold_arms'):
            continue
        runs += 1
        name = f.parent.parent.parent.name
        if (info['event'], info['released'], info['history']) != (EVENT, RELEASED, 64):
            bad.append('%s: event %s' % (name, (info['event'], info['released'], info['history'])))
        for start in ('mean', 'prof0'):
            r = info.get('restart:s_%s_r' % start, {})
            if (r.get('absorbed_before'), r.get('forecast_calls_before'), r.get('calibrators')) != (RELEASED, EVENT, info['new_sensors']):
                bad.append('%s: restart of s_%s_r %s' % (name, start, r))
            e = info.get('event:' + start, {})
            if (e.get('fit_windows'), e.get('trained_sensors')) != (EVENT, info['new_sensors']):
                bad.append('%s: refit %s' % (name, start))
        for arm, c in info['mask_counts'].items():
            if arm.startswith(('c64_', 's_')) and c['new_before_arrival'] != 0:
                bad.append('%s: %s read %d targets of new sensors from before the arrival' % (name, arm, c['new_before_arrival']))
    for k, by_seed in data.items():                                              # a scheduled arm chooses what its static twin chooses (selection ends at the event)
        for seed, rows in by_seed.items():
            for arm in SCHEDULED:
                if (arm, 'test', 'all') in rows:
                    twin = rows[('c64_' + arm.split('_')[1], 'test', 'all')]
                    twins += 1
                    if (rows[(arm, 'test', 'all')]['ridge'], rows[(arm, 'test', 'all')]['forget']) != (twin['ridge'], twin['forget']):
                        bad.append('%s/%d seed %d: %s chose other calibrator settings than its static twin' % (k[0], k[1], seed, arm))
    print('Check, stored by the runs (%d runs): event at stream window %d after %d absorbed windows, restarts, masks, and %d choices of calibrator settings against the static twin: %s' % (
        runs, EVENT, RELEASED, twins, 'OK' if not bad else 'PROBLEMS: ' + '; '.join(bad[:12])))
    return dict(runs=runs, problems=bad)


def view(data, keys, label, kind='calibrated', **kw):
    ok, missing = G38.complete(data, keys, STATIC + SCHEDULED, **kw)
    print('\n### %s: %d of %d dataset-years%s' % (label, len(ok), len(keys), ('; missing ' + ', '.join(missing)) if missing else ''))
    if not ok:
        return dict(n=0, missing=missing)
    items = G38.ratio_items(data, ok, PAIRS, kind, **kw)
    table = {}
    for name in PAIRS:
        table[name] = A.geometric(items, name)
        G38.line(name, table[name], items)
    return dict(n=len(ok), missing=missing, **table)


def blocks(root, keys):
    """Error per block of 256 stream windows (all sensors, all horizons, with the calibrator) relative to the static borrowed identity."""
    arms = SCHEDULED + ('c64_same', 'fit_emb', 'c64_mean')
    series, ends = collections.defaultdict(list), None
    for f in sorted((Path(root) / 'results_V').glob('hnd-*/*/seed*/e3_sums.npz')):
        setup = json.loads((f.parent / 'e3_setup.json').read_text())
        if (setup['dataset'], setup['year']) not in keys or setup.get('max_windows'):
            continue
        z = np.load(f)
        names = [str(v) for v in z['arm_names']]
        if not set(arms + ('c64_prof0',)) <= set(names):
            continue
        mae = z['block_abs_cnt'][:, 0] / np.maximum(z['block_abs_cnt'][:, 1], 1.)
        if ends is None or len(z['block_end']) < len(ends):
            ends = [int(v) for v in z['block_end']]
        for a in arms:
            series[a].append(np.log(mae[:, names.index(a)] / mae[:, names.index('c64_prof0')]))
    if not series:
        return None
    n = len(ends)
    print('\n| arm / c64_prof0, by block of 256 stream windows (block ends) | ' + ' | '.join(str(e) for e in ends) + ' |\n|---|' + '---:|' * n)
    out = dict(block_end=ends, runs=len(series[arms[0]]))
    for a in arms:
        gm = np.exp(np.mean([s[:n] for s in series[a]], 0))
        out[a] = [float(v) for v in gm]
        print('| %s | %s |' % (a, ' | '.join('%.3f' % v for v in gm)))
    print('All sensors, all horizons, with the calibrator; geometric mean over %d runs. The event is at stream window %d; the test portion starts where the validation portion ends.' % (out['runs'], EVENT))
    return out


def fits(root, keys):
    rows = collections.defaultdict(list)
    for f in sorted((Path(root) / 'results_V').glob('hnd-*/*/seed*/e10_info.json')):
        setup = json.loads((f.parent / 'e3_setup.json').read_text())
        if (setup['dataset'], setup['year']) not in keys or setup.get('max_windows'):
            continue
        for tag, rec in json.loads(f.read_text()).items():
            if isinstance(rec, dict) and 'holdout_mae' in rec:
                rows[tag].append((min(rec['holdout_mae']) / rec['holdout_mae'][0], rec['best_epoch'], rec['seconds'], rec.get('mean_abs_change', float('nan'))))
    if not rows:
        return None
    print('\n| fit | runs | holdout after / before (median) | kept the start | best epoch (median) | mean absolute change of the trained rows (median) | seconds (median, max) |\n|---|---:|---:|---:|---:|---:|---:|')
    out = {}
    for tag in sorted(rows):
        r = np.asarray(rows[tag], float)
        out[tag] = dict(runs=len(r), holdout=float(np.median(r[:, 0])), kept_start=int((r[:, 1] == 0).sum()), best_epoch=float(np.median(r[:, 1])),
                        seconds=float(np.median(r[:, 2])), seconds_max=float(r[:, 2].max()))
        print('| %s | %d | %.3f | %d | %d | %.4f | %.0f, %.0f |' % (tag, len(r), np.median(r[:, 0]), (r[:, 1] == 0).sum(), np.median(r[:, 1]), np.nanmedian(r[:, 3]), np.median(r[:, 2]), r[:, 2].max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--ref', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    result = dict(method_variant=METHOD)
    data = G33.load(Path(a.root) / 'results_V')
    print('# g40: borrow at arrival, refit at stream window 1,024 (the hand-over)')
    print('Endpoint: H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95%% CI by the g29 bootstrap over districts. Method arm: s_prof0_%s (calibrators of the new sensors %s at the event).' % (
        METHOD, 'restarted' if METHOD == 'r' else 'kept'))
    print('Runs found: %d dataset-years, %d with all three seeds.' % (len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if a.ref:
        ref = load_reference(a.ref)
        result['check_exact'] = agreement('same, stale5o, c64_same against g38', data, ref, *EXACT)
        result['check_refit'] = agreement('fit_emb, konly, c64_mean, c64_prof0 against g38 (they depend on refits)', data, ref, *REFIT)
    result['check_stored'] = integrity(a.root, data)
    new = result['new_test'] = view(data, PRIMARY, 'Primary, new sensors, test portion (entirely after the event), with the calibrator', cohort='new')
    result['new_val'] = view(data, PRIMARY, 'Primary, new sensors, validation portion (its first 1,024 windows are before the event), with the calibrator', cohort='new', portion='val')
    result['new_test_uncalibrated'] = view(data, PRIMARY, 'Primary, new sensors, test portion, without the calibrator', 'uncalibrated', cohort='new')
    old = result['old_test'] = view(data, PRIMARY, 'Primary, known sensors, test portion, with the calibrator', cohort='old')
    result['all_test'] = view(data, PRIMARY, 'Primary, all sensors, test portion, with the calibrator')
    result['secondary_new_test'] = view(data, SECONDARY, 'Secondary (PEMS05 2020 / 2025, TFNSW 2020), new sensors, test portion, with the calibrator', cohort='new')
    result['development_new_test'] = view(data, DEVELOPMENT, 'Development (TFNSW 2025; never part of a decision), new sensors, test portion, with the calibrator', cohort='new')
    print('\n### Time course (primary runs)')
    result['blocks'] = blocks(a.root, PRIMARY)
    print('\n### The fits (primary runs)')
    result['fits'] = fits(a.root, PRIMARY)
    print('\n### Rules fixed in PRESPEC.md (primary set, test portion; method arm %s)' % S)
    if not new.get('n') or new['missing'] or not old.get('n') or old['missing']:
        print('- not evaluated: incomplete (%s)' % ', '.join(new['missing'] or old['missing']))
    else:
        g = lambda t, n: t[n]['gm']
        gap = g(new, 'c64_prof0 / same') - g(new, 'fit_emb / same')
        recovery = (g(new, 'c64_prof0 / same') - g(new, S + ' / same')) / gap if gap > 0 else None
        r = dict(R1=new['%s / c64_prof0' % S]['hi'] < 1, R2=bool(recovery is not None and recovery >= .5), R3=new[S + ' / same']['hi'] <= A.MARGIN,
                 R4=new['%s / %s' % (S, M)]['hi'] < 1, H2=old['%s / konly' % S]['hi'] <= 1.01)
        f = lambda t, n: '%.3f, CI %.3f-%.3f' % (t[n]['gm'], t[n]['lo'], t[n]['hi'])
        v = lambda k: 'holds' if r[k] else 'does not hold'
        print('- R1, the refit improves a borrowed identity (new sensors, %s / c64_prof0 CI below 1): **%s** (%s)' % (S, v('R1'), f(new, '%s / c64_prof0' % S)))
        print('- R2, by at least half of what a full week before the stream gives (new sensors): **%s** (%s; c64_prof0 %.3f, %s %.3f, fit_emb %.3f of the same-year model)' % (
            v('R2'), 'recovery %.0f%%' % (100 * recovery) if recovery is not None else 'not evaluable: the full week is not better than the borrowed identity',
            g(new, 'c64_prof0 / same'), S, g(new, S + ' / same'), g(new, 'fit_emb / same')))
        print('- R3, to within %.2f of the same-year model (new sensors, %s / same CI): **%s** (%s)' % (A.MARGIN, S, v('R3'), f(new, S + ' / same')))
        print('- R4, borrowing is still worth it when a refit follows (new sensors, %s / %s CI below 1): **%s** (%s)' % (S, M, v('R4'), f(new, '%s / %s' % (S, M))))
        print('- H2, after the refit the arrivals no longer harm the mature sensors (known sensors, %s / konly CI at most 1.01): **%s** (%s)' % (S, v('H2'), f(old, '%s / konly' % S)))
        print('- "borrow at arrival, refit after 1,024 windows" as a working hand-over (needs R1, R2, R3): **%s**' % ('holds' if r['R1'] and r['R2'] and r['R3'] else 'does not hold'))
        print('- the other calibrator variant, without a rule: s_prof0_%s / c64_prof0 %s; %s' % (OTHER, f(new, 's_prof0_%s / c64_prof0' % OTHER), 's_prof0_%s / s_prof0_%s %s' % (
            METHOD, OTHER, f(new, 's_prof0_%s / s_prof0_%s' % (METHOD, OTHER)))))
        result['rules'] = dict({k: bool(x) for k, x in r.items()}, recovery=recovery)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
