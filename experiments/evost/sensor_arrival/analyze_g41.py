"""Pre-specified summary of g41 (PRESPEC.md): four families of repairs. Read-only.

  python analyze_g41.py DELIVERED_ROOT --g33 G33_RESULTS --g40 G40_ROOT --g39 G39_ROOT [--out summary.json]

DELIVERED_ROOT/results_V holds xft-*, hnd-*, rls-*, rtr-evl-* runs; DELIVERED_ROOT/ckpt_V the rtr trainings.
--g33: g33's delivered results_V (checks of xft); --g40: g40's delivered root (checks of hnd); --g39: g39's delivered root
(the post-hoc recomputation of rtr). Endpoint: H12 MAE, mean of seeds 51-53; geometric means with the g29 district bootstrap.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A
import analyze_g33 as G33
import analyze_g38 as G38
import analyze_g39 as G39

SIX = ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
PRIMARY = [(d, y) for d in SIX for y in (2020, 2025)]
CONFIGS = ('plain', 'plain_jitter', 'guard', 'guard_jitter')


def load_family(root, prefix):
    data = collections.defaultdict(dict)
    for d in sorted((Path(root) / 'results_V').glob(prefix + '-*')):
        for k, v in G33.load(d).items():
            for seed, rows in v.items():
                assert seed not in data[k], ('two runs of one dataset-year and seed', k, seed)
                data[k][seed] = rows
    return data


def agreement(label, mine, ref, arms, tolerance, cohorts=('all', 'old', 'new'), kinds=('calibrated', 'uncalibrated')):
    worst, n, where = 0., 0, None
    for k, runs in mine.items():
        for seed, rows in runs.items():
            for (arm, portion, cohort), row in rows.items():
                if arm not in arms or cohort not in cohorts:
                    continue
                other = ref.get(k, {}).get(seed, {}).get((arms[arm], portion, cohort))
                if other is None:
                    continue
                for kind in kinds:
                    d = abs(row[kind]['H12_MAE'] - other[kind]['H12_MAE']) / other[kind]['H12_MAE']
                    n += 1
                    if d > worst:
                        worst, where = d, '%s/%d seed %d %s/%s/%s %s' % (k[0], k[1], seed, arm, portion, cohort, kind)
    ok = bool(n and worst <= tolerance)
    print('- check, %s: max relative difference %.2e over %d comparisons%s (tolerance %.0e: %s)' % (
        label, worst, n, ' (%s)' % where if where else '', tolerance, 'OK' if ok else ('no comparison' if not n else 'DIFFERS')))
    return dict(worst=worst, comparisons=n, where=where, ok=ok)


def view(data, label, pairs, arms, kind='calibrated', **kw):
    ok, missing = G38.complete(data, PRIMARY, arms, **kw)
    print('\n#### %s: %d of %d dataset-years%s' % (label, len(ok), len(PRIMARY), ('; missing ' + ', '.join(missing)) if missing else ''))
    if not ok:
        return dict(n=0, missing=missing)
    items = G38.ratio_items(data, ok, pairs, kind, **kw)
    table = {}
    for name in pairs:
        table[name] = A.geometric(items, name)
        G38.line(name, table[name], items)
    return dict(n=len(ok), missing=missing, **table)


def fmt(s):
    return '%.3f, CI %.3f-%.3f' % (s['gm'], s['lo'], s['hi'])


# ---------------------------------------------------------------- xft
def family_xft(root, g33):
    print('\n## Family xft: cross-fitted calibrator at the start of the year (item 4)')
    data = load_family(root, 'xft')
    out = dict(checks={})
    if g33:
        ref = G33.load(g33)
        out['checks']['g33_exact'] = agreement("same / stale5o against g33's", data, ref, dict(same='same', stale5o='stale5o'), 1e-5)
        out['checks']['g33_fit'] = agreement("fit_emb against g33's", data, ref, dict(fit_emb='fit_emb'), 2e-3)
    out['checks']['uncalibrated_twin'] = agreement('fit_emb_x against fit_emb without the calibrator', data,
                                                   {k: {s: {(('fit_emb',) + key[1:]): r for key, r in rows.items() if key[0] == 'fit_emb'} for s, rows in v.items()} for k, v in data.items()},
                                                   dict(fit_emb_x='fit_emb'), 1e-12, kinds=('uncalibrated',))
    arms = ('same', 'stale5o', 'fit_emb', 'fit_emb_x')
    pairs = collections.OrderedDict([('fit_emb_x / fit_emb', ('fit_emb_x', 'fit_emb')), ('fit_emb / same', ('fit_emb', 'same')),
                                     ('fit_emb_x / same', ('fit_emb_x', 'same')), ('stale5o / same', ('stale5o', 'same'))])
    cal = out['test'] = view(data, 'Primary, test portion, with the calibrator, all sensors', pairs, arms)
    out['val'] = view(data, 'Primary, validation portion, with the calibrator', pairs, arms, portion='val')
    out['new'] = view(data, 'Primary, sensors the old checkpoint does not know, test portion, with the calibrator', pairs, arms, cohort='new')
    out['old'] = view(data, 'Primary, sensors the old checkpoint knows, test portion, with the calibrator', pairs, arms, cohort='old')
    if cal.get('n') == len(PRIMARY):
        s = cal['fit_emb_x / fit_emb']
        x1 = bool(s['gm'] <= .995 and s['hi'] < 1)
        print('\n- **X1**, cross-fitting the calibrator\'s initial fit improves the identity refresh (G at most 0.995, CI below 1): **%s** (%s)' % (
            'holds' if x1 else 'does not hold', fmt(s)))
        out['rules'] = dict(X1=x1)
    else:
        print('\n- X1 not evaluated: incomplete')
    return out


# ---------------------------------------------------------------- hnd
def family_hnd(root, g40):
    print('\n## Family hnd: the hand-over at stream window 1,024 (items 5 and 4)')
    data = load_family(root, 'hnd')
    out = dict(checks={})
    if g40:
        ref = load_family(g40, 'hnd')
        out['checks']['g40_exact'] = agreement("same / stale5o against g40's", data, ref, dict(same='same', stale5o='stale5o'), 1e-5)
        out['checks']['g40_fit'] = agreement("fit_emb / konly / c64_prof0 / s_prof0_r against g40's", data, ref,
                                             dict(fit_emb='fit_emb', konly='konly', c64_prof0='c64_prof0', s_prof0_r='s_prof0_r'), 2e-3)
    twin = {k: {s: {(('s_prof0_r',) + key[1:]): r for key, r in rows.items() if key[0] == 's_prof0_r'} for s, rows in v.items()} for k, v in data.items()}
    out['checks']['x_uncalibrated'] = agreement('x_prof0_r against s_prof0_r without the calibrator', data, twin, dict(x_prof0_r='s_prof0_r'), 1e-12, kinds=('uncalibrated',))
    out['checks']['x_known'] = agreement('x_prof0_r against s_prof0_r on the known sensors, with the calibrator', data, twin, dict(x_prof0_r='s_prof0_r'), 1e-12,
                                         cohorts=('old',), kinds=('calibrated',))
    bad, runs = [], 0
    for f in sorted((Path(root) / 'results_V').glob('hnd-*/*/seed*/e10_info.json')):
        info = json.loads(f.read_text())
        if not info.get('cold_arms'):
            continue
        runs += 1
        for name, which in (('s_prof0_r', 'new'), ('a_prof0_r', 'all'), ('x_prof0_r', 'new')):
            r = info.get('restart:' + name, {})
            if (r.get('which'), r.get('absorbed_before'), r.get('forecast_calls_before')) != (which, info['event'] - info['tail'], info['event']):
                bad.append('%s: %s %s' % (f.parent.parent.parent.name, name, r))
        for arm, c in info['mask_counts'].items():
            if arm.startswith(('c64_', 's_', 'a_', 'x_')) and c['new_before_arrival'] != 0:
                bad.append('%s: %s read targets of new sensors from before the arrival' % (f.parent.parent.parent.name, arm))
    print('- check, stored by the runs (%d runs): restarts and masks: %s' % (runs, 'OK' if not bad else 'PROBLEMS: ' + '; '.join(bad[:8])))
    out['checks']['stored'] = dict(runs=runs, problems=bad)
    arms = ('same', 'stale5o', 'fit_emb', 'konly', 'c64_prof0', 's_prof0_r', 'a_prof0_r', 'x_prof0_r')
    pairs = collections.OrderedDict([('a_prof0_r / konly', ('a_prof0_r', 'konly')), ('s_prof0_r / konly', ('s_prof0_r', 'konly')),
                                     ('a_prof0_r / s_prof0_r', ('a_prof0_r', 's_prof0_r')), ('x_prof0_r / s_prof0_r', ('x_prof0_r', 's_prof0_r')),
                                     ('s_prof0_r / same', ('s_prof0_r', 'same')), ('a_prof0_r / same', ('a_prof0_r', 'same')), ('x_prof0_r / same', ('x_prof0_r', 'same')),
                                     ('s_prof0_r / c64_prof0', ('s_prof0_r', 'c64_prof0')), ('fit_emb / konly', ('fit_emb', 'konly')), ('fit_emb / same', ('fit_emb', 'same'))])
    new = out['new_test'] = view(data, 'Primary, new sensors, test portion (after the event), with the calibrator', pairs, arms, cohort='new')
    old = out['old_test'] = view(data, 'Primary, known sensors, test portion, with the calibrator', pairs, arms, cohort='old')
    out['all_test'] = view(data, 'Primary, all sensors, test portion, with the calibrator', pairs, arms)
    out['new_test_uncalibrated'] = view(data, 'Primary, new sensors, test portion, without the calibrator', pairs, arms, 'uncalibrated', cohort='new')
    out['old_test_uncalibrated'] = view(data, 'Primary, known sensors, test portion, without the calibrator', pairs, arms, 'uncalibrated', cohort='old')
    if new.get('n') == len(PRIMARY) and old.get('n') == len(PRIMARY):
        a1, a2 = old['a_prof0_r / konly']['hi'] <= 1.01, new['a_prof0_r / s_prof0_r']['hi'] <= 1.01
        s = new['x_prof0_r / s_prof0_r']
        x2 = bool(s['gm'] <= .995 and s['hi'] < 1)
        v = lambda b: 'holds' if b else 'does not hold'
        print('\n- **A1**, refitting every row at the hand-over removes the harm to mature sensors (known, a_prof0_r / konly, CI at most 1.01): **%s** (%s)' % (v(a1), fmt(old['a_prof0_r / konly'])))
        print('- **A2**, … and costs the new sensors nothing (new, a_prof0_r / s_prof0_r, CI at most 1.01): **%s** (%s)' % (v(a2), fmt(new['a_prof0_r / s_prof0_r'])))
        print('- "refit every row at the hand-over" (needs A1 and A2): **%s**' % v(a1 and a2))
        print('- **X2**, the cross-fitted restart improves the new sensors (x_prof0_r / s_prof0_r, G at most 0.995, CI below 1): **%s** (%s)' % (v(x2), fmt(s)))
        print('- for comparison, g40\'s H2 reproduced: known, s_prof0_r / konly %s' % fmt(old['s_prof0_r / konly']))
        out['rules'] = dict(A1=bool(a1), A2=bool(a2), X2=x2)
    else:
        print('\n- A1, A2, X2 not evaluated: incomplete')
    return out


# ---------------------------------------------------------------- rls
def block_mae(run_dir):
    z = np.load(run_dir / 'e3_sums.npz', allow_pickle=False)
    bac = z['block_abs_cnt']                                          # [blocks, 2, arms]
    return [str(a) for a in z['arm_names']], bac[:, 0, :] / np.maximum(bac[:, 1, :], 1.)


def test_rows(run_dir):
    rows = [json.loads(line) for line in (run_dir / 'e3_metrics.jsonl').open()]
    return {r['arm']: r for r in rows if r['portion'] == 'test' and r['cohort'] == 'all'}


def family_rls(root):
    print('\n## Family rls: a guard against the calibrator\'s wind-up (item 6)')
    out = dict(tasks={})
    for task in sorted((Path(root) / 'results_V').glob('rls-*')):
        runs = [p for p in task.glob('*/seed*') if (p / 'completed.json').exists()]
        if not runs:
            continue
        d = runs[0]
        done, info = json.loads((d / 'completed.json').read_text()), json.loads((d / 'rls_info.json').read_text())
        if done.get('bounded_smoke'):
            continue
        row = dict(dataset=info['dataset'], year=info['year'], seed=info['seed'], backbone=info['backbone'], configs=info['configs'])
        cfg = {c: d / c for c in done['configs']}
        mae = {c: block_mae(p) for c, p in cfg.items()}
        for kind in ('plain', 'guard'):
            if kind in mae and kind + '_jitter' in mae:
                arms, a = mae[kind]; arms2, b = mae[kind + '_jitter']
                assert arms == arms2
                row['amplification_' + kind] = float(np.max(np.abs(b - a) / np.maximum(a, 1e-12)))
        tests = {c: test_rows(p) for c, p in cfg.items()}
        row['test_H12'] = {c: {arm: r['calibrated']['H12_MAE'] for arm, r in t.items()} for c, t in tests.items()}
        row['chosen'] = {c: {arm: (r['ridge'], r['forget']) for arm, r in t.items()} for c, t in tests.items()}
        if 'plain' in tests and 'guard' in tests:
            row['guard_vs_plain'] = max(abs(tests['guard'][arm]['calibrated']['H12_MAE'] - r['calibrated']['H12_MAE']) / r['calibrated']['H12_MAE'] for arm, r in tests['plain'].items())
            row['same_choice'] = all(tests['guard'][arm]['ridge'] == r['ridge'] and tests['guard'][arm]['forget'] == r['forget'] for arm, r in tests['plain'].items())
        out['tasks'][task.name] = row
    tf = {k: v for k, v in out['tasks'].items() if v['dataset'] == 'TFNSW'}
    pe = {k: v for k, v in out['tasks'].items() if v['dataset'] != 'TFNSW'}
    print('\n| task | amplification without guard | with guard | times the guard acted (guard / guard_jitter) | largest trace ratio without guard | guard vs plain, test H12 |\n|---|---:|---:|---:|---:|---:|')
    for k, v in sorted(tf.items()):
        c = v['configs']
        print('| %s | %.2e | %.2e | %d / %d | %.3g | %.2e |' % (k, v.get('amplification_plain', float('nan')), v.get('amplification_guard', float('nan')),
                                                         c.get('guard', {}).get('capped', -1), c.get('guard_jitter', {}).get('capped', -1),
                                                         c.get('plain', {}).get('max_trace_ratio', float('nan')), v.get('guard_vs_plain', float('nan'))))
    print('\n| PeMS check | guard vs plain, test H12 (largest over arms) | same ridge / forgetting chosen | times the guard acted |\n|---|---:|---|---:|')
    for k, v in sorted(pe.items()):
        print('| %s | %.2e | %s | %d |' % (k, v.get('guard_vs_plain', float('nan')), v.get('same_choice'), v['configs'].get('guard', {}).get('capped', -1)))
    expected_tf, expected_pe = 18, 12
    if len(tf) == expected_tf and len(pe) == expected_pe:
        g0 = any(v['amplification_plain'] > 1e-3 for v in tf.values())
        g1 = all(v['amplification_guard'] <= 1e-4 for v in tf.values())
        g2 = all(v['guard_vs_plain'] <= 1e-6 and v['same_choice'] for v in pe.values())
        logs = [np.log(x['test_H12']['guard'][arm] / x['test_H12']['plain'][arm]) for x in tf.values() for arm in x['test_H12']['plain']]
        g3_value = float(np.exp(np.mean(logs)))
        g3 = g3_value <= 1.01
        v = lambda b: 'holds' if b else 'does not hold'
        print('\n- **G0** (precondition), without the guard amplification above 1e-3 in at least one TFNSW task: **%s** (largest %.2e)' % (v(g0), max(x['amplification_plain'] for x in tf.values())))
        print('- **G1**, with the guard amplification at most 1e-4 in every TFNSW task: **%s** (largest %.2e)' % (v(g1), max(x['amplification_guard'] for x in tf.values())))
        print('- **G2**, the guard leaves every PeMS check unchanged (1e-6, same settings): **%s** (largest %.2e)' % (v(g2), max(x['guard_vs_plain'] for x in pe.values())))
        print('- **G3**, the guard costs no accuracy on TFNSW (geometric mean of guard / plain over %d task-arms at most 1.01): **%s** (%.4f; range %.3f-%.3f)' % (
            len(logs), v(g3), g3_value, float(np.exp(min(logs))), float(np.exp(max(logs)))))
        print('- "the guard enters the method" (needs G1, G2 and G3): **%s**' % v(g1 and g2 and g3))
        out['rules'] = dict(G0=bool(g0), G1=bool(g1), G2=bool(g2), G3=bool(g3), G3_value=g3_value)
    else:
        print('\n- G0-G2 not evaluated: %d of %d TFNSW tasks, %d of %d PeMS checks' % (len(tf), expected_tf, len(pe), expected_pe))
    if tf:
        print('\nTFNSW, test H12 MAE with the calibrator, seed mean, guard (plain):')
        for (bb, year) in sorted({(v['backbone'], v['year']) for v in tf.values()}):
            rows = [v for v in tf.values() if (v['backbone'], v['year']) == (bb, year)]
            arms = sorted(rows[0]['test_H12'].get('guard', {}))
            cell = lambda arm: '%s %.3f (%.3f)' % (arm, np.mean([r['test_H12']['guard'][arm] for r in rows]), np.mean([r['test_H12']['plain'][arm] for r in rows]))
            print('- %s %d (%d seeds): %s' % (bb, year, len(rows), ', '.join(cell(a) for a in arms if all('guard' in r['test_H12'] and 'plain' in r['test_H12'] for r in rows))))
    return out


# ---------------------------------------------------------------- rtr
def family_rtr(root, g39):
    print('\n## Family rtr: g39\'s diverged model retrained (item 7, post hoc)')
    out = {}
    choice = Path(root) / 'state_V' / 'RTR_CHOICE_V.json'
    for f in sorted((Path(root) / 'ckpt_V').glob('rtr-trn-*/train_report.json')):
        r = next(iter(json.loads(f.read_text())['runs'].values()))
        out[f.parent.name] = dict(epochs=r['epochs_run'], best_epoch=r['best_epoch'], validation=r['lowest_validation_loss'], test_T12=r['metrics']['test_train_stats']['T12_MAE'])
        print('- %s: %d epochs (best %d), lowest validation loss %.3f, test H12 MAE %.3f' % (f.parent.name, r['epochs_run'], r['best_epoch'], r['lowest_validation_loss'],
                                                                                     r['metrics']['test_train_stats']['T12_MAE']))
    if choice.exists():
        c = json.loads(choice.read_text())
        out['choice'] = c
        print('- usable if at most %.3f (1.5 x %.4f): chosen %s' % (c['limit'], c['nt_validation'], c['chosen']))
    runs = load_family(root, 'rtr-evl')
    if not g39 or not runs:
        print('- post-hoc recomputation not possible: %s' % ('no g39 root' if not g39 else 'no evaluation run'))
        return out
    data = G33.load(Path(g39) / 'results_V')
    key = ('PEMS06', 2025)
    assert key in runs and 52 in runs[key]
    original = data[key][52]
    data[key][52] = runs[key][52]
    print('- g39\'s analysis recomputed with the retrained model\'s evaluation in place of evl-PEMS06-2025-s52 (post hoc; the pre-specified verdict of g39 stands as delivered):')
    cal = G39.view(data, 'g39 post hoc, primary, test portion, with the calibrator')
    gap_tm, gap_nt = cal['stale_tm / same_tm']['gm'] - 1., cal['stale_nt / same_nt']['gm'] - 1.
    out['posthoc'] = dict(stale_tm_over_same_tm=cal['stale_tm / same_tm'], fit_tm_over_same_tm=cal['fit_tm / same_tm'], fit_tm_over_stale_tm=cal['fit_tm / stale_tm'],
                          recovery_tm=cal['recovery_tm'], share=gap_tm / gap_nt if gap_nt > 0 else None, same_tm_over_same_nt=cal['same_tm / same_nt'])
    rows = lambda rs, arm: rs[(arm, 'test', 'all')]['calibrated']['H12_MAE']
    print('- PEMS06/2025 seed 52, test H12 MAE with the calibrator, same_tm: %.3f before, %.3f after (same_nt %.3f)' % (rows(original, 'same_tm'), rows(runs[key][52], 'same_tm'),
                                                                                                                 rows(runs[key][52], 'same_nt')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33')
    ap.add_argument('--g40')
    ap.add_argument('--g39')
    ap.add_argument('--out')
    a = ap.parse_args()
    print('# g41: five repairs from the experiment audit (PRESPEC.md)')
    print('Endpoint: test-portion H12 MAE, mean of seeds 51-53; geometric means, 95% CI by the g29 bootstrap over districts.')
    result = dict(xft=family_xft(a.root, a.g33), hnd=family_hnd(a.root, a.g40), rtr=family_rtr(a.root, a.g39))
    if any((Path(a.root) / 'results_V').glob('rls-*')):                 # the rls family moved to a follow-up package on 2026-10-17
        result['rls'] = family_rls(a.root)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
