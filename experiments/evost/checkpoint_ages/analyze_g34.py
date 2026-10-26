"""Pre-specified summary of g34 (PRESPEC.md), read-only.

  python analyze_g34.py RESULT_ROOT --g33 ROOT [ROOT ...] [--g30 ROOT ...] [--out summary.json]

RESULT_ROOT holds results_V/<task id>/<dataset>_<year>/seed<seed>/e3_metrics.jsonl; the task id starts with the family:
  idr  g33's identity-refresh arms on PEMS04 and PEMS07               (rules of g33 on eight districts)
  age  checkpoint ages of 5, 10 and 15 years                           (rules G1, G2 per age)
  cmb  borrow + refit + online updating                                (rules K1-K6)
  rra  the relational residual branch on top of the full method        (rules of g32)
Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap
over districts (10,000 draws, seed 2029).
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A
import analyze_g32 as G32
import analyze_g33 as G33

SIX = ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
EIGHT = ('PEMS03', 'PEMS04', 'PEMS06', 'PEMS07', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
PRIMARY = [(d, y) for d in SIX for y in (2020, 2025)]
PRIMARY8 = [(d, y) for d in EIGHT for y in (2020, 2025)]
BIG = [(d, y) for d in ('PEMS04', 'PEMS07') for y in (2020, 2025)]
SECONDARY = [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2020)]
DEVELOPMENT = [('TFNSW', 2025)]
AGE_YEARS = {5: (2015, 2020, 2025), 10: (2020, 2025), 15: (2025,)}
AGE_SECONDARY = {5: [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2015), ('TFNSW', 2020)], 10: [('PEMS05', 2025), ('TFNSW', 2020)], 15: []}
CMB_ARMS = ('stale5o', 'new_prof', 'fit_emb', 'fit_prof', 'on_stale', 'on_prof', 'on_emb', 'on_fit', 'on_same')
CMB_SENSITIVITY = collections.OrderedDict([                           # the two other online learning rates of the development sweep; no rule
    ('on_fit_lo / fit_prof', ('on_fit_lo', 'fit_prof')), ('on_fit_hi / fit_prof', ('on_fit_hi', 'fit_prof')),
    ('on_same_lo / same', ('on_same_lo', 'same')), ('on_same_hi / same', ('on_same_hi', 'same')),
    ('on_fit_lo / on_same_lo', ('on_fit_lo', 'on_same_lo')), ('on_fit_hi / on_same_hi', ('on_fit_hi', 'on_same_hi')),
    ('on_fit_lo / same', ('on_fit_lo', 'same')), ('on_fit_hi / same', ('on_fit_hi', 'same'))])
CMB_PAIRS = collections.OrderedDict([
    ('K1: fit_prof / fit_emb', ('fit_prof', 'fit_emb')), ('K2: on_fit / fit_prof', ('on_fit', 'fit_prof')), ('K3: on_prof / stale5o', ('on_prof', 'stale5o')),
    ('K4: on_prof / on_stale', ('on_prof', 'on_stale')), ('K6: on_fit / on_same', ('on_fit', 'on_same')), ('on_emb / on_same', ('on_emb', 'on_same')),
    ('fit_emb / on_same', ('fit_emb', 'on_same')), ('on_emb / fit_emb', ('on_emb', 'fit_emb')), ('on_fit / on_emb', ('on_fit', 'on_emb')),
    ('on_fit / fit_emb', ('on_fit', 'fit_emb')), ('on_fit / on_prof', ('on_fit', 'on_prof')), ('on_stale / stale5o', ('on_stale', 'stale5o')),
    ('on_prof / new_prof', ('on_prof', 'new_prof')), ('fit_emb / stale5o', ('fit_emb', 'stale5o'))])
FIT_TOLERANCE, EXACT_TOLERANCE = 2e-3, 1e-5


def load(root, family):
    data = collections.defaultdict(dict)
    for f in sorted(Path(root).rglob('e3_metrics.jsonl')):
        if ('/%s-' % family) not in str(f) or not (f.parent / 'completed.json').exists():
            continue
        if json.loads((f.parent / 'completed.json').read_text()).get('bounded_smoke'):
            continue
        rows = [json.loads(line) for line in f.open()]
        key, seed = (rows[0]['dataset'], int(rows[0]['year'])), int(rows[0]['seed'])
        assert seed not in data[key], ('two runs of one dataset-year and seed', family, key, seed)
        data[key][seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
    return data


def items_of(data, keys, pairs, kind='calibrated', cohort='all', portion='test'):
    items, missing = [], []
    for k in keys:
        if k not in data or sorted(data[k]) != list(A.SEEDS):
            missing.append('%s/%d' % k); continue
        v = lambda arm: A.value(data[k], arm, kind, A.METRIC, portion, cohort)
        items.append((k, {name: (None if None in (v(a), v(b)) else v(a) / v(b)) for name, (a, b) in pairs.items()}))
    return items, missing


def show(label, items, missing, expected, names, fmt='%.3f'):
    print('\n### %s: %d of %d dataset-years%s' % (label, len(items), expected, ('; missing ' + ', '.join(missing)) if missing else ''))
    out = {}
    for name in names:
        part = [(k, r) for k, r in items if r.get(name) is not None]
        if part:
            s = out[name] = A.geometric(part, name)
            line = '- %s: **{0}** (95%% CI {0}-{0}; range {0}-{0}; below 1 in %d of %d dataset-years, %d districts)'.format(fmt)
            print(line % (name, s['gm'], s['lo'], s['hi'], s['min'], s['max'], sum(r[name] < 1 for _, r in part), s['n'], s['districts']))
    return out


def agreement(label, data, ref, arms, tolerance):
    """Largest relative difference of test H12 MAE between two sets of runs; arms = {arm here: arm there}."""
    worst, n = 0., 0
    for k, runs in data.items():
        for seed, rows in runs.items():
            for mine, theirs in arms.items():
                if k in ref and seed in ref[k] and (theirs, 'test', 'all') in ref[k][seed] and (mine, 'test', 'all') in rows:
                    for kind in ('calibrated', 'uncalibrated'):
                        x, y = rows[(mine, 'test', 'all')][kind]['H12_MAE'], ref[k][seed][(theirs, 'test', 'all')][kind]['H12_MAE']
                        worst, n = max(worst, abs(x - y) / y), n + 1
    if n:
        print('Check, %s: max relative difference %.2e over %d comparisons (tolerance %.0e: %s)' % (
            label, worst, n, tolerance, 'OK' if worst <= tolerance else 'DIFFERS: do not read the numbers that depend on it'))
    return dict(worst=worst, comparisons=n, ok=bool(worst <= tolerance)) if n else None


def recovery(summary, stale, arm):
    gap, got = summary[stale + ' / same']['gm'], summary[arm + ' / same']['gm']
    return (gap - got) / (gap - 1.)


def verdict(holds):
    return 'holds' if holds else 'does not hold'


# ---------------------------------------------------------------- family idr: the rules of g33 on eight districts
def family_idr(root, g33, g30, result):
    new = load(root, 'idr')
    print('\n## A. g33 arms on PEMS04 and PEMS07 (family idr): %d dataset-years, %d with all three seeds' % (len(new), sum(sorted(v) == list(A.SEEDS) for v in new.values())))
    if not new:
        return
    if g30:
        result['check_idr_g30'] = agreement('same, stale5o, no-backbone arms against g30', new, g30, {a: a for a in ('same', 'stale5o', 'nobb_zero', 'nobb_last')}, EXACT_TOLERANCE)
    data = dict(g33)
    for k, v in new.items():
        assert k not in data, ('dataset-year present in g33 and here', k)
        data[k] = v
    out = result['idr'] = {}
    for label, keys in (('PEMS04 and PEMS07 only (2 districts; no rule)', BIG), ('Eight districts x 2020, 2025', PRIMARY8)):
        for kind in ('calibrated', 'uncalibrated'):
            items, miss = G33.items_of(data, keys, kind)
            out[(label[:5] + kind)] = G33.show('%s, %s' % (label, 'with the calibrator' if kind == 'calibrated' else 'without'), items, miss, len(keys))
    p8, miss = G33.items_of(data, PRIMARY8)
    new8, _ = G33.items_of(data, PRIMARY8, 'calibrated', 'new')
    print('\n### Rules of g33 re-evaluated on eight districts')
    if miss:
        print('- not evaluated: incomplete (%s)' % ', '.join(miss)); return
    p = {n: A.geometric([(k, r) for k, r in p8 if r.get(n) is not None], n) for n in p8[0][1] if any(r.get(n) is not None for _, r in p8)}
    q = {n: A.geometric([(k, r) for k, r in new8 if r.get(n) is not None], n) for n in ('new_prof / stale5o',)}
    rho_f, rho_n = recovery(p, 'stale5o', 'fit_emb'), recovery(p, 'stale5o', 'new_prof')
    rules = dict(F1=p['fit_emb / stale5o']['hi'] < 1 and rho_f >= .5, F2=p['fit_emb / same']['hi'] <= A.MARGIN,
                 N1=p['new_prof / stale5o']['hi'] < 1 and rho_n >= 1 / 3 and p['new_prof / same']['gm'] < p['stale5o_mean / same']['gm'],
                 N2=q['new_prof / stale5o']['hi'] < 1, B1=p['same_prof / same_geo']['hi'] < 1 and p['don_prof / don_geo']['hi'] < 1)
    c = lambda n, s=p: '%.3f, CI %.3f-%.3f' % (s[n]['gm'], s[n]['lo'], s[n]['hi'])
    print('- F1 (fit_emb / stale5o CI below 1, recovery at least 1/2): **%s** (%s; recovery %.0f%%)' % (verdict(rules['F1']), c('fit_emb / stale5o'), 100 * rho_f))
    print('- F2 (fit_emb / same CI at most %.2f): **%s** (%s)' % (A.MARGIN, verdict(rules['F2']), c('fit_emb / same')))
    print('- N1 (new_prof / stale5o CI below 1, recovery at least 1/3, better than the mean embedding): **%s** (%s; recovery %.0f%%)' % (verdict(rules['N1']), c('new_prof / stale5o'), 100 * rho_n))
    print('- N2 (unknown sensors, new_prof / stale5o CI below 1): **%s** (%s)' % (verdict(rules['N2']), c('new_prof / stale5o', q)))
    print('- B1 (same_prof / same_geo and don_prof / don_geo, both CIs below 1): **%s** (%s; %s)' % (verdict(rules['B1']), c('same_prof / same_geo'), c('don_prof / don_geo')))
    result['idr_rules'] = dict({k: bool(v) for k, v in rules.items()}, recovery_fit_emb=rho_f, recovery_new_prof=rho_n)


# ---------------------------------------------------------------- family age
def age_pairs(k):
    pairs = collections.OrderedDict([('stale%do / same' % k, ('stale%do' % k, 'same')), ('fit%d / same' % k, ('fit%d' % k, 'same')),
                                     ('fit%d / stale%do' % (k, k), ('fit%d' % k, 'stale%do' % k)), ('mean%d / same_mean' % k, ('mean%d' % k, 'same_mean'))])
    if k > 5:
        pairs['fit%d / fit5' % k] = ('fit%d' % k, 'fit5')
        pairs['stale%do / stale5o' % k] = ('stale%do' % k, 'stale5o')
    return pairs


def family_age(root, g33, g30, result):
    data = load(root, 'age')
    print('\n## B. Checkpoint ages of 5, 10 and 15 years (family age): %d dataset-years, %d with all three seeds' % (len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if not data:
        return
    if g30:
        result['check_age_g30'] = agreement('same and stale<k>o against g30', data, g30, {a: a for a in ('same', 'stale5o', 'stale10o', 'stale15o')}, EXACT_TOLERANCE)
    result['check_age_g33_exact'] = agreement('mean5 and same_mean against g33', data, g33, dict(mean5='stale5o_mean', same_mean='same_mean'), EXACT_TOLERANCE)
    result['check_age_g33_fit'] = agreement('fit5 against fit_emb of g33 (a refit)', data, g33, dict(fit5='fit_emb'), FIT_TOLERANCE)
    out, rules = result.setdefault('age', {}), result.setdefault('age_rules', {})
    for k, years in AGE_YEARS.items():
        keys, pairs = [(d, y) for d in SIX for y in years], age_pairs(k)
        items, miss = items_of(data, keys, pairs)
        out['age%d' % k] = s = show('Age %d years, six districts x %s, with the calibrator' % (k, ', '.join(map(str, years))), items, miss, len(keys), pairs)
        raw, _ = items_of(data, keys, pairs, 'uncalibrated')
        out['age%d_uncalibrated' % k] = show('Age %d years, without the calibrator' % k, raw, miss, len(keys), pairs)
        if k == 5:
            for years2 in ((2015,), (2020, 2025), (2025,)):
                part, m2 = items_of(data, [(d, y) for d in SIX for y in years2], pairs)
                out['age5_%s' % '_'.join(map(str, years2))] = show('Age 5 years, target year(s) %s only' % ', '.join(map(str, years2)), part, m2, 6 * len(years2), pairs)
        if k == 10:
            part, m2 = items_of(data, [(d, 2025) for d in SIX], pairs)
            out['age10_2025'] = show('Age 10 years, target year 2025 only', part, m2, 6, pairs)
        sec, miss_s = items_of(data, AGE_SECONDARY[k], pairs)
        if AGE_SECONDARY[k]:
            out['age%d_secondary' % k] = show('Age %d years, secondary (PEMS05, TFNSW)' % k, sec, miss_s, len(AGE_SECONDARY[k]), pairs)
        if miss:
            print('- rules G1(%d), G2(%d) not evaluated: incomplete' % (k, k)); continue
        f, st = 'fit%d' % k, 'stale%do' % k
        rho = recovery(s, st, f)
        g1 = s['%s / %s' % (f, st)]['hi'] < 1 and rho >= .5
        g2 = s[f + ' / same']['hi'] <= A.MARGIN
        print('- **G1(%d)** refitting only the identities repairs most of the damage (CI of %s / %s below 1, recovery at least 1/2): **%s** (%.3f, CI %.3f-%.3f; recovery %.0f%%)' % (
            k, f, st, verdict(g1), s['%s / %s' % (f, st)]['gm'], s['%s / %s' % (f, st)]['lo'], s['%s / %s' % (f, st)]['hi'], 100 * rho))
        print('- **G2(%d)** ... to within the g29 margin of the same-year model (CI of %s / same at most %.2f): **%s** (%.3f, CI %.3f-%.3f)' % (
            k, f, A.MARGIN, verdict(g2), s[f + ' / same']['gm'], s[f + ' / same']['lo'], s[f + ' / same']['hi']))
        rules['G1_%d' % k], rules['G2_%d' % k], rules['recovery_%d' % k] = bool(g1), bool(g2), rho
    dev, miss_d = items_of(data, DEVELOPMENT, age_pairs(15))
    show('Development (TFNSW 2025; never part of a decision), age 15', dev, miss_d, 1, age_pairs(15))
    fits = collections.defaultdict(list)
    for f in sorted(Path(root).rglob('e10_info.json')):
        if '/age-' in str(f):
            info = json.loads(f.read_text())
            for tag, v in info.items():
                if tag.startswith('fit') and isinstance(v, dict) and 'holdout_mae' in v:
                    fits[tag].append((min(v['holdout_mae']) / v['holdout_mae'][0], v['best_epoch'], v['seconds']))
    for tag in sorted(fits, key=lambda t: int(t[3:] or 5)):
        r = np.asarray(fits[tag])
        print('- %s: holdout MAE inside the fit windows after / before, median %.2f (range %.2f-%.2f); best epoch median %d; seconds median %.0f, max %.0f (%d runs)' % (
            tag, np.median(r[:, 0]), r[:, 0].min(), r[:, 0].max(), np.median(r[:, 1]), np.median(r[:, 2]), r[:, 2].max(), len(r)))


# ---------------------------------------------------------------- family cmb
def family_cmb(root, g33, result):
    data = load(root, 'cmb')
    print('\n## C. Borrow + refit + online updating (family cmb): %d dataset-years, %d with all three seeds' % (len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if not data:
        return data
    result['check_cmb_g33_exact'] = agreement('same, stale5o, new_prof against g33', data, g33, {a: a for a in ('same', 'stale5o', 'new_prof')}, EXACT_TOLERANCE)
    result['check_cmb_g33_fit'] = agreement('fit_emb against g33 (a refit)', data, g33, dict(fit_emb='fit_emb'), FIT_TOLERANCE)
    pairs = collections.OrderedDict([(a + ' / same', (a, 'same')) for a in CMB_ARMS])
    pairs.update(CMB_PAIRS)
    pairs.update(CMB_SENSITIVITY)
    out = result['cmb'] = {}
    prim, miss = items_of(data, PRIMARY, pairs)
    out['primary'] = p = show('Primary (six districts x 2020, 2025), with the calibrator', prim, miss, len(PRIMARY), pairs)
    raw, _ = items_of(data, PRIMARY, pairs, 'uncalibrated')
    out['primary_uncalibrated'] = show('Primary, without the calibrator', raw, miss, len(PRIMARY), pairs)
    val, _ = items_of(data, PRIMARY, pairs, 'calibrated', 'all', 'val')
    out['primary_validation_portion'] = show('Primary, validation portion (the first half of the stream; contains the selection windows), with the calibrator', val, miss, len(PRIMARY), pairs)
    for cohort, text in (('old', 'sensors the old checkpoint knows'), ('new', 'sensors the old checkpoint does not know')):
        part, m2 = items_of(data, PRIMARY, pairs, 'calibrated', cohort)
        out['primary_' + cohort] = show('Primary, %s, with the calibrator' % text, part, m2, len(PRIMARY), pairs)
    for label, keys in (('PEMS04 and PEMS07 (no rule)', BIG), ('Eight districts (no rule)', PRIMARY8), ('Secondary (PEMS05 2020 / 2025, TFNSW 2020)', SECONDARY),
                        ('Development (TFNSW 2025; never part of a decision)', DEVELOPMENT)):
        part, m2 = items_of(data, keys, pairs)
        if part:
            out[label.split(' (')[0].lower().replace(' ', '_')] = show(label + ', with the calibrator', part, m2, len(keys), pairs)
    if prim:
        print('\n| dataset/year | same | stale5o | new_prof | fit_emb | fit_prof | on_stale | on_prof | on_emb | on_fit | on_same | on_fit / same | on_fit / on_same |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for keys in (PRIMARY, BIG, SECONDARY, DEVELOPMENT):
            for (d, y), r in items_of(data, keys, pairs)[0]:
                v = lambda arm: A.value(data[(d, y)], arm, 'calibrated')
                print('| %s/%d | %.2f | %s | %.3f | %.3f |' % (d, y, v('same'), ' | '.join('%.2f' % v(a) for a in CMB_ARMS), r['on_fit / same'], r['K6: on_fit / on_same']))
    print('\n### Rules fixed in PRESPEC.md (primary set)')
    if miss:
        print('- not evaluated: the primary set is incomplete'); return data
    c = lambda n: '%.4f, CI %.4f-%.4f' % (p[n]['gm'], p[n]['lo'], p[n]['hi'])
    rho = recovery(p, 'stale5o', 'on_prof')
    rules = dict(K1=p['K1: fit_prof / fit_emb']['hi'] < 1, K2=p['K2: on_fit / fit_prof']['hi'] < 1, K3=p['K3: on_prof / stale5o']['hi'] < 1 and rho >= .5,
                 K4=p['K4: on_prof / on_stale']['hi'] < 1, K5=p['on_fit / same']['hi'] <= A.MARGIN, K6=p['K6: on_fit / on_same']['hi'] <= A.MARGIN)
    print('- K1, a borrowed start helps the refit (fit_prof / fit_emb CI below 1): **%s** (%s)' % (verdict(rules['K1']), c('K1: fit_prof / fit_emb')))
    print('- K2, online updating helps the refitted table (on_fit / fit_prof CI below 1): **%s** (%s)' % (verdict(rules['K2']), c('K2: on_fit / fit_prof')))
    print('- K3, online updating without an offline fit repairs most of the damage (on_prof / stale5o CI below 1, recovery at least 1/2): **%s** (%s; recovery %.0f%%)' % (
        verdict(rules['K3']), c('K3: on_prof / stale5o'), 100 * rho))
    print('- K4, a borrowed start helps online-only learning (on_prof / on_stale CI below 1): **%s** (%s)' % (verdict(rules['K4']), c('K4: on_prof / on_stale')))
    print('- K5, the full method is within the g29 margin of the same-year model (on_fit / same CI at most %.2f): **%s** (%s)' % (A.MARGIN, verdict(rules['K5']), c('on_fit / same')))
    print('- K6, ... and of the same-year model whose table is also updated online (on_fit / on_same CI at most %.2f): **%s** (%s)' % (A.MARGIN, verdict(rules['K6']), c('K6: on_fit / on_same')))
    print('- online updating of the same-year model itself, on_same / same: %s' % c('on_same / same'))
    keep = 'fit_emb' if not rules['K1'] else 'fit_prof'
    carried = ('on_emb' if keep == 'fit_emb' else 'on_fit') if (p['on_emb / fit_emb']['hi'] < 1 if keep == 'fit_emb' else rules['K2']) else keep
    print('- method carried forward by the rule "a component is kept only if its own comparison holds": **%s** (%s / same %s)' % (carried, carried, c(carried + ' / same')))
    result['cmb_rules'] = dict({k: bool(v) for k, v in rules.items()}, recovery_on_prof=rho, carried_forward=carried)
    lives = collections.defaultdict(list)
    for f in sorted(Path(root).rglob('e10_info.json')):
        if '/cmb-' in str(f):
            for name, v in json.loads(f.read_text()).get('online', {}).items():
                lives[name].append((v['seconds_learning'], v['steps'], v['trace'][-1]['mean_abs_change'] if v['trace'] else 0.))
    for name, rows in lives.items():
        r = np.asarray(rows)
        print('- online table of arm %s: learning seconds median %.0f (max %.0f), optimiser steps median %d, mean absolute change of the table at the end median %.4f (%d runs)' % (
            name, np.median(r[:, 0]), r[:, 0].max(), np.median(r[:, 1]), np.median(r[:, 2]), len(r)))
    return data


# ---------------------------------------------------------------- family rra
def family_rra(root, cmb, result):
    data = G32.load(root, 'ref')
    print('\n## D. Relational residual branch on top of the full method (family rra): %d dataset-years, %d with all three seeds' % (len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    if not data:
        return
    if cmb:
        result['check_rra_cmb'] = agreement('base arm against on_fit of family cmb (a refit and an online process)', data, cmb, dict(base='on_fit'), FIT_TOLERANCE)
    out = result['rra'] = {}
    prim, miss = G32.items_of(data, PRIMARY)
    out['primary'] = t = G32.show('Primary (six districts x 2020, 2025)', prim, miss, len(PRIMARY))
    for label, keys in (('Secondary (PEMS05 2020 / 2025, TFNSW 2020)', SECONDARY), ('Development (TFNSW 2025; never part of a decision)', DEVELOPMENT)):
        part, m2 = G32.items_of(data, keys)
        if part:
            out[label.split(' (')[0].lower()] = G32.show(label, part, m2, len(keys))
    for cohort in ('old', 'new'):
        part, _ = G32.items_of(data, PRIMARY, cohort)
        s = G32.summary(part, ('G = rra / base', 'S = rra / self'))
        if s:
            print('- sensors the old checkpoint %s (%d dataset-years): G %.4f (%.4f-%.4f), S %.4f (%.4f-%.4f)' % (
                'knows' if cohort == 'old' else 'does not know', len(part), s['G = rra / base']['gm'], s['G = rra / base']['lo'], s['G = rra / base']['hi'],
                s['S = rra / self']['gm'], s['S = rra / self']['lo'], s['S = rra / self']['hi']))
    print('\n### Rules of g32, fixed again in PRESPEC.md (primary set)')
    if miss:
        print('- not evaluated: the primary set is incomplete'); return
    holds = {name: (t[name]['hi'] < 1) for name in G32.RULES if name in t}
    for name in G32.RULES:
        if name in t:
            print('- %s: **%s** (%.4f, CI %.4f-%.4f)' % (name, verdict(holds[name]), t[name]['gm'], t[name]['lo'], t[name]['hi']))
    s, r = holds['S = rra / self'], holds['R = rra / rand']
    if s and r and t['S = rra / self']['gm'] <= .99:
        reading = 'a graph component can be claimed (S and R hold, S at least 1%)'
    elif s and r:
        reading = 'detectable, below 1%: S and R hold but the gain over the same-size adapter without neighbours is under 1%'
    elif holds['G = rra / base'] and not s:
        reading = 'no graph component: the gain over the local path is not larger than that of the same-size adapter without neighbours'
    elif s and not r:
        reading = 'other sensors help, but any set of them serves as context (random candidates do as well)'
    else:
        reading = 'no graph component: the branch does not improve the local path'
    print('- reading: **%s**' % reading)
    result['rra_rules'] = dict({k.split(' ')[0]: bool(v) for k, v in holds.items()}, reading=reading)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33', nargs='+', required=True, help='result root(s) of g33')
    ap.add_argument('--g30', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    g33 = collections.defaultdict(dict)
    for root in a.g33:
        for k, v in G33.load(root).items():
            g33[k].update(v)
    g30 = A.load(a.g30) if a.g30 else None
    result = {}
    print('# g34: identity refresh, continued (districts, checkpoint age, combined method with online updating, graph branch on top)')
    print('Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means, 95% CI by the g29 bootstrap over districts.')
    family_idr(a.root, g33, g30, result)
    family_age(a.root, g33, g30, result)
    cmb = family_cmb(a.root, g33, result)
    family_rra(a.root, cmb, result)
    if a.out:
        Path(a.out).write_text(json.dumps({str(k): v for k, v in result.items()}, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
