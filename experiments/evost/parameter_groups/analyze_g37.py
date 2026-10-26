"""Pre-specified summary of g37 (PRESPEC.md): which part of a five-year-old frozen STAEFormer should be updated (read-only).

  python analyze_g37.py DELIVERED_ROOT [--g33 ROOT ...] [--out summary.json]

DELIVERED_ROOT holds results_V/mnt-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/{e3_metrics.jsonl, e10_info.json}.
Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap
over districts. Rules P1-P4 and C1 on six districts x 2020, 2025. --g33: result roots of g33 (same dataset-years and seeds)
for the check that `same`, `stale5o` and `fit_emb` reproduce the delivered runs.
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
ARMS = ('stale5o', 'fit_emb', 'fit_emb_sel', 'fit_all', 'fit_shared', 'fit_head', 'fit_same')
FITS = ('fit_emb_sel', 'fit_all', 'fit_shared', 'fit_head', 'fit_same')
PAIRS = collections.OrderedDict([
    ('fit_emb_sel / fit_all', ('fit_emb_sel', 'fit_all')), ('fit_emb_sel / fit_shared', ('fit_emb_sel', 'fit_shared')),
    ('fit_emb_sel / fit_head', ('fit_emb_sel', 'fit_head')), ('fit_emb_sel / fit_same', ('fit_emb_sel', 'fit_same')),
    ('fit_emb_sel / fit_emb', ('fit_emb_sel', 'fit_emb')), ('fit_emb_sel / stale5o', ('fit_emb_sel', 'stale5o')),
    ('fit_all / stale5o', ('fit_all', 'stale5o')), ('fit_shared / stale5o', ('fit_shared', 'stale5o')), ('fit_head / stale5o', ('fit_head', 'stale5o'))])
EXACT, REFIT = 1e-5, 2e-3


def items_of(data, keys, kind='calibrated', cohort='all'):
    items, missing = [], []
    for k in keys:
        if k not in data or sorted(data[k]) != list(A.SEEDS):
            missing.append('%s/%d' % k); continue
        v = lambda arm: A.value(data[k], arm, kind, cohort=cohort)
        same = v('same')
        if same is None or any(v(a) is None for a in ARMS):
            missing.append('%s/%d' % k); continue
        row = {arm + ' / same': v(arm) / same for arm in ARMS}
        for name, (a, b) in PAIRS.items():
            row[name] = v(a) / v(b)
        items.append((k, row))
    return items, missing


def show(label, items, missing, expected):
    print('\n### %s: %d of %d dataset-years%s' % (label, len(items), expected, ('; missing ' + ', '.join(missing)) if missing else ''))
    out = {}
    if not items:
        return out
    for name in [a + ' / same' for a in ARMS] + list(PAIRS):
        s = A.geometric(items, name)
        out[name] = s
        below = sum(1 for _, r in items if r[name] < 1)
        print('- %s: **%.3f** (95%% CI %.3f-%.3f; range %.3f-%.3f; below 1 in %d of %d dataset-years, %d districts)' % (
            name, s['gm'], s['lo'], s['hi'], s['min'], s['max'], below, s['n'], s['districts']))
    return out


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


def costs(root, keys, result):
    """Per arm: rate chosen, seconds per epoch, seconds of the three candidates, peak GPU memory, trained values (runs of the given dataset-years)."""
    rows = collections.defaultdict(list)
    ratio, chosen = [], collections.defaultdict(collections.Counter)
    for f in sorted((Path(root) / 'results_V').rglob('e10_info.json')):
        done = f.parent / 'completed.json'
        if not done.exists() or json.loads(done.read_text()).get('bounded_smoke'):
            continue
        setup = json.loads((f.parent / 'e3_setup.json').read_text())
        if (setup['dataset'], setup['year']) not in keys:
            continue
        info = json.loads(f.read_text())
        per_epoch = {}
        for name in FITS:
            r = info[name]
            spe = float(np.mean([c['seconds_per_epoch'] for c in r['candidates'].values() if c['seconds_per_epoch']]))
            per_epoch[name] = spe
            rows[name].append((spe, r['seconds'], r['peak_gpu_gib'] or 0., r['trained_values'], min(r['holdout_mae']) / r['holdout_mae'][0], r['best_epoch']))
            chosen[name]['%g' % r['selected_lr']] += 1
        ratio.append(per_epoch['fit_emb_sel'] / per_epoch['fit_all'])
    if not ratio:
        return None
    print('\n| arm | seconds per epoch (median) | seconds, three candidates (median, max) | peak GPU GiB (median, max) | trained values (median) | holdout after / before (median) | rate chosen (runs) |\n|---|---:|---:|---:|---:|---:|---|')
    table = {}
    for name in FITS:
        r = np.asarray(rows[name], float)
        table[name] = dict(seconds_per_epoch=float(np.median(r[:, 0])), seconds=float(np.median(r[:, 1])), seconds_max=float(r[:, 1].max()), peak_gpu_gib=float(np.median(r[:, 2])),
                           peak_gpu_gib_max=float(r[:, 2].max()), trained_values=float(np.median(r[:, 3])), holdout=float(np.median(r[:, 4])), chosen=dict(chosen[name]))
        print('| %s | %.1f | %.0f, %.0f | %.1f, %.1f | %s | %.3f | %s |' % (name, np.median(r[:, 0]), np.median(r[:, 1]), r[:, 1].max(), np.median(r[:, 2]), r[:, 2].max(),
                                                                 '{:,}'.format(int(np.median(r[:, 3]))), np.median(r[:, 4]),
                                                                 ', '.join('%s: %d' % (k, v) for k, v in sorted(chosen[name].items(), key=lambda kv: float(kv[0])))))
    med = float(np.median(ratio))
    print('- seconds per epoch, fit_emb_sel / fit_all, per run: median %.2f (range %.2f-%.2f; %d runs)' % (med, min(ratio), max(ratio), len(ratio)))
    result['costs'] = dict(arms=table, seconds_per_epoch_emb_over_all=dict(median=med, min=float(min(ratio)), max=float(max(ratio)), runs=len(ratio)))
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33', nargs='*', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    result = {}
    data = G33.load(Path(a.root) / 'results_V')
    print('# g37: which part of a five-year-old frozen STAEFormer should be updated (fair maintenance baselines)')
    print('Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53. Runs found: %d dataset-years, %d with all three seeds.' % (
        len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    ref = collections.defaultdict(dict)
    for root in a.g33:
        for k, v in G33.load(root).items():
            ref[k].update(v)
    if ref:
        result['check_g33_exact'] = agreement('same, stale5o against g33', data, ref, ('same', 'stale5o'), EXACT)
        result['check_g33_fit'] = agreement('fit_emb against g33 (a refit)', data, ref, ('fit_emb',), REFIT)
    prim, miss = items_of(data, PRIMARY)
    result['primary'] = p = show('Primary (six districts x 2020, 2025), with the calibrator', prim, miss, len(PRIMARY))
    raw, _ = items_of(data, PRIMARY, 'uncalibrated')
    result['primary_uncalibrated'] = show('Primary, without the calibrator', raw, miss, len(PRIMARY))
    for cohort, text in (('old', 'sensors the old checkpoint knows'), ('new', 'sensors the old checkpoint does not know')):
        part, m2 = items_of(data, PRIMARY, 'calibrated', cohort)
        result['primary_' + cohort] = show('Primary, %s, with the calibrator' % text, part, m2, len(PRIMARY))
    sec, miss_s = items_of(data, SECONDARY)
    result['secondary'] = show('Secondary (PEMS05 2020 / 2025, TFNSW 2020), with the calibrator', sec, miss_s, len(SECONDARY))
    dev, miss_d = items_of(data, DEVELOPMENT)
    result['development'] = show('Development (TFNSW 2025; never part of a decision), with the calibrator', dev, miss_d, len(DEVELOPMENT))
    if prim or sec or dev:
        print('\n| dataset/year | same | ' + ' | '.join(ARMS) + ' |\n|---|---:|' + '---:|' * len(ARMS))
        for (d, y), r in prim + sec + dev:
            v = lambda arm: A.value(data[(d, y)], arm, 'calibrated')
            print('| %s/%d | %.2f | %s |' % (d, y, v('same'), ' | '.join('%.3f' % r[arm + ' / same'] for arm in ARMS)))
    print('\n### Cost of the fits (primary dataset-years; H100 unless the runs say otherwise)')
    med = costs(a.root, PRIMARY, result)
    print('\n### Rules fixed in PRESPEC.md (primary set: %d of %d dataset-years)' % (len(prim), len(PRIMARY)))
    if miss or not prim:
        print('- not evaluated: incomplete (%s)' % ', '.join(miss))
    else:
        c = lambda n: '%.3f, CI %.3f-%.3f' % (p[n]['gm'], p[n]['lo'], p[n]['hi'])
        rules = dict(P1=p['fit_emb_sel / fit_all']['hi'] <= 1.01, P2=p['fit_emb_sel / fit_shared']['hi'] < 1, P3=p['fit_emb_sel / fit_head']['hi'] < 1,
                     P4=p['fit_emb_sel / fit_same']['hi'] <= A.MARGIN, C1=bool(med is not None and med <= .67))
        h = lambda k: 'holds' if rules[k] else 'does not hold'
        print('- P1, refitting only the embeddings is as accurate as fine-tuning everything (fit_emb_sel / fit_all CI at most 1.01): **%s** (%s)' % (h('P1'), c('fit_emb_sel / fit_all')))
        print('- P2, the embeddings are a better place than the shared weights (fit_emb_sel / fit_shared CI below 1): **%s** (%s)' % (h('P2'), c('fit_emb_sel / fit_shared')))
        print('- P3, ... and than the output head (fit_emb_sel / fit_head CI below 1): **%s** (%s)' % (h('P3'), c('fit_emb_sel / fit_head')))
        print('- P4, the refitted old model is within the g29 margin of the same-year model with the same refit (fit_emb_sel / fit_same CI at most %.2f): **%s** (%s)' % (A.MARGIN, h('P4'), c('fit_emb_sel / fit_same')))
        print('- C1, the embedding-only fit is cheaper to run than the full fine-tune (median seconds per epoch ratio at most 0.67): **%s** (%s)' % (h('C1'), 'median %.2f' % med if med is not None else 'no timing'))
        print('- "the embedding table is the place to maintain" (needs P1, P2, P3): **%s**' % ('holds' if rules['P1'] and rules['P2'] and rules['P3'] else 'does not hold'))
        print('- the worth of a second pass over the most recent windows for the same-year model, fit_same / same: %s' % c('fit_same / same'))
        print('- what choosing the rate is worth for the embedding fit, fit_emb_sel / fit_emb: %s' % c('fit_emb_sel / fit_emb'))
        result['rules'] = {k: bool(v) for k, v in rules.items()}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
