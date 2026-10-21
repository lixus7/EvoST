"""Pre-specified summary of g36 (PRESPEC.md): identity refresh with Graph WaveNet as a third backbone (read-only).

  python analyze_g36.py DELIVERED_ROOT [--bundles DIR] [--g33 ROOT ...] [--g35 ROOT] [--out summary.json]

DELIVERED_ROOT holds ckpt_V/<dataset>/train_report.json (the per-year Graph WaveNet models) and
results_V/gwn-<dataset>-<year>-s<seed>/<dataset>_<year>/seed<seed>/e3_metrics.jsonl.
Endpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53; geometric means of ratios with the g29 bootstrap
over districts. The rules are those of g33 (F1, F2, N1, N2, B1) with the precondition of g35, on eight districts x 2020,
2025, plus three readings about the two graphs of the model (L1 learned graph, L2 given edges, L3 given graph at all).
With --bundles (the directory that holds stae/<dataset>_<year>.npz) every trained model is compared with the last-value
forecast on its own validation split; dataset-years that involve a model no better than that are listed and the rules
are evaluated a second time without them.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import analyze_g29 as A
import analyze_g33 as G33

EIGHT = ('PEMS03', 'PEMS04', 'PEMS06', 'PEMS07', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
PRIMARY = [(d, y) for d in EIGHT for y in (2020, 2025)]
SECONDARY = [('PEMS05', 2020), ('PEMS05', 2025), ('TFNSW', 2020)]
DEVELOPMENT = [('TFNSW', 2025)]
EXTRA = collections.OrderedDict([('stale5o_mean / same_mean', ('stale5o_mean', 'same_mean')), ('same_mean / same', ('same_mean', 'same')),
                                 ('same_perm / same', ('same_perm', 'same')), ('same_self / same', ('same_self', 'same'))])
GRAPH = (('L1', 'same_mean / same', 'the learned graph carries information (uniform adaptive adjacency is worse)'),
         ('L2', 'same_perm / same', 'the given edges carry information (node-permuted static adjacency is worse)'),
         ('L3', 'same_self / same', 'the given graph carries information at all (self-loops only is worse)'))
FAITHFUL = 1.05


def geo(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return float(np.exp(np.log(v).mean())) if len(v) else None


def last_value_validation(bundle, batch=128):
    """The trainer's validation error (mean over batches of the masked MAE) of the forecast 'last observed value'."""
    import g28_common as g
    z, _ = g.load_bundle(bundle)
    raw = np.nan_to_num(np.asarray(z['raw_val'], np.float64), nan=0., posinf=0., neginf=0.)
    n, total, cnt = raw.shape[0] - g.SPAN, 0., 0
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        last = raw[idx + g.X_LEN - 1]                                  # [B, N]
        y = np.stack([raw[idx + g.X_LEN + h] for h in range(g.Y_LEN)], -1)
        valid = y != 0
        if valid.any():
            total, cnt = total + float((np.abs(y - last[..., None]) * valid).sum() / valid.sum()), cnt + 1
    return total / max(cnt, 1)


def failed_trainings(reports, bundles, result):
    """Dataset-years (as evaluated: target year, with its same-year and five-year-old models) that involve a failed optimisation."""
    bad_models, naive = [], {}
    for ds, rep in reports.items():
        for name, r in rep['runs'].items():
            key = (ds, r['year'])
            if key not in naive:
                naive[key] = last_value_validation(Path(bundles) / 'stae' / ('%s_%d.npz' % key))
            if not r['lowest_validation_loss'] < naive[key]:
                bad_models.append((ds, r['year'], r['seed'], r['lowest_validation_loss'], naive[key]))
    worst = max((r['lowest_validation_loss'] / naive[(ds, r['year'])] for ds, rep in reports.items() for r in rep['runs'].values()), default=None)
    print('\nTrained models against the last-value forecast on their own validation split: %d models, largest ratio %s; failed optimisations (ratio at least 1): %s' % (
        sum(len(rep['runs']) for rep in reports.values()), '%.3f' % worst if worst else '-',
        ', '.join('%s/%d seed %d (%.2f against %.2f)' % b for b in bad_models) or 'none'))
    affected = sorted({(ds, y) for ds, year, *_ in bad_models for y in (year, year + 5) if (ds, y) in PRIMARY + SECONDARY + DEVELOPMENT})
    result['failed_trainings'] = dict(models=bad_models, dataset_years=affected, largest_ratio=worst)
    return affected


def training(root, result):
    print('\n## The per-year Graph WaveNet models (train_gwn.py; not a result of the experiment)')
    reports = {f.parent.name: json.loads(f.read_text()) for f in sorted(Path(root).glob('ckpt_V/*/train_report.json'))}
    print('Datasets trained: %d (%s)' % (len(reports), ', '.join(reports)))
    if not reports:
        return
    print('\n| dataset | year | epochs run (3 seeds) | validation MAE | test T12 MAE, inputs normalised with train statistics | repository checkpoint, same windows | retrained / repository |\n|---|---|---|---:|---:|---:|---:|')
    ratios, pipeline = {}, []
    for ds, rep in reports.items():
        for year in (2015, 2020, 2025):
            runs = [r for r in rep['runs'].values() if r['year'] == year]
            if not runs:
                continue
            mine = float(np.mean([r['metrics']['test_train_stats']['T12_MAE'] for r in runs]))
            ref = [r['repository_checkpoint'] for r in runs if 'repository_checkpoint' in r]
            cell, ratio = '-', '-'
            if ref:
                theirs = float(np.mean([r['metrics']['test_train_stats']['T12_MAE'] for r in ref]))
                own = float(np.mean([r['metrics']['test_own_stats']['T12_MAE'] for r in runs])) / float(np.mean([r['metrics']['test_own_stats']['T12_MAE'] for r in ref]))
                ratios[ds] = dict(train_stats=mine / theirs, own_stats=own)
                cell, ratio = '%.3f' % theirs, '%.3f' % (mine / theirs)
                for r in ref:
                    if r.get('logged_test') and '12' in r['logged_test']:
                        pipeline.append(abs(r['metrics']['test_own_stats']['T12_MAE'] - r['logged_test']['12'][0]) / r['logged_test']['12'][0])
            print('| %s | %d | %s | %.3f | %.3f | %s | %s |' % (ds, year, ' / '.join(str(r['epochs_run']) for r in runs),
                                                            np.mean([r['lowest_validation_loss'] for r in runs]), mine, cell, ratio))
    if pipeline:
        print('\nCheck: the repository\'s 2025 checkpoints, evaluated here with the repository\'s input normalisation, against the test error in the repository\'s logs: '
              'max relative difference %.1e over %d checkpoints (%s)' % (max(pipeline), len(pipeline), 'OK' if max(pipeline) <= 2e-3 else 'DIFFERS'))
    if ratios:
        gm = geo([v['train_stats'] for v in ratios.values()])
        gm_own = geo([v['own_stats'] for v in ratios.values()])
        ok = gm_own <= FAITHFUL
        print('Retrained 2025 models against the repository\'s 2025 checkpoints (test T12 MAE, geometric mean over %d datasets): %.3f with the repository\'s input '
              'normalisation, %.3f with train statistics; range %.3f-%.3f.' % (len(ratios), gm_own, gm, min(v['own_stats'] for v in ratios.values()), max(v['own_stats'] for v in ratios.values())))
        print('- pre-specified reading (at most %.2f with the repository\'s normalisation): the retraining **%s** the repository\'s recipe' % (
            FAITHFUL, 'reproduces' if ok else 'does not reproduce'))
        result['training'] = dict(retrained_over_repository=ratios, geometric_mean_own_stats=gm_own, geometric_mean_train_stats=gm, faithful=bool(ok),
                                  pipeline_max_relative_difference=max(pipeline) if pipeline else None)


def extra_items(data, keys, kind='calibrated'):
    items = []
    for k in keys:
        if k in data and sorted(data[k]) == list(A.SEEDS):
            v = lambda arm: A.value(data[k], arm, kind)
            items.append((k, {n: (None if None in (v(a), v(b)) else v(a) / v(b)) for n, (a, b) in EXTRA.items()}))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--g33', nargs='*', default=[], help='g33 / g34 result roots: STAEFormer numbers of the same dataset-years, for a side-by-side line')
    ap.add_argument('--g35', help='delivered root of g35: STID numbers of the same dataset-years, for a side-by-side line')
    ap.add_argument('--bundles', help='directory with stae/<dataset>_<year>.npz: enables the check of every trained model against the last-value forecast')
    ap.add_argument('--out')
    a = ap.parse_args()
    result = {}
    print('# g36: identity refresh of a five-year-old frozen Graph WaveNet (its per-sensor embeddings generate the adaptive adjacency)')
    training(a.root, result)
    affected = []
    if a.bundles:
        reports = {f.parent.name: json.loads(f.read_text()) for f in sorted(Path(a.root).glob('ckpt_V/*/train_report.json'))}
        affected = failed_trainings(reports, a.bundles, result)
    data = G33.load(Path(a.root) / 'results_V')
    print('\n## The experiment\nEndpoint: test-portion H12 MAE with the calibrator, mean of seeds 51-53. Runs found: %d dataset-years, %d with all three seeds.' % (
        len(data), sum(sorted(v) == list(A.SEEDS) for v in data.values())))
    prim, miss = G33.items_of(data, PRIMARY)
    result['primary'] = p = G33.show('Primary (eight districts x 2020, 2025), with the calibrator', prim, miss, len(PRIMARY))
    raw, _ = G33.items_of(data, PRIMARY, 'uncalibrated')
    result['primary_uncalibrated'] = G33.show('Primary, without the calibrator', raw, miss, len(PRIMARY))
    for kind in ('calibrated', 'uncalibrated'):
        items = extra_items(data, PRIMARY, kind)
        for name in EXTRA:
            part = [(k, r) for k, r in items if r[name] is not None]
            if part:
                s = A.geometric(part, name)
                result['%s (%s)' % (name, kind)] = s
                print('- %s, %s: **%.3f** (95%% CI %.3f-%.3f; %d dataset-years)' % (name, 'with the calibrator' if kind == 'calibrated' else 'without', s['gm'], s['lo'], s['hi'], s['n']))
    cohorts = {}
    for cohort, text in (('old', 'sensors the old checkpoint knows'), ('new', 'sensors the old checkpoint does not know')):
        part, m2 = G33.items_of(data, PRIMARY, 'calibrated', cohort)
        cohorts[cohort] = result['primary_' + cohort] = G33.show('Primary, %s, with the calibrator' % text, part, m2, len(PRIMARY))
    sec, miss_s = G33.items_of(data, SECONDARY)
    result['secondary'] = G33.show('Secondary (PEMS05 2020 / 2025, TFNSW 2020), with the calibrator', sec, miss_s, len(SECONDARY))
    dev, miss_d = G33.items_of(data, DEVELOPMENT)
    result['development'] = G33.show('Development (TFNSW 2025; never part of a decision), with the calibrator', dev, miss_d, len(DEVELOPMENT))
    if prim:
        print('\n| dataset/year | same | stale5o | stale5o_mean | don_prof | new_prof | fit_emb | stale5o / same | fit_emb / same |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for (d, y), r in prim + sec + dev:
            v = lambda arm: A.value(data[(d, y)], arm, 'calibrated')
            print('| %s/%d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f | %.3f |' % (d, y, v('same'), v('stale5o'), v('stale5o_mean'), v('don_prof'), v('new_prof'), v('fit_emb'),
                                                                               r['stale5o / same'], r['fit_emb / same']))
    def rules_on(keys, label):
        items, missing = G33.items_of(data, keys)
        print('\n### Rules of g33, fixed again in PRESPEC.md (%s: %d of %d dataset-years)' % (label, len(items), len(keys)))
        if missing:
            print('- not evaluated: incomplete (%s)' % ', '.join(missing))
            return None
        s = lambda its, n: A.geometric([(k, r) for k, r in its if r.get(n) is not None], n)
        q = {n: s(items, n) for n in items[0][1] if any(r.get(n) is not None for _, r in items)}
        new = {n: s(G33.items_of(data, keys, 'calibrated', 'new')[0], n) for n in ('new_prof / stale5o',)}
        gap = q['stale5o / same']
        rho = lambda arm: (gap['gm'] - q[arm + ' / same']['gm']) / (gap['gm'] - 1.)
        c = lambda n, src=q: '%.3f, CI %.3f-%.3f' % (src[n]['gm'], src[n]['lo'], src[n]['hi'])
        aged = gap['lo'] > 1
        print('- precondition, the five-year-old Graph WaveNet is worse than the same-year one (lower CI bound of stale5o / same above 1): **%s** (%s)' % ('holds' if aged else 'does not hold', c('stale5o / same')))
        if not aged:
            print('- F1, F2, N1 are not evaluated: there is no loss to repair; this is itself the finding for this backbone')
            return dict(aged=False)
        rules = dict(F1=q['fit_emb / stale5o']['hi'] < 1 and rho('fit_emb') >= .5, F2=q['fit_emb / same']['hi'] <= A.MARGIN,
                     N1=q['new_prof / stale5o']['hi'] < 1 and rho('new_prof') >= 1 / 3 and q['new_prof / same']['gm'] < q['stale5o_mean / same']['gm'],
                     N2=new['new_prof / stale5o'] is not None and new['new_prof / stale5o']['hi'] < 1,
                     B1=q['same_prof / same_geo']['hi'] < 1 and q['don_prof / don_geo']['hi'] < 1)
        h = lambda k: 'holds' if rules[k] else 'does not hold'
        print('- F1 (fit_emb / stale5o CI below 1, recovery at least 1/2): **%s** (%s; recovery %.0f%%)' % (h('F1'), c('fit_emb / stale5o'), 100 * rho('fit_emb')))
        print('- F2 (fit_emb / same CI at most %.2f): **%s** (%s)' % (A.MARGIN, h('F2'), c('fit_emb / same')))
        print('- N1 (new_prof / stale5o CI below 1, recovery at least 1/3, better than the mean embedding): **%s** (%s; recovery %.0f%%)' % (h('N1'), c('new_prof / stale5o'), 100 * rho('new_prof')))
        if new['new_prof / stale5o'] is not None:
            print('- N2 (unknown sensors, new_prof / stale5o CI below 1): **%s** (%s)' % (h('N2'), c('new_prof / stale5o', new)))
        print('- B1 (same_prof / same_geo and don_prof / don_geo, both CIs below 1): **%s** (%s; %s)' % (h('B1'), c('same_prof / same_geo'), c('don_prof / don_geo')))
        return dict({k: bool(v) for k, v in rules.items()}, aged=True, recovery_fit_emb=rho('fit_emb'), recovery_new_prof=rho('new_prof'))

    result['rules'] = rules_on(PRIMARY, 'primary set')
    graph = extra_items(data, PRIMARY)
    if len(graph) == len(PRIMARY):
        print('\n### The two graphs of the model (primary set, with the calibrator; fixed in PRESPEC.md)')
        readings = {}
        for tag, name, text in GRAPH:
            s = A.geometric([(k, r) for k, r in graph if r[name] is not None], name)
            readings[tag] = bool(s['lo'] > 1)
            print('- %s, %s (lower CI bound of %s above 1): **%s** (%.3f, CI %.3f-%.3f)' % (tag, text, name, 'holds' if s['lo'] > 1 else 'does not hold', s['gm'], s['lo'], s['hi']))
        result['graph_readings'] = readings
    drop = [k for k in PRIMARY if k in affected]
    if drop:
        result['rules_without_failed_trainings'] = rules_on([k for k in PRIMARY if k not in affected],
                                                            'primary set without %s, which involve a failed optimisation' % ', '.join('%s/%d' % k for k in drop))
        print('- if the two evaluations disagree on a rule, the claim is limited to what both support (PRESPEC.md)')
    for label, roots in (('STAEFormer (g33 / g34)', a.g33), ('STID (g35)', [str(Path(a.g35) / 'results_V')] if a.g35 else [])):
        ref = collections.defaultdict(dict)
        for root in roots:
            for k, v in G33.load(root).items():
                ref[k].update(v)
        common = [k for k in PRIMARY if k in ref and sorted(ref[k]) == list(A.SEEDS) and k in data and sorted(data[k]) == list(A.SEEDS)]
        if common:
            mine, theirs = G33.items_of(data, common)[0], G33.items_of(ref, common)[0]
            print('\n### Side by side with %s on the %d common dataset-years, with the calibrator' % (label, len(common)))
            for name in ('stale5o / same', 'fit_emb / same', 'stale5o_mean / same', 'same_mean / same', 'new_prof / same', 'same_prof / same', 'same_geo / same'):
                x, y = A.geometric(mine, name), A.geometric(theirs, name)
                print('- %s: Graph WaveNet %.3f (%.3f-%.3f), %s %.3f (%.3f-%.3f)' % (name, x['gm'], x['lo'], x['hi'], label.split(' ')[0], y['gm'], y['lo'], y['hi']))
            s = [(k, dict(r=A.value(data[k], 'same', 'calibrated') / A.value(ref[k], 'same', 'calibrated'))) for k in common]
            gm_ = A.geometric(s, 'r')
            print('- same-year Graph WaveNet + calibrator / same-year %s + calibrator: %.3f (%.3f-%.3f)' % (label.split(' ')[0], gm_['gm'], gm_['lo'], gm_['hi']))
    fits, chosen, by_rate = [], collections.Counter(), collections.defaultdict(list)
    for f in sorted((Path(a.root) / 'results_V').rglob('e10_info.json')):
        if not (f.parent / 'completed.json').exists() or json.loads((f.parent / 'completed.json').read_text()).get('bounded_smoke'):
            continue
        fit = json.loads(f.read_text()).get('fit')
        if fit:
            h = fit['holdout_mae']
            fits.append((min(h) / h[0], fit['best_epoch'], fit['seconds']))
            chosen['%g' % fit['selected_lr']] += 1
            for rate, c in fit['candidates'].items():
                by_rate[rate].append(min(c['holdout_mae']) / c['holdout_mae'][0])
    if fits:
        r = np.asarray(fits)
        print('\nEmbedding fit: holdout MAE inside the fit windows after / before, median %.2f (range %.2f-%.2f); best epoch median %d; seconds (four candidate rates) median %.0f, max %.0f (%d runs)' % (
            np.median(r[:, 0]), r[:, 0].min(), r[:, 0].max(), np.median(r[:, 1]), np.median(r[:, 2]), r[:, 2].max(), len(r)))
        print('Learning rate chosen per run on that holdout: ' + ', '.join('%s in %d runs' % (k, chosen[k]) for k in sorted(chosen, key=float)) +
              '; median holdout after / before per candidate: ' + ', '.join('%s %.3f' % (k, np.median(by_rate[k])) for k in sorted(by_rate, key=float)))
        result['fit_rates_chosen'] = dict(chosen)
        result['fit_holdout_median_by_rate'] = {k: float(np.median(v)) for k, v in by_rate.items()}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
