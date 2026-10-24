"""Recompute the annual-benchmark summaries of the paper from the committed per-cell results; no model execution.

Usage:
    python scripts/render_annual_results.py            # print the ten-network table (Table 9 of the extended version)
    python scripts/render_annual_results.py --check    # also compare every recomputed value with results/evost/annual/summary.json

Inputs: results/evost/annual/cells.csv (one row per network, year, evaluation seed and stream) and
results/evost/annual/per_step.csv (per-step errors of each network, stream and seed, averaged over years).
Aggregation, as in the paper: for each seed, average the yearly values with equal weight; report the mean and the sample
standard deviation (ddof=1) over the five seeds. Ten-network averages weight every network equally. Relative reductions
are (reference - EvoST) / reference, computed per network and then averaged.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'evost' / 'annual'
NETWORKS = ('PEMS03', 'PEMS04', 'PEMS05', 'PEMS06', 'PEMS07', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12', 'TFNSW')
STREAMS = ('frozen', 'frozen+RLS', 'refit', 'EvoST')
METRICS = ('MAE', 'RMSE', 'MAPE')
HORIZONS = ('Avg', '3', '6', '12')
STEP_METHOD = {'evost': 'EvoST', 'evost_frozen': 'frozen', 'evost_frozen_rls': 'frozen+RLS', 'evost_refit': 'refit'}


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def table_from_cells():
    by = {}
    for r in csv.DictReader(open(ROOT / 'cells.csv')):
        for h in HORIZONS:
            for m in METRICS:
                col = '%s_%s' % ('Avg' if h == 'Avg' else 'H' + h, m)
                by.setdefault((r['dataset'], r['stream'], '%s_%s' % (h, m), int(r['seed'])), []).append(float(r[col]))
    out = {}
    for (net, stream, key, seed), values in by.items():
        out.setdefault((net, stream, key), {})[seed] = mean(values)
    return {k: dict(mean=mean(list(v.values())), sd=sd(list(v.values())), seeds=len(v)) for k, v in out.items()}


def per_step_counts():
    series = {}
    for r in csv.DictReader(open(ROOT / 'per_step.csv')):
        if not r['horizon'].startswith('step'):      # the file also keeps the cumulative horizons and Avg
            continue
        key = (r['dataset'], STEP_METHOD[r['method']], int(r['horizon'][4:]))
        for m in METRICS:
            series.setdefault(key + (m,), []).append(float(r[m]))
    avg = {k: mean(v) for k, v in series.items()}
    res = {}
    for ref in ('frozen+RLS', 'frozen'):
        lower, total, red = 0, 0, {m: [] for m in METRICS}
        for net in NETWORKS:
            for m in METRICS:
                per_net = []
                for step in range(1, 13):
                    e, r = avg[(net, 'EvoST', step, m)], avg[(net, ref, step, m)]
                    lower += e < r
                    total += 1
                    per_net.append((r - e) / r)
                red[m].append(mean(per_net))
        res[ref] = dict(evost_lower=lower, combinations=total, mean_reduction={m: mean(v) for m, v in red.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    table = table_from_cells()
    print('Avg errors of the same per-year STAEFormer checkpoints (mean +- sample SD over five seeds; MAPE in percent)')
    print('%-7s ' % 'Network' + ' | '.join('%-26s' % s for s in STREAMS))
    for net in NETWORKS:
        cells = []
        for s in STREAMS:
            cells.append(' '.join('%6.2f+-%.2f' % (table[(net, s, 'Avg_' + m)]['mean'], table[(net, s, 'Avg_' + m)]['sd']) for m in METRICS[:1]) +
                         '  ' + ' '.join('%.2f' % table[(net, s, 'Avg_' + m)]['mean'] for m in METRICS[1:]))
        print('%-7s ' % net + ' | '.join('%-26s' % c for c in cells))
    steps = per_step_counts()
    for ref, v in steps.items():
        print('per-step: EvoST below %s in %d of %d network-metric-step combinations; mean reduction %s' % (
            ref, v['evost_lower'], v['combinations'], ', '.join('%s %.1f%%' % (m, 100 * x) for m, x in v['mean_reduction'].items())))
    if not a.check:
        return 0
    summary = json.loads((ROOT / 'summary.json').read_text())
    problems = []
    for key, v in summary['table'].items():
        net, stream = key.split('/')
        for k, x in v['values'].items():
            mine = table[(net, stream, k)]
            for f in ('mean', 'sd'):
                if abs(mine[f] - x[f]) > 1e-9 * max(1.0, abs(x[f])):
                    problems.append('%s %s %s: %r vs %r' % (key, k, f, mine[f], x[f]))
    for ref, name in (('frozen+RLS', 'EvoST vs Frozen+RLS'), ('frozen', 'EvoST vs Frozen')):
        want = summary['per_step'][name]
        if steps[ref]['evost_lower'] != want['evost_lower'] or steps[ref]['combinations'] != want['combinations']:
            problems.append('per-step %s counts differ' % name)
        for m in METRICS:
            if abs(steps[ref]['mean_reduction'][m] - want['mean_reduction_ten_networks'][m]) > 1e-9:
                problems.append('per-step %s %s reduction differs' % (name, m))
    for name, (e, r) in (('EvoST vs frozen', ('EvoST', 'frozen')), ('frozen+RLS vs frozen', ('frozen+RLS', 'frozen')),
                         ('EvoST vs frozen+RLS', ('EvoST', 'frozen+RLS')), ('refit vs frozen', ('refit', 'frozen'))):
        for m in METRICS:
            mine = mean([(table[(n, r, 'Avg_' + m)]['mean'] - table[(n, e, 'Avg_' + m)]['mean']) / table[(n, r, 'Avg_' + m)]['mean'] for n in NETWORKS])
            want = summary['matched_controls'][name]['ten_network_mean'][m]
            if abs(mine - want) > 1e-9:
                problems.append('matched control %s %s: %r vs %r' % (name, m, mine, want))
    print('check:', 'all values agree with summary.json' if not problems else '%d differences' % len(problems))
    for p in problems[:20]:
        print('  ', p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
