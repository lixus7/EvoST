"""Check paper numbers that the two render scripts do not cover, from files in this repository; no model execution.

Every check prints the value printed in the paper, the recomputed value rounded the same way, and whether they agree. The
checks cover the results listed below; they do not verify every number of the paper.
Figure and table numbers are those of the submission, with the extended version's in brackets where they differ.

  sensitivity    Section 6.8, Figure 6(a, b): calibrator settings and identity learning rates of the annual runs
  budget         Section 6.8, Figure 6(c): history budget with and without the calibrator
  ages           Section 6.8, Figure 6(d): checkpoint ages
  known-unseen   Section 6.3 and Table 10 of the extended version: sensors the old checkpoint knows and unseen ones
  new-sensors    Section 6.6, Figure 4 [5]: annual-benchmark sensors absent from the preceding year's graph
  case-study     Section 6.9, Figure 7 [9]: PEMS12 in 2020 with the 2015 checkpoint
  arrival-recal  Section 6.5: rebuilding every calibrator in the new-only refit
  arrival-scale  Appendix A.3 [A.5]: the calibrators' scale s restricted to the targets the arrival masks allow

Usage: python scripts/check_paper_numbers.py [SECTION ...]      (all sections by default)
"""
import collections
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'evost'
NETWORKS = ('PEMS03', 'PEMS04', 'PEMS05', 'PEMS06', 'PEMS07', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12', 'TFNSW')
ROWS = []


def claim(section, text, printed, value, fmt):
    shown = fmt(value) if callable(fmt) else fmt.format(value)
    ROWS.append((section, text, printed, shown, shown == printed))


def below(limit):
    return lambda v: 'below %g' % limit if v < limit else '%.3f' % v


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def network_mean(per_network):
    """Mean within each network, then equal weight over the ten networks."""
    assert sorted(per_network) == sorted(NETWORKS), sorted(per_network)
    return sum(sum(v) / len(v) for v in per_network.values()) / len(per_network)


def sensitivity():
    s = 'sensitivity'
    runs, chosen = collections.defaultdict(dict), collections.Counter()
    for r in read_csv(RES / 'annual' / 'rls_selection.csv'):
        run, setting = (r['dataset'], r['year'], r['seed']), (float(r['ridge']), float(r['forget']))
        assert setting not in runs[run], ('duplicate row', run, setting)
        runs[run][setting] = float(r['validation_H12_MAE'])
        chosen[run] += int(r['chosen'])
    assert all(len(grid) == 8 for grid in runs.values()) and all(chosen[run] == 1 for run in runs), 'eight settings and one choice per run'
    claim(s, 'annual runs in panel (a)', '1,165', len(runs), '{:,}')
    excess = {}
    for setting in sorted(next(iter(runs.values()))):
        per = collections.defaultdict(list)
        for (net, _, _), grid in runs.items():
            per[net].append(grid[setting] / min(grid.values()))
        excess[setting] = 100 * (network_mean(per) - 1)
    claim(s, '(a) excess validation H12 MAE at lambda 0.99, rho 1 (%)', '5.7', excess[(1.0, 0.99)], '{:.1f}')
    claim(s, '(a) excess validation H12 MAE at lambda 0.99, rho 100 (%)', '4.1', excess[(100.0, 0.99)], '{:.1f}')
    claim(s, '(a) largest excess at lambda 0.999 or 1 (%)', 'below 1',
          max(excess[(r, l)] for r in (1.0, 100.0) for l in (0.999, 1.0)), below(1))
    across_forget = max(max(excess[(r, l)] for l in (0.99, 0.995, 0.999, 1.0)) - min(excess[(r, l)] for l in (0.99, 0.995, 0.999, 1.0))
                        for r in (1.0, 100.0))
    across_ridge = max(abs(excess[(1.0, l)] - excess[(100.0, l)]) for l in (0.99, 0.995, 0.999, 1.0))
    claim(s, '(a) error varies more across forgetting factors than across ridge penalties', 'yes',
          across_forget > across_ridge, lambda v: 'yes' if v else 'no')
    fits, selected, per_fit = collections.defaultdict(dict), collections.Counter(), collections.Counter()
    for r in read_csv(RES / 'annual' / 'identity_lr.csv'):
        run = (r['dataset'], r['year'], r['seed'])
        assert r['lr'] not in fits[run], ('duplicate row', run, r['lr'])
        fits[run][r['lr']] = float(r['best_holdout_MAE'])
        selected[r['lr']] += int(r['selected'])
        per_fit[run] += int(r['selected'])
    assert all(sorted(c) == ['0.001', '0.003', '0.01'] for c in fits.values()) and all(per_fit[run] == 1 for run in fits), \
        'three rates and one choice per fit'
    claim(s, '(b) annual identity fits', '1,160', len(fits), '{:,}')
    claim(s, '(b) fits that selected the rate 1e-3', '1,100', selected['0.001'], '{:,}')
    per = collections.defaultdict(list)
    for (net, _, _), cand in fits.items():
        per[net].append(cand['0.01'] / min(cand.values()))
    claim(s, '(b) excess holdout MAE of the rate 1e-2 (%)', '4.0', 100 * (network_mean(per) - 1), '{:.1f}')


def budget():
    s = 'budget'
    sys.path.insert(0, str(ROOT / 'experiments' / 'evost' / 'history_budget'))
    import analyze_g29 as A
    import analyze_g38 as G
    pairs = collections.OrderedDict([('stale5o / same', ('stale5o', 'same')), ('fit_emb / same', ('fit_emb', 'same')),
                                     ('fit_emb / stale5o', ('fit_emb', 'stale5o'))])
    gm = {}
    for b in (64, 1024, 2048):
        files = sorted((RES / 'history_budget' / 'runs').glob('bud-*/*/seed*/b%d/e3_metrics.jsonl' % b))
        per = G.load_runs(files)
        ok, missing = G.complete(per, G.PRIMARY, ('same', 'stale5o', 'fit_emb'))
        assert not missing and len(ok) == 12, (b, missing)
        for kind in ('calibrated', 'uncalibrated'):
            for name in pairs:
                gm[(b, kind, name)] = A.geometric(G.ratio_items(per, ok, pairs, kind), name)['gm']
    claim(s, '(c) old checkpoint, 1,024 windows, calibrated ratio', '1.175', gm[(1024, 'calibrated', 'stale5o / same')], '{:.3f}')
    claim(s, '(c) refresh, 1,024 windows, calibrated ratio', '1.046', gm[(1024, 'calibrated', 'fit_emb / same')], '{:.3f}')
    claim(s, '(c) refresh, 2,048 windows, calibrated ratio', '1.026', gm[(2048, 'calibrated', 'fit_emb / same')], '{:.3f}')
    claim(s, '(c) uncalibrated MAE reduction of refresh, 64 windows (%)', '8', 100 * (1 - gm[(64, 'uncalibrated', 'fit_emb / stale5o')]), '{:.0f}')
    claim(s, '(c) uncalibrated MAE reduction of refresh, 2,048 windows (%)', '33', 100 * (1 - gm[(2048, 'uncalibrated', 'fit_emb / stale5o')]), '{:.0f}')


def ages():
    s = 'ages'
    panels = {p['age_years']: p for p in json.loads((ROOT / 'figures' / 'evost' / 'checkpoint_age.json').read_text())['panels']}
    for age, settings, gap, ratio, recovery in ((5, 18, '18.3', '1.016', '91'), (10, 12, '21.7', '1.033', '85'), (15, 6, '21.2', '1.023', '89')):
        p = panels[age]
        old, ref = p['old']['gm'], p['refresh']['gm']
        claim(s, '(d) %d-year-old checkpoints: settings' % age, str(settings), p['settings'], '{}')
        claim(s, '(d) %d-year-old checkpoints: gap (%%)' % age, gap, 100 * (old - 1), '{:.1f}')
        claim(s, '(d) %d-year-old checkpoints: refreshed ratio' % age, ratio, ref, '{:.3f}')
        claim(s, '(d) %d-year-old checkpoints: gap recovered (%%)' % age, recovery, 100 * (old - ref) / (old - 1), '{:.0f}')


def known_unseen():
    s = 'known-unseen'
    sys.path.insert(0, str(ROOT / 'scripts'))
    import render_evost_results as R
    data = R.load(37)
    nodes = {c: {p: data[(*p, 'same', c)][51]['nodes'] for p in R.PANELS} for c in ('all', 'old', 'new')}
    assert all(nodes['old'][p] + nodes['new'][p] == nodes['all'][p] for p in R.PANELS)
    total = {c: sum(v.values()) for c, v in nodes.items()}
    claim(s, 'sensors over the twelve settings', '18,996', total['all'], '{:,}')
    claim(s, 'known sensors', '15,384', total['old'], '{:,}')
    claim(s, 'unseen sensors', '3,612', total['new'], '{:,}')
    claim(s, 'unseen share (%)', '19.0', 100 * total['new'] / total['all'], '{:.1f}')
    shares = [100 * nodes['new'][p] / nodes['all'][p] for p in R.PANELS]
    claim(s, 'unseen share per setting (%)', '5.4--46.0', (min(shares), max(shares)), lambda v: '%.1f--%.1f' % v)
    claim(s, 'known share (%)', '81', 100 * total['old'] / total['all'], '{:.0f}')
    table = {'old': ('1.169 [1.157, 1.182]', '1.067 [1.055, 1.079]', '1.026 [1.015, 1.037]', '1.018 [1.010, 1.028]', '85'),
             'new': ('1.226 [1.207, 1.247]', '1.128 [1.114, 1.141]', '1.027 [1.017, 1.038]', '1.022 [1.015, 1.028]', '88'),
             'all': ('1.177 [1.166, 1.187]', '1.078 [1.068, 1.089]', '1.025 [1.015, 1.037]', '1.018 [1.010, 1.027]', '86')}
    name = dict(old='known', new='unseen', all='all')
    for c, printed in table.items():
        gm = {}
        for arm, text in zip(('stale5o', 'fit_shared', 'fit_emb_sel', 'fit_all'), printed):
            r = R.ratio(data, arm, cohort=c)
            gm[arm] = r['gm']
            claim(s, 'Table 10, %s sensors, %s' % (name[c], arm), text, (r['gm'], r['lo'], r['hi']), lambda v: '%.3f [%.3f, %.3f]' % v)
        claim(s, 'Table 10, %s sensors, recovery (%%)' % name[c], printed[4],
              100 * (gm['stale5o'] - gm['fit_emb_sel']) / (gm['stale5o'] - 1), '{:.0f}')
        if c != 'all':
            wins = sum(R.value(data, p, 'fit_emb_sel', c) < min(R.value(data, p, 'stale5o', c), R.value(data, p, 'fit_shared', c))
                       for p in R.PANELS)
            claim(s, 'settings where refresh beats Old and Shared, %s sensors' % name[c], '12', wins, '{}')


def load_records(folder, prefix=''):
    """Test-portion rows of every completed run below `folder`, keyed (dataset, year, arm, cohort) -> {seed: row}."""
    data = collections.defaultdict(dict)
    for path in sorted(Path(folder).glob(prefix + '*/**/e3_metrics.jsonl')):
        done = json.loads(path.with_name('completed.json').read_text())
        assert done['status'] == 'completed' and not done.get('bounded_smoke'), path
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r['portion'] == 'test':
                key = (r['dataset'], r['year'], r['arm'], r['cohort'])
                assert r['seed'] not in data[key], ('duplicate measurement', path, key)
                data[key][r['seed']] = r
    return data


def new_sensors():
    s = 'new-sensors'
    rows = read_csv(RES / 'annual' / 'new_sensors.csv')
    mean = collections.defaultdict(dict)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['network'], r['method'])].append(r)
    for (net, method), g in groups.items():
        assert len(g) == 5 and len({r['seed'] for r in g}) == 5, ('five distinct seeds', net, method)
        for m in ('MAE', 'RMSE', 'MAPE'):
            mean[(net, method)][m] = sum(float(r['H12_' + m]) for r in g) / 5
    printed = {('PEMS07', 'EvoST'): ('13.4', '23.6'), ('TFNSW', 'EvoST'): ('85.2', '182.5'),
               ('PEMS07', 'Frozen+RLS'): ('13.6', '23.8'), ('TFNSW', 'Frozen+RLS'): ('85.4', '183.0'),
               ('PEMS07', 'STAEFormer'): ('15.1', '26.4'), ('TFNSW', 'STAEFormer'): ('137.0', '272.2')}
    for (net, method), (mae, rmse) in printed.items():
        claim(s, '%s, %s, H12 MAE' % (net, method), mae, mean[(net, method)]['MAE'], '{:.1f}')
        claim(s, '%s, %s, H12 RMSE' % (net, method), rmse, mean[(net, method)]['RMSE'], '{:.1f}')
    for net in ('PEMS07', 'TFNSW'):
        for m in ('MAE', 'RMSE'):
            best = min((v[m], k[1]) for k, v in mean.items() if k[0] == net)[1]
            claim(s, 'lowest %s of the plotted configurations, %s' % (m, net), 'EvoST', best, '{}')
    claim(s, 'largest excess of Frozen+RLS MAE over EvoST (%)', '1.3',
          max(100 * (mean[(n, 'Frozen+RLS')]['MAE'] / mean[(n, 'EvoST')]['MAE'] - 1) for n in ('PEMS07', 'TFNSW')), '{:.1f}')
    years = collections.Counter((r['network']) for r in read_csv(RES / 'annual' / 'new_sensors_years.csv') if r['arrival_year'] == '1')
    claim(s, 'arrival years, PEMS07 and TFNSW', '19, 19', (years['PEMS07'], years['TFNSW']), lambda v: '%d, %d' % v)
    stid = [int(r['years']) for r in rows if r['network'] == 'PEMS07' and r['method'] == 'STID']
    claim(s, 'PEMS07 STID years per seed', '13--17', (min(stid), max(stid)), lambda v: '%d--%d' % v)


def case_study():
    s = 'case-study'
    import statistics
    sensors = read_csv(RES / 'case_study' / 'sensors.csv')
    assert len({r['sensor_id'] for r in sensors}) == len(sensors), 'duplicate sensor'
    known = [r for r in sensors if r['known'] == '1']
    unseen = [r for r in sensors if r['known'] == '0']
    claim(s, 'sensors with valid test targets', '2,171', len(sensors), '{:,}')
    claim(s, 'sensors the old checkpoint has seen', '1,920', len(known), '{:,}')
    claim(s, 'sensors it has not seen', '251', len(unseen), '{:,}')
    for name, group, old, evost in (('seen', known, '1.13', '1.01'), ('unseen', unseen, '1.18', '1.00')):
        claim(s, 'median ratio of the old checkpoint, %s sensors' % name, old, statistics.median(float(r['ratio_old']) for r in group), '{:.2f}')
        claim(s, 'median ratio of EvoST, %s sensors' % name, evost, statistics.median(float(r['ratio_evost']) for r in group), '{:.2f}')
    claim(s, 'sensors where EvoST beats the old checkpoint (%)', '97',
          100 * sum(float(r['ratio_evost']) < float(r['ratio_old']) for r in sensors) / len(sensors), '{:.0f}')
    case = json.loads((RES / 'case_study' / 'case.json').read_text())
    claim(s, 'day of panel (b)', '2020-01-27', case['day'], '{}')
    curves = collections.defaultdict(list)
    for r in read_csv(RES / 'case_study' / 'forecasts.csv'):
        curves[r['sensor']].append(r)
    assert sorted(curves) == ['p50', 'p90', 'unseen'] and all(len(c) == 288 for c in curves.values()), 'three curves of 288 points'
    for model, printed in (('evost', ('5.4', '2.5', '14.1')), ('old', ('8.3', '3.9', '25.8'))):
        for label, text in zip(('p50', 'p90', 'unseen'), printed):
            c = curves[label]
            claim(s, 'mean absolute difference from the target-year forecasts, %s, %s sensor' % (model, label), text,
                  sum(abs(float(r[model]) - float(r['target_year'])) for r in c) / len(c), '{:.1f}')


def arrival_recal():
    s = 'arrival-recal'
    sys.path.insert(0, str(ROOT / 'scripts'))
    import render_evost_results as R
    data = load_records(RES / 'sensor_arrival' / 'recalibration_control', 'hnd-')
    r = R.ratio(data, 'r_prof0_r', reference='konly', cohort='old')
    claim(s, 'established sensors, new-only refit with every calibrator rebuilt, above reference (%)', '0.9', 100 * (r['gm'] - 1), '{:.1f}')
    a = R.ratio(data, 'a_prof0_r', reference='r_prof0_r', cohort='old')
    claim(s, 'all-identity refit against this control, geometric-mean change (%)', '-0.5', 100 * (a['gm'] - 1), '{:.1f}')
    claim(s, 'all-identity refit against this control, 95% interval (%)', '-1.1 to +0.1', (100 * (a['lo'] - 1), 100 * (a['hi'] - 1)),
          lambda v: '%.1f to %+.1f' % v)
    lower = sum(R.value(data, p, 'a_prof0_r', 'old') < R.value(data, p, 'r_prof0_r', 'old') for p in R.PANELS)
    claim(s, 'settings where the all-identity refit is lower', '8', lower, '{}')


def arrival_scale():
    s = 'arrival-scale'
    table = json.loads((RES / 'sensor_arrival' / 'scale_check.json').read_text())
    ratios = {r['setting']: r['legal_over_all'] for r in table}
    claim(s, 'change of s when restricted to the targets the masks allow (%)', '-3.4 to +8.9',
          (100 * (min(ratios.values()) - 1), 100 * (max(ratios.values()) - 1)), lambda v: '%.1f to %+.1f' % v)
    claim(s, 'equivalent factor on the ridge penalty', '0.93--1.19', (min(ratios.values()) ** 2, max(ratios.values()) ** 2),
          lambda v: '%.2f--%.2f' % v)
    largest = sorted(ratios, key=lambda k: -abs(ratios[k] - 1))[:2]
    claim(s, 'settings with the largest scale changes', 'PEMS03/2025, PEMS06/2020', sorted(largest), ', '.join)
    settings, seeds = (('PEMS03', 2025), ('PEMS06', 2020)), (51, 52, 53)
    quantities = {'new': [('c64_prof0', 'same'), ('s_prof0_r', 'same'), ('a_prof0_r', 'same'), ('a_prof0_r', 's_prof0_r'), ('x_prof0_r', 's_prof0_r')],
                  'all': [('c64_prof0', 'same'), ('s_prof0_r', 'same'), ('a_prof0_r', 'same')],
                  'old': [('c64_prof0', 'konly'), ('s_prof0_r', 'konly'), ('a_prof0_r', 'konly')]}
    rule = {name: {} for name in ('full', 'arrival')}
    chosen = {name: {} for name in ('full', 'arrival')}
    for name in rule:
        for ds, year in settings:
            for seed in seeds:
                d = RES / 'sensor_arrival' / 'scale_check_runs' / ('scl-%s-%d-s%d-%s' % (ds, year, seed, name)) / ('%s_%d' % (ds, year)) / ('seed%d' % seed)
                rows = [json.loads(line) for line in (d / 'e3_metrics.jsonl').read_text().splitlines()]
                rule[name].setdefault((ds, year), {})[seed] = {(r['arm'], r['portion'], r['cohort']): r for r in rows}
                chosen[name][(ds, year, seed)] = {a: (v['chosen']['ridge'], v['chosen']['forget'])
                                                  for a, v in json.loads((d / 'e3_selection.json').read_text())['selection'].items()}
    claim(s, 'selected calibration settings under the restricted scale', 'unchanged', chosen['full'] == chosen['arrival'],
          lambda v: 'unchanged' if v else 'changed')
    ref = load_records(RES / 'sensor_arrival' / 'runs', 'hnd-')
    def value(runs, arm, cohort):
        return sum(runs[seed][(arm, 'test', cohort)]['calibrated']['H12_MAE'] for seed in seeds) / len(seeds)
    worst_setting = worst_mean = 0.
    for cohort, pairs in quantities.items():
        for a, b in pairs:
            factor = 1.
            for k in settings:
                change = (value(rule['arrival'][k], a, cohort) / value(rule['arrival'][k], b, cohort)) / \
                         (value(rule['full'][k], a, cohort) / value(rule['full'][k], b, cohort)) - 1
                worst_setting = max(worst_setting, abs(change))
                factor *= (1 + change) ** (1 / 12)
            worst_mean = max(worst_mean, abs(factor - 1))
    full_equals_arrival = all(rule['full'][k][seed][(arm, 'test', c)]['calibrated']['H12_MAE'] == ref[(*k, arm, c)][seed]['calibrated']['H12_MAE']
                          for k in settings for seed in seeds for arm in ('same', 'konly', 'c64_prof0', 's_prof0_r', 'a_prof0_r')
                          for c in ('all', 'old', 'new'))
    claim(s, 'runs with the original scale reproduce the arrival records', 'yes', full_equals_arrival, lambda v: 'yes' if v else 'no')
    claim(s, 'largest change of a per-setting ratio (%)', '0.034', 100 * worst_setting, '{:.3f}')
    claim(s, 'largest change of a twelve-setting geometric mean (%)', 'below 0.01', 100 * worst_mean, below(0.01))


SECTIONS = collections.OrderedDict([('sensitivity', sensitivity), ('budget', budget), ('ages', ages), ('known-unseen', known_unseen),
                                    ('new-sensors', new_sensors), ('case-study', case_study), ('arrival-recal', arrival_recal),
                                    ('arrival-scale', arrival_scale)])


def main(argv):
    names = argv or list(SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        raise SystemExit('unknown section: %s (choose from %s)' % (', '.join(unknown), ', '.join(SECTIONS)))
    for name in names:
        SECTIONS[name]()
    width = max(len(r[1]) for r in ROWS)
    for section, text, printed, shown, ok in ROWS:
        print('%-13s %-*s  paper %-9s recomputed %-9s %s' % (section, width, text, printed, shown, 'ok' if ok else 'DIFFERENT'))
    bad = [r for r in ROWS if not r[4]]
    print('%d of %d checks agree with the paper' % (len(ROWS) - len(bad), len(ROWS)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
