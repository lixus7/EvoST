"""Rebuild EvoST result documentation from committed measurements; no model execution."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/evost/parameter_groups'))
# Study folders, keyed by the study numbers used in the source comments (g33-g42 in the development record).
FOLDER = {33: 'refresh_and_donors', 34: 'checkpoint_ages', 35: 'stid', 36: 'graph_wavenet', 37: 'parameter_groups',
          38: 'history_budget', 39: 'time_inputs', 40: 'new_sensor_refit', 41: 'sensor_arrival', 42: 'prompt_and_scratch'}
import analyze_g29 as aggregate

REGIONS = ('PEMS03', 'PEMS06', 'PEMS08', 'PEMS10', 'PEMS11', 'PEMS12')
PANELS = [(d, y) for d in REGIONS for y in (2020, 2025)]
ARMS = [('Old checkpoint',37,'stale5o'), ('Output head',37,'fit_head'),
        ('Shared weights',37,'fit_shared'), ('Added node prompt',42,'fit_prompt'),
        ('Recent-data scratch',42,'fit_scratch'), ('Identity refresh (EvoST)',37,'fit_emb_sel'),
        ('Identities + prompt',42,'fit_emb_prompt'), ('All parameters',37,'fit_all')]

def summary(n):
    return json.loads((ROOT / f'results/evost/{FOLDER[n]}/summary.json').read_text())

def load(n, prefix=''):
    data = collections.defaultdict(dict)
    for path in sorted((ROOT / f'results/evost/{FOLDER[n]}/runs').glob(prefix+'*/**/e3_metrics.jsonl')):
        done = json.loads(path.with_name('completed.json').read_text())
        assert done['status'] == 'completed' and not done.get('bounded_smoke'), path
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r['portion'] != 'test':
                continue
            key = (r['dataset'], r['year'], r['arm'], r['cohort'])
            assert r['seed'] not in data[key], ('duplicate measurement', path, key)
            data[key][r['seed']] = r
    return data

def value(data, panel, arm, cohort='all', metric='H12_MAE'):
    rows = data[(*panel, arm, cohort)]
    assert sorted(rows) == [51,52,53], (panel, arm, cohort, sorted(rows))
    return statistics.mean(r['calibrated'][metric] for r in rows.values())

def ratio(data, arm, reference='same', cohort='all', panels=PANELS, metric='H12_MAE'):
    rows = [(p, {'ratio':value(data,p,arm,cohort,metric)/value(data,p,reference,cohort,metric)}) for p in panels]
    return aggregate.geometric(rows, 'ratio')

def check(actual, expected):
    for key in ('gm','lo','hi'):
        assert abs(actual[key]-expected[key]) < 1e-12, (key,actual[key],expected[key])

def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'

def verify_horizon_sums(n, data):
    count=0
    for line in (ROOT/f'results/evost/{FOLDER[n]}/horizon_sums.jsonl').read_text().splitlines():
        r=json.loads(line)
        metric=data[(r['dataset'],r['year'],r['arm'],r['cohort'])][r['seed']]
        for kind in ('calibrated','uncalibrated'):
            cumulative=[0.,0.,0.]
            maes=[]
            for h,values in enumerate(r[kind],1):
                cumulative=[a+b for a,b in zip(cumulative,values)]
                mae=cumulative[0]/cumulative[2]
                maes.append(mae)
                if h in (3,6,12):
                    assert abs(mae-metric[kind][f'H{h}_MAE'])<1e-10
                    assert abs(math.sqrt(cumulative[1]/cumulative[2])-metric[kind][f'H{h}_RMSE'])<1e-10
            assert abs(statistics.mean(maes)-metric[kind]['Avg_MAE'])<1e-10
        count+=1
    assert count==(288 if n==37 else 252), count

def build():
    data = {n:load(n) for n in (37,42)}
    for n, rows in data.items():
        verify_horizon_sums(n, rows)
    sums = {n:summary(n) for n in (34,35,36,37,38,40,41,42)}
    text = '''# EvoST results

The primary experiment uses five-year-old STAEFormer checkpoints on twelve PeMS region-year settings. All maintenance arms and the target-year reference use delayed-feedback RLS calibration. Ratios below one favor the maintained model. [Protocol](evost_protocol.md) · [source and run commands](../experiments/evost/README.md) · [measurement records](../results/evost).

## Which parameters require maintenance?

The intervals use 10,000 district bootstrap draws after averaging three seeds within each setting. The prompt and scratch controls use the same recent observations and target-year reference as the parameter-update comparison.

'''
    rows = []
    for label,n,arm in ARMS:
        actual=ratio(data[n],arm)
        check(actual,sums[n]['primary' if n==37 else 'test'][arm+' / same'])
        rows.append([label,f"{actual['gm']:.3f}",f"[{actual['lo']:.3f}, {actual['hi']:.3f}]"])
    text += table(['Updated parameters / control','Relative H12 MAE','95% interval'],rows)
    old=sums[37]['primary']['stale5o / same']['gm']
    refreshed=sums[37]['primary']['fit_emb_sel / same']['gm']
    full=sums[37]['primary']['fit_all / same']['gm']
    text += f'Identity refresh reduces the gap to the calibrated target-year model from {100*(old-1):.1f}% to {100*(refreshed-1):.1f}%. It captures {100*(old-refreshed)/(old-full):.0f}% of the improvement from full-parameter fine-tuning on the same recent data.\n\n'
    text += '## Absolute MAE over the one-hour forecast horizon\n\nCumulative MAE across steps 1–12, averaged over seeds 51, 52, and 53. The target-year model is trained separately; recent-data scratch uses only the maintenance history.\n\n'
    columns=[(37,'same'),(37,'stale5o'),(37,'fit_shared'),(42,'fit_prompt'),(42,'fit_scratch'),(37,'fit_emb_sel'),(37,'fit_all')]
    rows=[]
    for panel in PANELS:
        # The shared reference and identity control must agree across the two studies.
        for arm in ('same','stale5o','fit_emb_sel'):
            assert abs(value(data[37],panel,arm)-value(data[42],panel,arm)) < 1e-8
        rows.append([f'{panel[0]}/{panel[1]}']+[f'{value(data[n],panel,arm):.2f}' for n,arm in columns])
    text+=table(['Region/year','Target-year','Old','Shared','Prompt','Scratch','EvoST','Full FT'],rows)
    text+='## Other horizons and metrics\n\nRatios use the matching metric of the calibrated target-year reference. H3, H6, and H12 aggregate steps 1–3, 1–6, and 1–12; Avg averages the twelve cumulative-horizon MAEs. The committed horizon sums independently verify this convention.\n\n'
    rows=[]
    for label,n,arm in ARMS:
        vals=[ratio(data[n],arm,metric=m)['gm'] for m in ('H3_MAE','H6_MAE','H12_MAE','H12_RMSE','Avg_MAE')]
        rows.append([label]+[f'{v:.3f}' for v in vals])
    text+=table(['Control','H3 MAE','H6 MAE','H12 MAE','H12 RMSE','Avg MAE'],rows)
    text+='## Additional architectures\n\nSixteen region-year settings per architecture. STAEFormer uses fixed-rate refresh in this extension. Each ratio uses that architecture\'s calibrated target-year model. Recovery is the fraction of its old-to-reference error gap closed.\n\n'
    rows=[]
    for name,n in [('STAEFormer',34),('STID',35),('Graph WaveNet',36)]:
        s=sums[n]['idr']['Eightcalibrated'] if n==34 else sums[n]['primary']
        a,b=s['stale5o / same']['gm'],s['fit_emb / same']['gm']
        rows.append([name,'16',f'{a:.3f}',f'{b:.3f}',f'{100*(a-b)/(a-1):.0f}%'])
    text+=table(['Backbone','Settings','Old','Identity refresh','Gap recovered'],rows)
    text+='## Checkpoint age\n\n![Checkpoint-age results](assets/evost_checkpoint_age.png)\n\n[Vector figure](assets/evost_checkpoint_age.pdf) · [editable data and plot code](../figures/evost). The age groups contain different sets of region-year settings.\n\n'
    text+='## Recent-history budget\n\nFit windows include the identity holdout. Selection uses an additional validation prefix for calibration. The identity learning rate is fixed in this study. Ratios use the calibrated target-year reference at the same budget.\n\n'
    bud=sums[38]['bud']['Primary (six districts x 2020, 2025), test portion, with the calibrator']
    text+=table(['Fit windows','Selection windows','Old','Identity refresh'],[[int(b),int(b)//2,f"{r['stale5o / same']['gm']:.3f}",f"{r['fit_emb / same']['gm']:.3f}"] for b,r in bud.items()])
    text+='## Sensor arrival and joint refitting\n\nThese results use the test portion after the scheduled event. New and all-sensor errors are divided by the calibrated target-year reference. Established-sensor errors are divided by the mature model without arrivals. Arrival history and the shared-scale convention are specified in the [protocol](evost_protocol.md#arrival-and-joint-refitting).\n\n'
    rows=[]
    for name,n,arm,prefix in [('Borrow and keep',40,'c64_prof0','hnd-'),('Refit new identities',41,'s_prof0_r','hnd-'),('Refit all (EvoST)',41,'a_prof0_r','hnd-')]:
        ds=load(n,prefix)
        s=sums[n] if n==40 else sums[n]['hnd']
        vals=[]
        for cohort,denom in [('new','same'),('all','same'),('old','konly')]:
            actual=ratio(ds,arm,reference=denom,cohort=cohort)
            check(actual,s[cohort+'_test'][arm+' / '+denom])
            vals.append(f"{actual['gm']:.3f}")
        rows.append([name]+vals)
    text+=table(['Strategy','New','All','Established'],rows)
    text+='## Fitting cost\n\nMedian costs from the primary STAEFormer maintenance study on H100 runs. Search time includes three candidate learning rates. These measurements cover fitting, excluding RLS calibration.\n\n'
    rows=[]
    for name,arm in [('Identity refresh','fit_emb_sel'),('Full fine-tuning','fit_all'),('Shared weights','fit_shared'),('Output head','fit_head')]:
        c=sums[37]['costs']['arms'][arm]
        rows.append([name,f"{c['trained_values']/1e6:.2f}M",f"{c['seconds_per_epoch']:.1f}",f"{c['seconds']:.0f}",f"{c['peak_gpu_gib']:.1f}"])
    text+=table(['Updated group','Trainable values','Seconds/epoch','Search seconds','Peak GiB'],rows)
    text+='## Complete study records\n\nThe saved summaries include calibrated and uncalibrated comparisons, established/new-sensor cohorts, and the additional diagnostic branches. Primary, secondary, and development settings keep their original labels.\n\n'
    for n,title in [(33,'Fixed-rate refresh and donors'),(34,'Districts and checkpoint ages'),(35,'STID'),(36,'Graph WaveNet'),(37,'Parameter groups'),(38,'History budgets and initialization'),(39,'Time-input sensitivity'),(40,'New-identity refitting'),(41,'Joint refitting'),(42,'Prompt and scratch controls')]:
        text+=f'- [{title}: summary](../results/evost/{FOLDER[n]}/summary.json) · [per-run records](../results/evost/{FOLDER[n]}/runs)\n'
    text+='\nThe [annual benchmark](evost_annual.md) reports EvoST and its matched controls on the same-year checkpoints of ten networks.\n'
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='verify measurements and require the committed page to be current')
    args=p.parse_args()
    result=build()
    out=ROOT/'docs/evost_results.md'
    if args.check:
        assert out.read_text()==result, 'Result page is stale; run without --check.'
    else:
        out.write_text(result)
    print('Primary ratios, district intervals, shared controls, and arrival ratios verified; result page is current.')

if __name__=='__main__':
    main()
