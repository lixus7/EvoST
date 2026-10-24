"""EvoST round 001, annual row of Table 2: EvoST and its matched frozen / frozen+RLS controls on the same-year checkpoint.

One run = one cell (dataset, year, evaluation seed) of a bundle written by export_annual.py: layout 'g30-stae' with only
the same-year checkpoint, the 0C manifest files (checkpoint seed = evaluation seed - 9). The g37 modules are used
unchanged (e3_confirm, e5_stae, e10_refresh, e15_maintain, g28_common, stae_model). This file adds the bundle class
without a five-year-old checkpoint, the arms, the reference check without an assertion, the test-only stream (D2), the
benchmark metrics and the record.

Arms (base forecast; every arm is scored without and with the per-sensor RLS calibrator):
  same    the same-year checkpoint as it is            uncalibrated = frozen        calibrated = frozen+RLS
  evost   the same-year checkpoint with only its        uncalibrated = identity      calibrated = EvoST, the new
          embedding table (12 x sensors x 80) refitted                 refit only                row of Table 2
          on the last (at most) 2,048 train windows, whose last 512 window starts are held out; rate chosen from
          e15_maintain.EMB_RATES on the holdout, i.e. g37's fit_emb_sel applied to the same-year checkpoint
          (e15_maintain.MaintainBundle.tune)
--mode regression replaces evost by g37's fit_same (SAME_RATES) to reproduce a g37 cell; never a result.

Protocol (D2). e3_confirm.run, unchanged: calibrator fitted by exact ridge on the last (at most) 2,048 train windows,
(ridge, forget) chosen per arm prequentially on the first min(1,024, validation windows) validation windows, then one
continuous validation + test stream; its rows are kept as the side output (e3_metrics.jsonl, g37's protocol). Main
result, added here: with the chosen (ridge, forget) a new calibrator is fitted again on the train tail and streams over
the test segment only (labels released after 12 windows, absorbed every 64). Validation labels serve the selection of
(ridge, forget) only; no calibrator state of the validation stream reaches the test stream.

Metrics of the main result: the benchmark of Table 2 (vendor utils/metric.py as called by 0C): elementwise mask
target != 0 (non-finite targets were set to 0, as in the cached FastData targets), cumulative horizons 1..12, single
steps 1..12, Avg = mean of the 12 cumulative values; MAE, RMSE, MAPE x 100; all sensors of the test segment;
accumulated in float64 sums per sensor and horizon. Non-finite forecasts (D7), see NONFINITE_POLICY: 'benchmark'
counts a non-finite forecast on a valid target as error 0 in the denominator (the NaN path of masked_*_np; for +-inf
this differs from the vendor code), 'strict' is null for a metric whose range contains one; counts are kept for all
forecasts, valid targets and masked targets; numeric_valid says whether the strict metrics exist.

Gates (a failed gate makes the cell 'rejected'; every output is written either way):
  checkpoint   the bundle's recorded checkpoint SHA-256, the manifest SHA-256 and the relative path equal the 0C
               manifest entry of this cell (the original .pkl was hashed by the exporter before loading)
  train_stats  the bundle's input mean and std equal the 0C B path's: |d| <= 1e-10 |ref| + 1e-12 each
  zero_c_B     'same' uncalibrated, complete test segment, all sensors: all 75 values within the 0C screening line
               |new - old| <= 1e-4 + 1e-5 |old|
  reference    graphs with stored reference windows (<= 3,000 sensors): the g37 run-time rule
               max |standalone - original| <= 1e-3 (1 + max |original|) on the two windows; 'not_applicable' otherwise
A cell is 'accepted' when checkpoint, train_stats and zero_c_B pass and reference passes or is not applicable. Bounded
runs (--max-windows) are 'bounded_smoke' and never accepted. --role formal requires the cases file, the complete
segments, the annual mode and none of the test-only overrides of the g37 modules (OVERRIDES).
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # deterministic CUDA matrix products, as g37
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import e15_maintain as e15
import g28_common as g

VERSION = 'e24-annual-2026-10-24c'     # c: zero-label fallback (PEMS06-2005-OPTIONS.md, option A); otherwise identical to b
LABELS = [str(i) for i in range(1, 13)] + ['step%d' % i for i in range(1, 13)] + ['Avg']
SCREEN_ABS, SCREEN_REL = 1e-4, 1e-5
RUNTIME_RULE = 1e-3                     # g30 / g37 run-time reference rule: max|d| <= 1e-3 (1 + max|ref|)
REQUIRED = ('checkpoint', 'train_stats', 'zero_c_B', 'reference')
OVERRIDES = ('G33_FIT_EPOCHS', 'G30_SUB', 'G30_CACHE_GIB', 'G37_ARMS')          # test-only knobs of the g37 modules
FORMAL_EPOCHS, FORMAL_PATIENCE, FORMAL_HOLDOUT = 16, 3, 512
MODES = dict(annual=dict(evost=('same', ('table',), e15.EMB_RATES)),
             regression=dict(fit_same=('same', ('table',), e15.SAME_RATES)))
NONFINITE_POLICY = dict(
    benchmark='a non-finite forecast on a valid target counts as error 0 and stays in the denominator (the NaN path of vendor '
              'masked_*_np); +-inf is treated the same way, which is not what the vendor code does with inf',
    strict='a metric whose range holds a non-finite forecast on a valid target is null in this JSON; null must never be read as 0',
    counts='nonfinite_forecasts_total (all forecast positions), _on_valid_targets, _on_masked_targets',
    numeric_valid='true when no forecast on a valid target is non-finite, i.e. every strict metric exists')


class Arrays:
    """The bundle's arrays, each read once: StaeBundle keys its forecast cache by id() of the raw split, so the
    test-only stream must receive the very objects e3_confirm.run used (and never a new object at a reused address)."""

    def __init__(self, z):
        self._z, self._memo, self.files = z, {}, z.files

    def __getitem__(self, key):
        if key not in self._memo:
            self._memo[key] = self._z[key]
        return self._memo[key]


class AnnualBundle(e15.MaintainBundle):
    def __init__(self, path):
        e5.StaeBundle.__init__(self, path)             # RefreshBundle.__init__ would require a five-year-old checkpoint
        assert self.meta.get('annual') and self.ckpt_years == [self.year], (self.meta.get('annual'), self.ckpt_years)
        self.z = Arrays(self.z)
        self.path = Path(path).resolve()
        self.old = None
        self._variant, self._donors, self.info = {}, None, {}
        self._tuned, self.peak_seen = {}, 0.
        self.zero_label = None                         # e24-annual c: set by main() only when the fitting window has no valid label
        e10.RefreshBundle.last = self

    def tuned(self, name, device):
        """e24-annual c: with no valid label in the fitting window the arm keeps the checkpoint identities and no candidate is fitted;
        otherwise exactly e15_maintain.MaintainBundle.tuned."""
        if self.zero_label is None or name in self._tuned:
            return super().tuned(name, device)
        which, _groups, _rates = e15.ARMS[name]
        ckpt = self.old if which == 'old' else self.year
        model = self.model(ckpt, device)
        table = e5.StaeBundle.table(self, ckpt, device)
        self._tuned[name] = (model, table)
        self.info[name] = dict(skipped='no_valid_fit_label', identities='checkpoint', valid_fit_targets=0, candidates=None, lr=None,
                               selected_lr=None, best_epoch=None, epochs_run=None, seconds=0., peak_gpu_gib=None,
                               window_start=self.zero_label['window_start'], window_count=self.zero_label['window_count'])
        return self._tuned[name]

    def adjacency(self, device):
        return None


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, variant: dict(name=name, base='stae', ckpt=B.year, norm=B.year, age=0, variant=variant)
    return [arm('same', 'full')] + [arm(name, name) for name in e15.ARMS]


def check_reference(B, seed, arms, weights, adj, device):
    """e5_stae.check_reference without its assertion: the same comparison, returned as a record. A mismatch is a
    rejection of the cell decided in main(), not an error that would end the task and invite a rerun."""
    index = B.z['ref_index']
    if len(index) == 0:
        return dict(applicable=False, reason='no reference windows stored for this graph size (more than 3,000 sensors)')
    raw = B.z['raw_test']
    x_raw = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in index])
    report = {}
    for arm in arms:
        if arm['base'] != 'stae' or arm.get('variant') != 'full':
            continue
        name = 'ref/%d/y%d/%s' % (seed, arm['ckpt'], 'year' if arm['norm'] == B.year else 'own')
        if name not in B.z.files:
            report[arm['name']] = dict(passed=False, reason='%s missing from the bundle' % name)
            continue
        ref = torch.as_tensor(B.z[name], device=device)
        mine = B.forward(e5.key_of(arm), x_raw, device).to(device)
        worst, scale = float((mine - ref).abs().max()), float(ref.abs().max())
        threshold = RUNTIME_RULE * (1 + scale)
        report[arm['name']] = dict(max_abs_difference=worst, reference_abs_max=scale, threshold=threshold, ratio=worst / threshold,
                                   windows=[int(i) for i in index], passed=bool(worst <= threshold))       # NaN compares False
    return dict(applicable=True, passed=bool(report) and all(v['passed'] for v in report.values()), arms=report)


e3.Bundle = AnnualBundle
e3.arms_of = arms_of
e3.check_reference = check_reference


class BenchAcc:
    """Per virtual node and horizon float64 sums: absolute, squared and absolute percentage error over valid targets,
    the count of valid targets, and the counts of non-finite forecasts on valid targets and at all positions."""

    KEYS = ('abs', 'sq', 'ape', 'cnt', 'bad', 'bad_all')

    def __init__(self, n, device, h=g.Y_LEN):
        for k in self.KEYS:
            setattr(self, k, torch.zeros(n, h, dtype=torch.float64, device=device))

    def add(self, pred, y):
        """pred, y [..., n, h]; leading dims are summed."""
        valid = g.target_valid(y)
        finite = torch.isfinite(pred)
        ok = valid & finite
        zero = torch.zeros((), dtype=torch.float64, device=y.device)
        e = torch.where(ok, (pred - y).to(torch.float64), zero)
        ape = torch.where(ok, e.abs() / torch.where(valid, y.abs().to(torch.float64), torch.ones((), dtype=torch.float64, device=y.device)), zero)
        extra = tuple(range(e.dim() - 2))
        total = (lambda v: v.sum(extra)) if extra else (lambda v: v)
        self.abs += total(e.abs())
        self.sq += total(e * e)
        self.ape += total(ape)
        self.cnt += total(valid.to(torch.float64))
        self.bad += total((valid & ~finite).to(torch.float64))
        self.bad_all += total((~finite).to(torch.float64))

    def arrays(self, sl=slice(None)):
        return {k: getattr(self, k)[sl].cpu().numpy() for k in self.KEYS}


def bench_metrics(s):
    """75 values {label: [MAE, RMSE, MAPE]} from per-node per-horizon sums (benchmark and strict conventions)."""
    a, q, p, c, b = (s[k].sum(0) for k in ('abs', 'sq', 'ape', 'cnt', 'bad'))
    out = dict(benchmark={}, strict={})
    cum = dict(benchmark=[], strict=[])
    for i in range(1, 13):
        for label, lo in ((str(i), 0), ('step%d' % i, i - 1)):
            n, bad = c[lo:i].sum(), b[lo:i].sum()
            v = [a[lo:i].sum() / n, np.sqrt(q[lo:i].sum() / n), 100. * p[lo:i].sum() / n] if n > 0 else [np.nan] * 3
            out['benchmark'][label] = [float(x) for x in v]
            out['strict'][label] = [float(x) if bad == 0 else float('nan') for x in v]
        for conv in cum:
            cum[conv].append(out[conv][str(i)])
    for conv in cum:
        out[conv]['Avg'] = [float(np.mean([v[j] for v in cum[conv]])) for j in range(3)]
    return out


def screen(mine, ref):
    """The 0C screening line on all 75 values; every value outside it is listed with its difference, line and ratio."""
    within, items, failures = 0, [], []
    for label in LABELS:
        for j, name in enumerate(('MAE', 'RMSE', 'MAPE')):
            o, n = float(ref[label][j]), float(mine[label][j])
            line = SCREEN_ABS + SCREEN_REL * abs(o)
            d = abs(n - o)
            ok = d <= line                                              # NaN compares False
            within += ok
            item = dict(label=label, metric=name, reference=o, new=n, difference=d, line=line, ratio=d / line)
            items.append(item)
            if not ok:
                failures.append(item)
    worst = max(items, key=lambda v: v['ratio'] if np.isfinite(v['ratio']) else np.inf)
    return dict(values=len(items), within=within, all_within=within == 75, max_abs_diff=max(v['difference'] for v in items),
                max_line_use=worst['ratio'], worst=worst, failures=failures)


def nan_to_none(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    return obj


def test_only(B, seed, device, max_windows, arms, chosen):
    """D2 main stream: a new calibrator fitted on the train tail with the chosen (ridge, forget), test segment only."""
    n, z = B.n, B.z
    train, test = z['raw_train'], z['raw_test']
    fit_n = min(e3.FIT_WINDOWS, g.n_windows(train))
    fit_piece = (train, g.n_windows(train) - fit_n, fit_n)
    n_test = g.n_windows(test) if not max_windows else min(max_windows, g.n_windows(test))
    _, y_fit = g.get_windows(train, fit_piece[1], fit_n)
    y_fit = g.clean_y(y_fit)
    zero_label = int(g.target_valid(y_fit).sum()) == 0                    # e24-annual c trigger, as e3_confirm.run
    if zero_label:
        assert e3.ZERO_LABEL is not None, 'no valid label in the fitting window and no fallback scale was set'
        scale = e3.ZERO_LABEL['scale']
    else:
        scale = float(y_fit[g.target_valid(y_fit)].std())                     # as e3_confirm.run
    del y_fit
    r = len(arms)
    ridge = torch.tensor([chosen[a['name']][0] for a in arms], dtype=torch.float64).repeat_interleave(n)
    forget = torch.tensor([chosen[a['name']][1] for a in arms], dtype=torch.float64).repeat_interleave(n)
    _, fn = e3.make_chunks(B, None, None, device, arms, [fit_piece])         # e3_confirm.run's fitted(), unchanged
    model = g.NodeRLS(n * r, g.Y_LEN, scale, ridge, forget, device)
    step = 256 if n * r <= 24000 else 64
    for s in (range(0, fit_n, step) if not zero_label else ()):          # e24-annual c: no fit, a fresh prior for the test stream
        x, b, y, valid, _ = fn(slice(s, min(s + step, fit_n)))
        model.fit(x, b, y, valid)
    T, fn = e3.make_chunks(B, None, None, device, arms, [(test, 0, n_test)])
    cal, raw = BenchAcc(n * r, device), BenchAcc(n * r, device)
    chunk = e3.chunk_of(n * r)
    g.rls_stream(model, T, fn, e3.DELAY, e3.CADENCE, lambda t, pred, y, count: cal.add(pred, y), chunk=chunk)
    for start in range(0, T, chunk):                                         # uncalibrated forecasts of the same windows
        x, b, y, _, _ = fn(slice(start, min(start + chunk, T)))
        raw.add(b, y)
    out, sums = {}, {}
    for i, arm in enumerate(arms):
        sl = slice(i * n, (i + 1) * n)
        out[arm['name']] = {}
        for kind, acc in (('calibrated', cal), ('uncalibrated', raw)):
            s = acc.arrays(sl)
            m = bench_metrics(s)
            bad_valid, bad_all = int(s['bad'].sum()), int(s['bad_all'].sum())
            out[arm['name']][kind] = dict(benchmark=m['benchmark'], strict=m['strict'],
                                          g28=g.summarize(s['abs'], s['sq'], s['cnt']),
                                          nonfinite_forecasts_total=bad_all, nonfinite_forecasts_on_valid_targets=bad_valid,
                                          nonfinite_forecasts_on_masked_targets=bad_all - bad_valid,
                                          forecasts=int(n * g.Y_LEN * T), valid_targets=int(s['cnt'].sum()), numeric_valid=bad_valid == 0)
            for k, v in s.items():
                sums['%s/%s/%s/node' % (arm['name'], kind, k)] = v.sum(1)
                sums['%s/%s/%s/horizon' % (arm['name'], kind, k)] = v.sum(0)
        worse = {}
        for j, name in enumerate(('MAE', 'RMSE', 'MAPE')):
            c, u = out[arm['name']]['calibrated']['benchmark']['Avg'][j], out[arm['name']]['uncalibrated']['benchmark']['Avg'][j]
            worse[name] = bool(c > u)
        out[arm['name']]['calibrated_worse_than_uncalibrated_Avg'] = worse
        out[arm['name']]['ridge'], out[arm['name']]['forget'] = chosen[arm['name']]
    return out, sums, dict(windows=int(n_test), fit_windows=int(fit_n), scale=scale, chunk=chunk)


def formal_problems(opt):
    """Why this invocation cannot be a formal run (empty when it can)."""
    problems = ['%s is set' % k for k in OVERRIDES if k in os.environ]
    if not opt.cases:
        problems.append('no cases file')
    if opt.max_windows:
        problems.append('--max-windows %d' % opt.max_windows)
    if opt.mode != 'annual':
        problems.append('mode %s' % opt.mode)
    if (e10.FIT['epochs'], e10.FIT['patience'], e10.FIT['holdout']) != (FORMAL_EPOCHS, FORMAL_PATIENCE, FORMAL_HOLDOUT):
        problems.append('fit protocol %s' % e10.FIT)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', default='', help='cases-annual.json: 0C manifest entry and B path of every cell (gates)')
    ap.add_argument('--mode', default='annual', choices=sorted(MODES))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-windows', type=int, default=0, help='bounded smoke only; 0 = complete split')
    ap.add_argument('--role', default='unspecified', help='smoke | regression | formal (formal is checked, see formal_problems)')
    opt = ap.parse_args()
    if opt.role == 'formal' and formal_problems(opt):
        print('REFUSED formal run: ' + '; '.join(formal_problems(opt)), flush=True)
        return 2
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if opt.device == 'auto' else opt.device))
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)          # as e15_maintain, so that a run can be repeated
    e15.ARMS = dict(MODES[opt.mode])
    started = time.time()
    B = AnnualBundle(opt.bundle)
    assert opt.seed in B.meta['seeds'], (opt.seed, B.meta['seeds'])
    tr = B.z['raw_train']                                              # e24-annual c: the zero-label trigger (read only)
    fit_first, fit_count = g.n_windows(tr) - min(e3.FIT_WINDOWS, g.n_windows(tr)), min(e3.FIT_WINDOWS, g.n_windows(tr))
    if int(g.target_valid(g.clean_y(g.get_windows(tr, fit_first, fit_count)[1])).sum()) == 0:
        fallback_scale = float(B.norm[B.year][1])                      # training-split input std of the year, as the 0C B path
        assert np.isfinite(fallback_scale) and fallback_scale > 0, fallback_scale
        B.zero_label = dict(window_start=int(fit_first), window_count=int(fit_count),
                            mask='g28_common.target_valid(clean_y(y)): finite and non-zero', scale=fallback_scale,
                            scale_source='meta norm[year].std: training-split input standard deviation (not a fitting-label statistic)')
        e3.ZERO_LABEL = dict(scale=fallback_scale, source=B.zero_label['scale_source'])
    cell = '%s/%d/s%d' % (B.meta['dataset'], B.year, opt.seed)
    case = None
    if opt.cases:
        case = {c['cell']: c for c in json.loads(Path(opt.cases).read_text())}[cell]
    with torch.no_grad():
        began = time.time()
        rows, arms, reference, extra = e3.run(B, opt.seed, device, opt.max_windows, out)    # selection + val/test replay
        replay_seconds = time.time() - began
        selection = json.loads((out / 'e3_selection.json').read_text())['selection']
        chosen = {name: (v['chosen']['ridge'], v['chosen']['forget']) for name, v in selection.items()}
        began = time.time()
        main_result, sums, stream = test_only(B, opt.seed, device, opt.max_windows, arms, chosen)
        stream['seconds'] = time.time() - began
    with (out / 'e3_metrics.jsonl').open('w') as f:                    # side output: g37's validation + test replay
        for row in rows:
            f.write(json.dumps(nan_to_none(row), allow_nan=False) + '\n')
    np.savez_compressed(out / 'e24_sums.npz', **sums)
    gates, stats_diff = {}, None
    ckpt = B.meta['checkpoints']['%d/y%d' % (opt.seed, B.year)]
    if case is not None:
        gates['checkpoint'] = bool(ckpt['sha256'] == case['sha256'] == ckpt['manifest_sha256'] and ckpt['rel'] == case['rel'])
        stats = case['zero_c']['train_stats']
        stats_diff = [abs(B.norm[B.year][0] - stats['mean']), abs(B.norm[B.year][1] - stats['std'])]
        gates['train_stats'] = bool(stats_diff[0] <= 1e-10 * abs(stats['mean']) + 1e-12 and stats_diff[1] <= 1e-10 * abs(stats['std']) + 1e-12)
        if not opt.max_windows:
            check = screen(main_result['same']['uncalibrated']['benchmark'], case['zero_c']['B'])
            gates['zero_c_B'] = check['all_within']
        else:
            check = dict(not_applicable='bounded smoke: the 0C B path covers the complete test segment')
    else:
        check = dict(not_applicable='no cases file given')
    gates['reference'] = reference['passed'] if reference['applicable'] else 'not_applicable'
    passed = {k: gates.get(k) is True or (k == 'reference' and gates.get(k) == 'not_applicable') for k in REQUIRED}
    if opt.max_windows:
        status = 'bounded_smoke'
    else:
        status = 'accepted' if all(passed.values()) else 'rejected'
    peak = max(B.peak_seen, torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else None
    fit_protocol = dict(epochs=e10.FIT['epochs'], patience=e10.FIT['patience'], holdout_window_starts=e10.FIT['holdout'],
                        rates={name: list(v[2]) for name, v in e15.ARMS.items()}, groups={name: list(v[1]) for name, v in e15.ARMS.items()},
                        stride=e10.fit_stride(B.n), first_batch_windows=e10.fit_batch(B.n),
                        overrides_set=[k for k in OVERRIDES if k in os.environ])
    record = dict(version=VERSION, cell=cell, dataset=B.meta['dataset'], year=B.year, seed=opt.seed, nodes=B.n, mode=opt.mode, role=opt.role,
                  max_windows=opt.max_windows, status=status, gates=gates,
                  reasons=[] if opt.max_windows else [k for k in REQUIRED if not passed[k]],
                  not_evaluated=[k for k in REQUIRED if gates.get(k) is None or (gates.get(k) == 'not_applicable' and k != 'reference')],
                  zero_c_B_check=check, zero_c_record=None if case is None else {k: v for k, v in case['zero_c'].items() if k != 'B'},
                  train_stats_abs_diff=stats_diff, reference_check=reference,
                  checkpoint=ckpt, norm=dict(mean=B.norm[B.year][0], std=B.norm[B.year][1]), arms=arms, chosen=chosen,
                  main_result=main_result, main_stream=stream, nonfinite_policy=NONFINITE_POLICY, fit_protocol=fit_protocol, fits=B.info,
                  bundle=dict(path=str(B.path), bytes=B.path.stat().st_size, version=B.meta['annual']['version'],
                              train_windows=int(g.n_windows(B.z['raw_train'])), validation_windows=int(g.n_windows(B.z['raw_val'])),
                              test_windows=int(g.n_windows(B.z['raw_test']))),
                  timing=dict(replay_including_fit_and_selection=replay_seconds, select=extra['select_seconds'], replay_stream=extra['stream_seconds'],
                              test_only=stream['seconds'], total=time.time() - started),
                  provenance=dict(host=socket.gethostname(), entry_output_root=os.environ.get('EVOST_VIRGA_OUT'), argv=sys.argv[1:]),
                  device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, peak_gpu_gib=peak,
                  torch=torch.__version__, numpy=np.__version__)
    if B.zero_label is not None:                                       # e24-annual c: recorded only when the fallback ran
        pairs = {}
        for kind, (a, b) in (('uncalibrated', ('evost', 'same')), ('calibrated', ('evost', 'same'))):
            ea, eb = main_result[a][kind]['strict'], main_result[b][kind]['strict']
            both = [(x, y) for lab in LABELS for x, y in zip(ea[lab], eb[lab])]
            diffs = [float(abs(x - y)) for x, y in both if np.isfinite(x) and np.isfinite(y)]
            pairs[kind] = dict(max_abs_difference=max(diffs) if diffs else None, finite_pairs=len(diffs),
                               nonfinite_mismatch=int(sum(bool(np.isfinite(x)) != bool(np.isfinite(y)) for x, y in both)))
        record['zero_label_fallback'] = dict(B.zero_label, valid_fit_targets=0, identities='checkpoint (no candidate fit)',
                                             rls='prior (theta=0, P=I/ridge, count=0) in selection, replay and test; no train-tail fit',
                                             refresh_vs_frozen=pairs['uncalibrated'], evost_vs_frozen_rls=pairs['calibrated'])
    g.save_json(out / 'e24_record.json', nan_to_none(record))
    g.save_json(out / 'completed.json', dict(status='completed', cell_status=status, rows=len(rows), seconds=time.time() - started,
                                             bounded_smoke=bool(opt.max_windows), mode=opt.mode, role=opt.role))
    brief = dict(status='e24_completed', cell=cell, mode=opt.mode, cell_status=status, reasons=record['reasons'], nodes=B.n,
                 seconds=round(time.time() - started, 1), peak_gpu_gib=peak)
    for name in main_result:
        brief[name] = [main_result[name][k]['benchmark']['Avg'][0] for k in ('uncalibrated', 'calibrated')]
    if case is not None and not opt.max_windows:
        brief['zero_c_B_max_line_use'] = check['max_line_use']
    print(json.dumps(nan_to_none(brief)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
