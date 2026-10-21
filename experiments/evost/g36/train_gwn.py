"""g36: train the project's Graph WaveNet for one dataset, panel years 2015 / 2020 / 2025, with the project's recipe.

  python train_gwn.py --dataset PEMS08 --head HEAD_DIR --bundles BUNDLE_DIR --out OUT_DIR [--years 2015 2020 2025] [--seeds 42 43 44]

The repository keeps Graph WaveNet checkpoints for the last year only, so the per-year models that the identity-refresh
experiment needs are trained here: the project's model class (gwn_model.py, verbatim), its optimisation recipe as read from
`src/trainer/default_trainer.py` in the baseline repository and the per-dataset configuration `conf/<DATASET>/retrain_gwn_<dataset>.json`:
AdamW, learning rate / loss / warm-up / gradient clipping of the configuration, batches of 128 shuffled windows, at most
100 epochs, the epoch with the lowest validation error kept, stop after more than 5 epochs without improvement. As in the
repository's trainer the model stays in training mode while the validation error is computed (BatchNorm then uses batch
statistics and its running statistics also see the validation inputs); forecasts are made in evaluation mode.

Differences from the repository run, all recorded in the output:
  - inputs of every split are normalised with the statistics of the train windows (the convention of the paper's
    pipeline, g29 onwards); the repository normalises validation and test inputs with their own statistics. Training
    inputs are the same in both.
  - shuffling uses torch.randperm; runs are not bit-comparable with the repository's.
For 2025 the repository's own checkpoint is evaluated on the same windows as a reference (not a result); with the
repository's normalisation of the test inputs it must reproduce the test error in the repository's log.

Data: <HEAD_DIR>/<DATASET>_<year>_head.npz (train rows before the tail) + the tail, validation and test rows of the g30
bundle <BUNDLE_DIR>/stae/<DATASET>_<year>.npz, checked to attach (hash of the tail, input statistics); the static
adjacency of that year from the g29 bundle <BUNDLE_DIR>/gcn/<DATASET>_<year>.npz (row-normalised as in the repository).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as func
import g28_common as g
import gwn_model as gm

PATIENCE, HORIZONS = 5, (3, 6, 12)


def seed_anything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False                             # as utils.initialize.seed_anything of the repository
    torch.backends.cudnn.deterministic = True


class Split:
    """Windows of one raw split on the device: x normalised [.., N, 12], y raw with non-finite values set to 0."""

    def __init__(self, raw, mean, std, device):
        rows = torch.as_tensor(np.asarray(raw, np.float32), device=device)
        self.y = torch.nan_to_num(rows, nan=0., posinf=0., neginf=0.)
        self.x = torch.nan_to_num((rows - float(mean)) / float(std), nan=0., posinf=0., neginf=0.)
        self.count = rows.shape[0] - g.SPAN
        self.step = torch.arange(g.X_LEN, device=device)

    def batch(self, idx):
        rows = idx[:, None] + self.step[None]                        # [B, 12]
        return self.x[rows].transpose(1, 2), self.y[rows + g.X_LEN].transpose(1, 2)


def masked_mae(y, pred):
    """utils.metric.masked_mae_np(y, pred, 0): mean absolute error over targets that are not 0."""
    valid = y != 0
    return float(((y - pred).abs() * valid).sum() / valid.sum().clamp_min(1)), int(valid.sum())


def validation_loss(model, split, batch, adj):
    """Mean over batches of the masked MAE, as the trainer computes it (the model is left in training mode, as there)."""
    total, n = 0., 0
    with torch.no_grad():
        for s in range(0, split.count, batch):
            x, y = split.batch(torch.arange(s, min(s + batch, split.count), device=split.x.device))
            value, valid = masked_mae(y, gm.forward_original(model, x, adj))
            if valid and np.isfinite(value):
                total, n = total + value, n + 1
    return total / n if n else float('inf')


def test_metrics(predict, split, batch=128):
    """utils.metric.cal_metric: masked MAE / RMSE over horizons 1..i for i = 3, 6, 12, and the mean of the 12 cumulative MAEs."""
    abs_sum = torch.zeros(g.Y_LEN, dtype=torch.float64, device=split.x.device)
    sq_sum, cnt = abs_sum.clone(), abs_sum.clone()
    with torch.no_grad():
        for s in range(0, split.count, batch):
            x, y = split.batch(torch.arange(s, min(s + batch, split.count), device=split.x.device))
            err, valid = (predict(x) - y).to(torch.float64), (y != 0)
            abs_sum += (err.abs() * valid).sum((0, 1)); sq_sum += (err ** 2 * valid).sum((0, 1)); cnt += valid.sum((0, 1))
    ca, cs, cc = abs_sum.cumsum(0), sq_sum.cumsum(0), cnt.cumsum(0).clamp_min(1)
    cum = (ca / cc).cpu().numpy()
    out = dict(Avg_MAE=float(cum.mean()), H12_MAE_single=float(abs_sum[-1] / cnt[-1].clamp_min(1)))
    for h in HORIZONS:
        out['T%d_MAE' % h] = float(cum[h - 1])
        out['T%d_RMSE' % h] = float((cs[h - 1] / cc[h - 1]).sqrt())
    return out


def own_stats(raw):
    """Statistics of the input windows of a split (utils.data_convert.z_score on that split's x)."""
    v = np.asarray(g.window_views(np.asarray(raw, np.float32))[:g.n_windows(raw)], np.float64)
    mean, std = float(np.nanmean(v)), float(np.nanstd(v))
    return mean, (std if np.isfinite(std) and std >= 1e-8 else 1.)


def evaluate(state, dropout, splits, adj):
    """Test and validation metrics of a state dict in evaluation mode; splits = {name: Split}."""
    n = state[gm.E1].shape[0]
    model = gm.original(n, dropout).to(adj.device)
    model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    model.eval()
    return {name: test_metrics(lambda x: gm.forward_original(model, x, adj), split, 64) for name, split in splits.items()}


def adjacency(bundle_dir, dataset, year, ids, device):
    """Row-normalised static adjacency of a dataset-year, in the sensor order of the data bundle."""
    zg, _ = g.load_bundle(bundle_dir / 'gcn' / ('%s_%d.npz' % (dataset, year)))
    assert [str(s) for s in zg['sensor_ids']] == ids, 'the adjacency bundle lists other sensors'
    a = zg['adj_norm'] if 'adj_norm' in zg.files else g.normalize_adj(zg['adj'])
    a = torch.as_tensor(np.asarray(a, np.float32), device=device)
    assert a.shape == (len(ids), len(ids)) and bool(torch.isfinite(a).all())
    return a


def train_one(dataset, year, seed, conf, head_dir, bundle_dir, device, max_epochs):
    began = time.time()
    with np.load(head_dir / ('%s_%d_head.npz' % (dataset, year)), allow_pickle=False) as zh:
        head, hmeta = np.asarray(zh['raw_train_head'], np.float32), json.loads(str(zh['meta_json']))
    z, meta = g.load_bundle(bundle_dir / 'stae' / ('%s_%d.npz' % (dataset, year)))
    tail, val, test = (np.asarray(z[k], np.float32) for k in ('raw_train', 'raw_val', 'raw_test'))
    assert hashlib.sha256(np.ascontiguousarray(tail).tobytes()).hexdigest() == hmeta['tail_sha256'], 'the exported head does not belong to this bundle'
    train = np.concatenate([head, tail])
    assert train.shape == (meta['split_lengths']['train'], int(meta['nodes'])), (train.shape, meta['split_lengths'])
    mean, std = (float(v) for v in (meta['norm'][str(year)]['mean'], meta['norm'][str(year)]['std']))
    assert abs(mean - hmeta['mean']) <= 1e-6 * (1 + abs(mean)) and abs(std - hmeta['std']) <= 1e-6 * (1 + abs(std)), (mean, std, hmeta['mean'], hmeta['std'])
    n = train.shape[1]
    ids = [str(s) for s in z['sensor_ids']]
    adj = adjacency(bundle_dir, dataset, year, ids, device)
    dropout = float(conf.get('dropout') or 0.)
    assert dropout == 0.
    lossfunc = dict(mse=func.mse_loss, huber=func.smooth_l1_loss, mae=func.l1_loss)[conf['loss']]
    batch, epochs = int(conf['batch_size']), min(int(conf['epoch']), max_epochs)
    tr, va = Split(train, mean, std, device), Split(val, mean, std, device)

    seed_anything(seed)
    model = gm.original(n, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf['lr']))
    warm = int(conf.get('warmup_epochs') or 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: min(1., (ep + 1) / warm)) if warm > 0 else None
    clip = float(conf.get('grad_clip') or 0.)
    gen = torch.Generator().manual_seed(seed)
    lowest, counter, best, curve, skipped = 1e7, 0, None, [], 0
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(tr.count, generator=gen).to(device)
        total, cn = 0., 0
        for s in range(0, tr.count, batch):
            x, y = tr.batch(order[s:s + batch])
            optimizer.zero_grad()
            loss = lossfunc(y, gm.forward_original(model, x, adj), reduction='mean')
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            total, cn = total + float(loss), cn + 1
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        v = validation_loss(model, va, batch, adj)
        curve.append((total / cn if cn else float('nan'), v))
        if v <= lowest:
            counter, lowest = 0, round(v, 4)
            best = (epoch, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
        else:
            counter += 1
            if counter > PATIENCE:
                break
    assert best is not None and np.isfinite(lowest), 'training never produced a finite validation error'
    state = {k: v.numpy() for k, v in best[1].items()}                      # float32 weights; BatchNorm's batch counter stays an integer
    splits = dict(test_train_stats=Split(test, mean, std, device), test_own_stats=Split(test, *own_stats(test), device),
                  val_train_stats=va, val_own_stats=Split(val, *own_stats(val), device))
    info = dict(dataset=dataset, year=year, seed=seed, nodes=n, conf={k: conf.get(k) for k in ('lr', 'loss', 'dropout', 'warmup_epochs', 'grad_clip', 'batch_size', 'epoch')},
                epochs_run=len(curve), best_epoch=best[0], lowest_validation_loss=lowest, curve=curve, batches_skipped_nonfinite=skipped,
                train_windows=tr.count, val_windows=va.count, mean=mean, std=std, metrics=evaluate(state, dropout, splits, adj),
                standalone_max_abs_difference=gm.check_against_original(state, dropout),
                parameters=int(sum(v.size for k, v in state.items() if 'running_' not in k and 'num_batches' not in k)),
                seconds=time.time() - began, torch=torch.__version__, device=str(device))
    assert info['standalone_max_abs_difference'] <= 1e-4, info['standalone_max_abs_difference']
    return state, ids, info, splits, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--head', required=True, help='directory with <DATASET>_<year>_head.npz, <DATASET>_gwn2025.npz and gwn_conf.json')
    ap.add_argument('--bundles', required=True, help='directory with stae/<DATASET>_<year>.npz and gcn/<DATASET>_<year>.npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--years', type=int, nargs='+', default=[2015, 2020, 2025])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-epochs', type=int, default=100, help='bounded smoke only')
    ap.add_argument('--resume', action='store_true', help='continue a run that was cut off: models already written to --out are kept, the others are trained')
    opt = ap.parse_args()
    out, head = Path(opt.out).resolve(), Path(opt.head).resolve()
    out.mkdir(parents=True, exist_ok=opt.resume)
    assert not (out / 'completed.json').exists(), 'this run is complete'
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else ('cpu' if opt.device == 'auto' else opt.device))
    conf = json.loads((head / 'gwn_conf.json').read_text())[opt.dataset]
    start, report = time.time(), dict(dataset=opt.dataset, runs={}, reference={})
    ref_path = head / ('%s_gwn2025.npz' % opt.dataset)
    for year in opt.years:
        for seed in opt.seeds:
            name = '%s_%d_s%d' % (opt.dataset, year, seed)
            if opt.resume and (out / (name + '.npz')).exists():        # written completely (the file is renamed into place)
                with np.load(out / (name + '.npz'), allow_pickle=False) as zd:
                    info = json.loads(str(zd['meta_json']))
                report['runs'][name] = {k: v for k, v in info.items() if k != 'curve'}
                print(json.dumps(dict(run=name, kept=True)), flush=True)
                continue
            state, ids, info, splits, adj = train_one(opt.dataset, year, seed, conf, head, Path(opt.bundles), device, opt.max_epochs)
            if year == 2025 and ref_path.exists():                     # the repository's own 2025 checkpoint on the same windows
                with np.load(ref_path, allow_pickle=False) as zr:
                    ref = {k[len('gwn/%d/' % seed):]: zr[k] for k in zr.files if k.startswith('gwn/%d/' % seed)}
                    logged = json.loads(str(zr['meta_json']))['seeds'].get(str(seed), {}).get('logged_test')
                if ref:
                    assert ref[gm.E1].shape[0] == info['nodes']
                    info['repository_checkpoint'] = dict(metrics=evaluate(ref, 0., splits, adj), logged_test=logged)
            np.savez(out / (name + '.partial.npz'), sensor_ids=np.asarray(ids), meta_json=np.asarray(json.dumps(info)), **{'gwn/' + k: v for k, v in state.items()})
            (out / (name + '.partial.npz')).replace(out / (name + '.npz'))
            report['runs'][name] = {k: v for k, v in info.items() if k != 'curve'}
            m = info['metrics']
            print(json.dumps(dict(run=name, nodes=info['nodes'], epochs=info['epochs_run'], best=info['best_epoch'], val=info['lowest_validation_loss'],
                                  test_T12=round(m['test_train_stats']['T12_MAE'], 3), test_T12_own_stats=round(m['test_own_stats']['T12_MAE'], 3),
                                  repository_T12_own_stats=(round(info['repository_checkpoint']['metrics']['test_own_stats']['T12_MAE'], 3) if 'repository_checkpoint' in info else None),
                                  seconds=round(info['seconds'], 1))), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
    report.update(seconds=time.time() - start, peak_gpu_gib=peak, bounded_smoke=bool(opt.max_epochs < 100), years=opt.years, seeds=opt.seeds,
                  gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)
    (out / 'train_report.json').write_text(json.dumps(report, indent=1) + '\n')
    g.save_json(out / 'completed.json', dict(status='completed', runs=len(report['runs']), seconds=time.time() - start, bounded_smoke=bool(opt.max_epochs < 100)))
    print(json.dumps(dict(status='train_gwn_completed', dataset=opt.dataset, runs=len(report['runs']), seconds=round(time.time() - start, 1), peak_gpu_gib=peak)), flush=True)


if __name__ == '__main__':
    main()
