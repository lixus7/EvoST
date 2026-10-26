"""g39: train the project's STAEFormer for one dataset, variant and panel year, with the project's recipe.

  python train_stae.py --dataset PEMS06 --variant tm --year 2025 --head HEAD_DIR --bundles BUNDLE_DIR --conf stae_conf.json --out OUT_DIR [--seeds 42 43 44]

Variants (stae_time.py): tm = with the original STAEFormer's time-of-day and day-of-week embeddings; nt = without them, i.e. the
project's model as the benchmark repository trains it. Both are trained here with the same code, data and recipe, so that
they differ only in the time inputs. The recipe is the project's, as read from `src/trainer/default_trainer.py` and the
per-dataset configuration `conf/<DATASET>/retrain_staeformer_<dataset>.json` exported from the baseline repository:
AdamW, learning rate, loss, gradient clipping, batch size and epochs of the configuration (lr 1e-3, MSE, batches of 16, at
most 100 epochs), the epoch with the lowest validation error kept, stop after more than 5 epochs without improvement.

Differences from the repository run, recorded in the output (as for STID in g35):
  - inputs of every split are normalised with the statistics of the train windows (the paper's convention); the
    repository normalises validation and test inputs with their own statistics;
  - shuffling uses torch.randperm; runs are not bit-comparable with the repository's.
For variant nt the repository's own checkpoint of the same year and seed (stored in the g30 bundle) is evaluated on the same
windows as a reference (not a result): retrained nt against repository says whether the recipe was reproduced.

Data: <HEAD_DIR>/<DATASET>_<year>_head.npz (train rows before the tail, exported for g35) + the tail, validation and test
rows of the g30 bundle <BUNDLE_DIR>/stae/<DATASET>_<year>.npz, checked to attach (hash of the tail, input statistics).
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
import stae_time as st
import stae_time_settings as settings

PATIENCE, HORIZONS = 5, (3, 6, 12)


def seed_anything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Split:
    """Windows of one raw split on the device: x normalised [.., N, 12], y raw with non-finite values set to 0, and the time of
    day / day of week of every input step (the split starts at row first_row of the year's matrix)."""

    def __init__(self, raw, mean, std, device, year, first_row):
        rows = torch.as_tensor(np.asarray(raw, np.float32), device=device)
        self.y = torch.nan_to_num(rows, nan=0., posinf=0., neginf=0.)
        self.x = torch.nan_to_num((rows - float(mean)) / float(std), nan=0., posinf=0., neginf=0.)
        self.count = rows.shape[0] - g.SPAN
        self.step = torch.arange(g.X_LEN, device=device)
        tod, dow = st.calendar(year, first_row + np.arange(rows.shape[0]))
        self.tod, self.dow = torch.as_tensor(tod, device=device), torch.as_tensor(dow, device=device)

    def batch(self, idx):
        rows = idx[:, None] + self.step[None]                        # [B, 12]
        return self.x[rows].transpose(1, 2), self.y[rows + g.X_LEN].transpose(1, 2), self.tod[rows], self.dow[rows]


def masked_mae(y, pred):
    """utils.metric.masked_mae_np(y, pred, 0): mean absolute error over targets that are not 0."""
    valid = y != 0
    return float(((y - pred).abs() * valid).sum() / valid.sum().clamp_min(1)), int(valid.sum())


MICRO = [None]                                                           # windows per forward / backward piece; halved on an out-of-memory error


def pieces(fn, x, *rest):
    """fn over x[s:s+m] for pieces of MICRO[0] windows, concatenated; the same result as one call up to floating-point rounding.
    The attention of this STAEFormer materialises an N x N matrix per window, head and step (head size 26 / 38 has no fused
    kernel), so large graphs need smaller pieces; MICRO is halved on an out-of-memory error and the call repeated."""
    while True:
        try:
            m = MICRO[0] or len(x)
            return torch.cat([fn(x[s:s + m], *(r[s:s + m] for r in rest)) for s in range(0, len(x), m)])
        except torch.OutOfMemoryError:
            assert (MICRO[0] or len(x)) > 1, 'out of GPU memory with one window per piece'
            MICRO[0] = max(1, (MICRO[0] or len(x)) // 2)
            torch.cuda.empty_cache()


def validation_loss(model, split, batch):
    """Mean over batches of the masked MAE, as the trainer computes it."""
    total, n = 0., 0
    with torch.no_grad():
        for s in range(0, split.count, batch):
            x, y, tod, dow = split.batch(torch.arange(s, min(s + batch, split.count), device=split.x.device))
            value, valid = masked_mae(y, pieces(model, x, tod, dow))
            if valid and np.isfinite(value):
                total, n = total + value, n + 1
    return total / n if n else float('inf')


def test_metrics(model, split, batch=64):
    """utils.metric.cal_metric: masked MAE / RMSE over horizons 1..i for i = 3, 6, 12, and the mean of the 12 cumulative MAEs."""
    abs_sum = torch.zeros(g.Y_LEN, dtype=torch.float64, device=split.x.device)
    sq_sum, cnt = abs_sum.clone(), abs_sum.clone()
    with torch.no_grad():
        for s in range(0, split.count, batch):
            x, y, tod, dow = split.batch(torch.arange(s, min(s + batch, split.count), device=split.x.device))
            err, valid = (pieces(model, x, tod, dow) - y).to(torch.float64), (y != 0)
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


def evaluate(model, splits):
    model.eval()
    return {name: test_metrics(model, split) for name, split in splits.items()}


def repository_model(z, seed, year, device):
    """The repository's own STAEFormer of this year and seed (no time inputs), as stored in the g30 bundle of the year."""
    shift = {42: 51, 43: 52, 44: 53}.get(seed)                         # evaluation seeds 51-53 hold the checkpoints of seeds 42-44
    prefix = 'ckpt/%s/y%d/' % (shift, year)
    state = {k[len(prefix):]: torch.as_tensor(np.asarray(z[k])) for k in z.files if k.startswith(prefix)}
    if not state:
        return None
    table = np.asarray(z['emb/%d/y%d' % (shift, year)], np.float32)
    model = st.TimeStae(table.shape[1], time=False).to(device)
    state[st.EMBEDDING] = torch.as_tensor(table)
    model.load_state_dict(state)
    return model.eval()


def train_one(dataset, variant, year, seed, conf, head_dir, bundle_dir, device, max_epochs, time_init_std=None, patience=PATIENCE):
    began = time.time()
    with np.load(head_dir / ('%s_%d_head.npz' % (dataset, year)), allow_pickle=False) as zh:
        head, hmeta = np.asarray(zh['raw_train_head'], np.float32), json.loads(str(zh['meta_json']))
    z, meta = g.load_bundle(bundle_dir / 'stae' / ('%s_%d.npz' % (dataset, year)))
    tail, val, test = (np.asarray(z[k], np.float32) for k in ('raw_train', 'raw_val', 'raw_test'))
    assert hashlib.sha256(np.ascontiguousarray(tail).tobytes()).hexdigest() == hmeta['tail_sha256'], 'the exported head does not belong to this bundle'
    train = np.concatenate([head, tail])
    lengths = meta['split_lengths']
    assert train.shape == (lengths['train'], int(meta['nodes'])) and len(val) == lengths['val'] and len(test) == lengths['test'], (train.shape, lengths)
    mean, std = (float(v) for v in (meta['norm'][str(year)]['mean'], meta['norm'][str(year)]['std']))
    assert abs(mean - hmeta['mean']) <= 1e-6 * (1 + abs(mean)) and abs(std - hmeta['std']) <= 1e-6 * (1 + abs(std)), (mean, std, hmeta['mean'], hmeta['std'])
    n = train.shape[1]
    dropout = float(conf.get('dropout') or 0.)
    lossfunc = dict(mse=func.mse_loss, huber=func.smooth_l1_loss, mae=func.l1_loss)[conf['loss']]
    batch, epochs = int(conf['batch_size']), min(int(conf['epoch']), max_epochs)
    first_val, first_test = lengths['train'], lengths['train'] + lengths['val']
    tr, va = Split(train, mean, std, device, year, 0), Split(val, mean, std, device, year, first_val)

    seed_anything(seed)
    model = st.TimeStae(n, time=(variant == 'tm'), dropout=dropout, time_init_std=time_init_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf['lr']))
    warm = int(conf.get('warmup_epochs') or 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: min(1., (ep + 1) / warm)) if warm > 0 else None
    clip = float(conf.get('grad_clip') or 0.)
    gen = torch.Generator().manual_seed(seed)
    lowest, counter, best, curve, skipped, seconds = 1e7, 0, None, [], 0, []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        order = torch.randperm(tr.count, generator=gen).to(device)
        total, cn = 0., 0
        for s in range(0, tr.count, batch):
            x, y, tod, dow = tr.batch(order[s:s + batch])
            while True:                                                 # the batch's gradient, in pieces of MICRO[0] windows if memory requires
                try:
                    optimizer.zero_grad(set_to_none=True)
                    m, value = MICRO[0] or len(x), 0.
                    for q in range(0, len(x), m):
                        part = lossfunc(y[q:q + m], model(x[q:q + m], tod[q:q + m], dow[q:q + m]), reduction='mean') * (len(x[q:q + m]) / len(x))
                        part.backward()
                        value += float(part)
                    break
                except torch.OutOfMemoryError:
                    assert (MICRO[0] or len(x)) > 1, 'out of GPU memory with one window per piece'
                    MICRO[0] = max(1, (MICRO[0] or len(x)) // 2)
                    torch.cuda.empty_cache()
            loss = torch.tensor(value)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            total, cn = total + float(loss), cn + 1
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        v = validation_loss(model, va, batch)
        curve.append((total / cn if cn else float('nan'), v))
        seconds.append(time.time() - t0)
        if v <= lowest:
            counter, lowest = 0, round(v, 4)
            best = (epoch, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
        else:
            counter += 1
            if counter > patience:
                break
    assert best is not None and np.isfinite(lowest), 'training never produced a finite validation error'
    model.load_state_dict(best[1])
    splits = dict(test_train_stats=Split(test, mean, std, device, year, first_test), test_own_stats=Split(test, *own_stats(test), device, year, first_test),
                  val_train_stats=va, val_own_stats=Split(val, *own_stats(val), device, year, first_val))
    info = dict(dataset=dataset, variant=variant, year=year, seed=seed, nodes=n, conf={k: conf.get(k) for k in ('lr', 'loss', 'dropout', 'warmup_epochs', 'grad_clip', 'batch_size', 'epoch')},
                epochs_run=len(curve), best_epoch=best[0], lowest_validation_loss=lowest, curve=curve, seconds_per_epoch=seconds, batches_skipped_nonfinite=skipped,
                train_windows=tr.count, val_windows=va.count, mean=mean, std=std, metrics=evaluate(model, splits), model_dim=model.backbone.model_dim,
                parameters=int(sum(v.numel() for v in best[1].values())), shared_parameters=int(sum(v.numel() for k, v in best[1].items() if k != st.EMBEDDING)),
                seconds=time.time() - began, torch=torch.__version__, device=str(device), bounded=bool(epochs < int(conf['epoch'])), time_init_std=time_init_std, patience=patience,
                micro_batch=MICRO[0] or batch)
    if variant == 'nt':
        ref = repository_model(z, seed, year, device)
        if ref is not None:
            info['repository_checkpoint'] = dict(metrics=evaluate(ref, splits))
    state = {k: v.numpy().astype(np.float32) for k, v in best[1].items()}
    return state, [str(s) for s in z['sensor_ids']], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--variant', choices=('tm', 'nt'), required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--head', required=True, help='directory with <DATASET>_<year>_head.npz (exported for g35)')
    ap.add_argument('--bundles', required=True, help='directory with stae/<DATASET>_<year>.npz')
    ap.add_argument('--conf', required=True, help='stae_conf.json: the repository configuration per dataset')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--max-epochs', type=int, default=1000, help='bounded smoke only')
    ap.add_argument('--time-init-std', type=float, default=settings.TIME_INIT_STD, help='the default is fixed in PRESPEC.md; other values: development probe only')
    ap.add_argument('--patience', type=int, default=PATIENCE, help='development probe only')
    opt = ap.parse_args()
    out, head = Path(opt.out).resolve(), Path(opt.head).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else ('cpu' if opt.device == 'auto' else opt.device))
    conf = json.loads(Path(opt.conf).read_text())[opt.dataset]
    bounded = opt.max_epochs < int(conf['epoch'])
    start, report = time.time(), dict(dataset=opt.dataset, variant=opt.variant, year=opt.year, runs={})
    torch.use_deterministic_algorithms(False)
    for seed in opt.seeds:
        state, ids, info = train_one(opt.dataset, opt.variant, opt.year, seed, conf, head, Path(opt.bundles), device, opt.max_epochs, opt.time_init_std, opt.patience)
        name = '%s_%s_%d_s%d' % (opt.dataset, opt.variant, opt.year, seed)
        np.savez(out / (name + '.npz'), sensor_ids=np.asarray(ids), meta_json=np.asarray(json.dumps(info)), **{'stae/' + k: v for k, v in state.items()})
        report['runs'][name] = {k: v for k, v in info.items() if k not in ('curve', 'seconds_per_epoch')}
        m = info['metrics']
        print(json.dumps(dict(run=name, nodes=info['nodes'], epochs=info['epochs_run'], best=info['best_epoch'], val=info['lowest_validation_loss'],
                              test_T12=round(m['test_train_stats']['T12_MAE'], 3), test_T12_own_stats=round(m['test_own_stats']['T12_MAE'], 3),
                              repository_T12_own_stats=(round(info['repository_checkpoint']['metrics']['test_own_stats']['T12_MAE'], 3) if 'repository_checkpoint' in info else None),
                              seconds=round(info['seconds'], 1), seconds_per_epoch=round(float(np.median(info['seconds_per_epoch'])), 1))), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else None
    report.update(seconds=time.time() - start, peak_gpu_gib=peak, bounded_smoke=bounded, seeds=opt.seeds, gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)
    (out / 'train_report.json').write_text(json.dumps(report, indent=1) + '\n')
    g.save_json(out / 'completed.json', dict(status='completed', runs=len(report['runs']), seconds=time.time() - start, bounded_smoke=bounded))
    print(json.dumps(dict(status='train_stae_completed', dataset=opt.dataset, variant=opt.variant, year=opt.year, runs=len(report['runs']),
                          seconds=round(time.time() - start, 1), peak_gpu_gib=peak)), flush=True)


if __name__ == '__main__':
    main()
