"""E19 (g39): is the ageing of the per-sensor identity an artefact of the missing time inputs? g29 protocol.

The STAEFormer of the benchmark (and of g30-g38) carries no time-of-day / day-of-week inputs. Without them the per-sensor
embedding may have to hold the sensor's daily profile, which is exactly what changes over the years. Here STAEFormer is
retrained per year with the repository's recipe in two variants that differ only in the time inputs (train_stae.py:
tm = the original's time-of-day and day-of-week embeddings, nt = without them), and the identity-refresh experiment of g33
is repeated for both on the same windows, calibrator and stream:

  same_rp, stale_rp     the repository's checkpoints (g33's arms same / stale5o): the link to every earlier package
  same_nt, same_tm      the retrained same-year model of each variant, its own table
  stale_nt, stale_tm    the retrained five-year-old model of each variant, its own table (a sensor it does not know gets the
                        mean of its table per input step, as for the repository's checkpoints)
  fit_nt, fit_tm        the retrained five-year-old model, its table refitted on the fit windows with g33's settings (rate
                        3e-3, at most 16 epochs, patience 3, the last 512 fit windows as holdout, every 2nd window, every 4th
                        above 1,600 sensors), the shared weights - the time embeddings included - frozen

Time of day / day of week of every input step come from the row of the year's matrix (stae_time.calendar): the splits are
consecutive, the bundle's train split is the last 2,072 rows of the train split.

  python e19_timeinput.py --bundle stae/<DS>_<year>.npz --models DIR [DIR ...] --seed 51 --out OUT
--models: directories holding <DS>_<variant>_<year>_s<seed>.npz from train_stae.py (variant nt / tm, the target year and the
year five before, the training seed 42 / 43 / 44 of evaluation seed 51 / 52 / 53).
"""
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn as nn
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import g28_common as g
import stae_model as sm
import stae_time as st

VARIANTS = ('nt', 'tm')
TRAIN_SEED = {51: 42, 52: 43, 53: 44}
MODELS = []                                                              # set by main(): the directories of the trained models


class Memo:
    """A bundle archive whose arrays are read once, so that a raw split is one object wherever it is asked for."""

    def __init__(self, z):
        self._z, self.files, self._seen = z, z.files, {}

    def __getitem__(self, key):
        if key not in self._seen:
            self._seen[key] = self._z[key]
        return self._seen[key]


class TimeBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.z = Memo(self.z)
        lengths = self.meta['split_lengths']
        self.first_row = {'raw_train': lengths['train'] - int(self.meta['train_tail_rows']), 'raw_val': lengths['train'],
                          'raw_test': lengths['train'] + lengths['val']}
        self._retrained, self._time, self._fit = {}, None, {}

    # ---- the retrained models ----
    def retrained(self, variant, year, device):
        """(model, table in the target year's sensor order, index of every target sensor in that year or -1, meta)."""
        key = (variant, year)
        if key not in self._retrained:
            name = '%s_%s_%d_s%d.npz' % (self.meta['dataset'], variant, year, TRAIN_SEED[self.seed])
            path = next((Path(d) / name for d in MODELS if (Path(d) / name).exists()), None)
            assert path is not None, 'missing retrained model ' + name
            with np.load(path, allow_pickle=False) as z:
                state = {k[len('stae/'):]: torch.as_tensor(z[k]) for k in z.files if k.startswith('stae/')}
                ids, meta = [str(s) for s in z['sensor_ids']], json.loads(str(z['meta_json']))
            assert (meta['dataset'], meta['variant'], meta['year'], meta['seed']) == (self.meta['dataset'], variant, year, TRAIN_SEED[self.seed]) and not meta['bounded'] or \
                os.environ.get('G39_ALLOW_BOUNDED') == '1', ('a bounded or foreign checkpoint', path, meta.get('bounded'))
            mean, std = self.norm[year]
            assert abs(meta['mean'] - mean) <= 1e-6 * (1 + abs(mean)) and abs(meta['std'] - std) <= 1e-6 * (1 + abs(std)), 'normalisation differs from the bundle'
            table = state.pop(st.EMBEDDING).float()
            model = st.TimeStae(table.shape[1], time=(variant == 'tm')).to(device)
            own = model.state_dict()
            own.update({k: v for k, v in state.items()})
            own[st.EMBEDDING] = table
            model.load_state_dict(own)
            for p in model.parameters():
                p.requires_grad_(False)
            target = [str(s) for s in self.z['sensor_ids']]
            if year == self.year:
                assert ids == target, 'the same-year model was trained on another sensor list'
                index = np.arange(len(target))
            else:
                position = {s: i for i, s in enumerate(ids)}
                index = np.array([position.get(s, -1) for s in target])
                assert np.array_equal(index >= 0, np.asarray(self.z['known/y%d' % year], bool)), 'sensor mapping differs from the bundle'
            mapped, _ = sm.mapped_table(table, index)
            self._retrained[key] = (model.eval(), mapped.contiguous().to(device), index, meta)
            self.info['model:%s:%d' % key] = {k: meta[k] for k in ('epochs_run', 'best_epoch', 'lowest_validation_loss', 'nodes', 'model_dim', 'shared_parameters', 'seconds')}
        return self._retrained[key]

    # ---- forecasts: the position of every window is known to base(), the forward pass of a variant model reads it there ----
    def times(self, raw, first, count, device):
        name = next(k for k in ('raw_train', 'raw_val', 'raw_test') if raw is self.z[k])
        tod, dow = st.window_time(self.year, self.first_row[name], np.arange(first, first + count))
        return tod.to(device), dow.to(device)

    def base(self, key, raw, first, count, device):
        """e5_stae.StaeBundle.base with the position of each block handed to the forward pass."""
        parts = []
        for block in range(first // e5.BLOCK_WINDOWS, (first + count - 1) // e5.BLOCK_WINDOWS + 1):
            slot = (key, id(raw), block)
            if slot in self._cache:
                self._cache.move_to_end(slot)
            else:
                start = block * e5.BLOCK_WINDOWS
                size = min(e5.BLOCK_WINDOWS, g.n_windows(raw) - start)
                self._time = self.times(raw, start, size, device)
                value = self.forward(key, np.array(g.get_windows(raw, start, size)[0], dtype=np.float32), device)
                self._time = None
                self._cache[slot] = value
                self._cache_bytes += value.numel() * value.element_size()
                while self._cache_limit and self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                    _, dropped = self._cache.popitem(last=False)
                    self._cache_bytes -= dropped.numel() * dropped.element_size()
            lo = max(first, block * e5.BLOCK_WINDOWS) - block * e5.BLOCK_WINDOWS
            hi = min(first + count, (block + 1) * e5.BLOCK_WINDOWS) - block * e5.BLOCK_WINDOWS
            parts.append(self._cache[slot][lo:hi])
        return torch.cat(parts)

    def _forward(self, key, x_raw, device, sub):
        base, ckpt, norm, variant = key
        if not (isinstance(variant, str) and variant[:3] in ('nt:', 'tm:')):
            return super()._forward(key, x_raw, device, sub)
        v, kind = variant.split(':')
        model, table, _, _ = self.retrained(v, ckpt, device)
        if kind == 'fit':
            table = self.fit_time(v, device)
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        tod, dow = self._time
        with torch.no_grad():
            out = torch.cat([model(x[i:i + sub], tod[i:i + sub], dow[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))

    # ---- g33's refit of the table, on the retrained five-year-old model of a variant ----
    def fit_time(self, variant, device):
        if variant in self._fit:
            return self._fit[variant]
        began = time.time()
        model, start, index, _ = self.retrained(variant, self.old, device)
        raw = self.z['raw_train']
        n_fit = min(e3.FIT_WINDOWS, g.n_windows(raw))
        first = g.n_windows(raw) - n_fit
        stride = e10.fit_stride(self.n)
        train = np.arange(first, first + n_fit - e10.FIT['holdout'], stride)
        hold = np.arange(first + n_fit - e10.FIT['holdout'], first + n_fit, stride)
        mean, std = self.norm[self.old]
        tod_all, dow_all = self.times(raw, 0, g.n_windows(raw), device)

        def batch(idx):
            x = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in idx])
            y = g.clean_y(np.stack([np.array(g.get_windows(raw, int(i), 1)[1][0]) for i in idx])).to(device)
            t = torch.as_tensor(np.asarray(idx, np.int64), device=device)
            return g.normalize_x(x, mean, std).to(device), y, g.target_valid(y), tod_all[t], dow_all[t]

        size = [e10.fit_batch(self.n)]

        def guarded(fn):
            while True:
                try:
                    return fn(size[0])
                except torch.cuda.OutOfMemoryError:
                    assert size[0] > 1, 'out of GPU memory with one window per step'
                    size[0] = max(1, size[0] // 2)
                    torch.cuda.empty_cache()

        def holdout(table):
            def run(b):
                total = count = 0.
                with torch.no_grad():
                    for s in range(0, len(hold), b):
                        x, y, valid, tod, dow = batch(hold[s:s + b])
                        total += float(((model(x, tod, dow, table) - y).abs() * valid).sum())
                        count += float(valid.sum())
                return total / max(count, 1.)
            return guarded(run)

        table = nn.Parameter(start.clone())
        optim = torch.optim.Adam([table], lr=e10.FIT['lr'])
        gen = torch.Generator().manual_seed(7919 * self.seed + self.year)
        history = [holdout(start)]
        best = (history[0], start.clone(), 0)
        for epoch in range(e10.FIT['epochs']):
            order = train[torch.randperm(len(train), generator=gen).numpy()]
            s = 0
            while s < len(order):
                def step(b):
                    x, y, valid, tod, dow = batch(order[s:s + b])
                    with torch.enable_grad():
                        loss = ((model(x, tod, dow, table) - y).abs() * valid).sum() / valid.sum().clamp_min(1)
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                    optim.step()
                    return b
                s += guarded(step)
            history.append(holdout(table.detach()))
            if history[-1] < best[0]:
                best = (history[-1], table.detach().clone(), epoch + 1)
            elif epoch + 1 - best[2] >= e10.FIT['patience']:
                break
        self.info['fit:' + variant] = dict(checkpoint_year=self.old, lr=e10.FIT['lr'], max_epochs=e10.FIT['epochs'], patience=e10.FIT['patience'], stride=stride,
                                           train_windows=int(len(train)), holdout_windows=int(len(hold)), batch_windows=size[0], holdout_mae=history, best_epoch=best[2],
                                           epochs_run=len(history) - 1, mean_abs_change=float((best[1] - start).abs().mean()), seconds=time.time() - began)
        self._fit[variant] = best[1].contiguous()
        return self._fit[variant]


def arms_of(B, seed):
    B.seed = seed
    arm = lambda name, ckpt, variant: dict(name=name, base='stae', ckpt=ckpt, norm=ckpt, age=B.year - ckpt, variant=variant)
    arms = [arm('same_rp', B.year, 'full'), arm('stale_rp', B.old, 'full')]
    for v in VARIANTS:
        arms += [arm('same_' + v, B.year, v + ':full'), arm('stale_' + v, B.old, v + ':full'), arm('fit_' + v, B.old, v + ':fit')]
    B.info.update(sensors=B.n, known_to_old=int(np.asarray(B.z['known/y%d' % B.old], bool).sum()), old_year=B.old, train_seed=TRAIN_SEED[seed])
    return arms


def check_reference(B, seed, arms, weights, adj, device):
    return e5.check_reference(B, seed, [a for a in arms if a.get('variant') == 'full'], weights, adj, device)


e3.Bundle = TimeBundle
e3.arms_of = arms_of
e3.make_chunks = e5.make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    i = sys.argv.index('--models')
    j = next((k for k in range(i + 1, len(sys.argv)) if sys.argv[k].startswith('--')), len(sys.argv))
    MODELS[:] = sys.argv[i + 1:j]
    del sys.argv[i:j]
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(e10.RefreshBundle.last.info, indent=2) + '\n')
