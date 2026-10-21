"""E5 (g30 S): the g29 protocol with STAEFormer base forecasts.

`e3_confirm.run` (the frozen g29 protocol: fit on the last 2048 train windows, selection on the first 1024
validation windows, one stream over validation + test, same masks and metrics) is used unchanged. Only three
things are replaced: the list of arms, the provider of base forecasts, and the reference check.

Arms (see PRESPEC.md): same; stale<k> / stale<k>o (old checkpoint on all sensors, mean embedding for sensors it
does not know; target-year / own input statistics); known5 (the five-year-old checkpoint run only on the sensors
it knows, last observed value for the others; called `stale5c` in PRESPEC.md); nobb_zero / nobb_last.
"""
import collections
import os
import numpy as np
import torch
import e3_confirm as e3
import g28_common as g
import stae_model as sm

BLOCK_WINDOWS = 128
# Memory controls; neither changes a forecast. On graphs with more than 4000 sensors the spatial attention of one window
# takes about 9 GiB, and a run of g30 (PEMS07/2025, eight forecast keys) failed on the execution platform when the cache and the attention
# of two windows were held together. Above 4000 sensors: one window per forward and a cache bounded at 6 GiB.
# Environment overrides exist for tests only.
SUB_OVERRIDE = int(os.environ.get('G30_SUB', '0'))                  # windows per forward
CACHE_GIB_OVERRIDE = os.environ.get('G30_CACHE_GIB')                # bound of the forecast cache in GiB; 0 = unbounded


class StaeBundle(e3.Bundle):
    def __init__(self, path):
        self.z, self.meta = g.load_bundle(path)
        m = self.meta
        assert m['layout'] == 'g30-stae', m.get('layout')
        self.year, self.n = int(m['year']), int(m['nodes'])
        self.g29 = True                                        # new_mask/y<year> naming of the g29 layout
        self.ckpt_years = [int(y) for y in m['ckpt_years']]
        self.norm = {int(k): (v['mean'], v['std']) for k, v in m['norm'].items()}
        self.seed, self._models, self._tables, self._known = None, {}, {}, {}
        self._cache, self._cache_bytes = collections.OrderedDict(), 0
        self._sub = SUB_OVERRIDE or (32 if self.n <= 800 else (16 if self.n <= 2600 else (8 if self.n <= 4000 else 1)))   # windows per forward
        gib = float(CACHE_GIB_OVERRIDE) if CACHE_GIB_OVERRIDE is not None else (6. if self.n > 4000 else 0.)
        self._cache_limit = int(gib * 2 ** 30)                  # least recently used blocks are dropped beyond it and recomputed if asked again
        self._cache_device = None

    def adjacency(self, device):
        return None

    def model(self, ckpt_year, device):
        if ckpt_year not in self._models:
            prefix = 'ckpt/%d/y%d/' % (self.seed, ckpt_year)
            state = {k[len(prefix):]: torch.as_tensor(self.z[k]) for k in self.z.files if k.startswith(prefix)}
            self._models[ckpt_year] = sm.StaeModel(dropout=float(self.meta['dropout'])).load_shared(state).to(device)
        return self._models[ckpt_year]

    def table(self, ckpt_year, device):
        if ckpt_year not in self._tables:
            self._tables[ckpt_year] = torch.as_tensor(self.z['emb/%d/y%d' % (self.seed, ckpt_year)], dtype=torch.float32, device=device)
        return self._tables[ckpt_year]

    def known(self, ckpt_year, device):
        if ckpt_year not in self._known:
            self._known[ckpt_year] = torch.as_tensor(np.asarray(self.z['known/y%d' % ckpt_year], bool), device=device)
        return self._known[ckpt_year]

    def _forward(self, key, x_raw, device, sub):
        """Base forecast [len, N, 12] (float32, CPU) for raw input windows x_raw [len, N, 12]; key = (base, ckpt, norm, variant)."""
        _, ckpt, norm, variant = key
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model, table = self.model(ckpt, device), self.table(ckpt, device)
        if variant == 'common':
            known = self.known(ckpt, device)
            last = torch.nan_to_num(torch.from_numpy(x_raw[..., -1:]), nan=0., posinf=0., neginf=0.).to(device)
            out = last.expand(-1, -1, g.Y_LEN).clone()
            if bool(known.any()):                              # nothing to run when no sensor survived
                sub_table = table[:, known]
                for i in range(0, len(x), sub):
                    out[i:i + sub, known] = model(x[i:i + sub][:, known], sub_table)
        else:
            out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))

    def cache_device(self, device):
        """Where cached base forecasts live. On cards with at least 40 GB they stay on the GPU: on the execution platform, host-memory
        allocations of jobs that kept several GB of forecasts in RAM failed with ENOMEM (g30 groups s03 and s12)."""
        if self._cache_device is None:
            big = device.type == 'cuda' and torch.cuda.get_device_properties(device).total_memory >= 40 * 2 ** 30
            self._cache_device = device if big else torch.device('cpu')
        return self._cache_device

    def forward(self, key, x_raw, device):
        """_forward with a smaller sub-batch after an out-of-memory error (attention memory depends on the kernel in use)."""
        while True:
            try:
                return self._forward(key, x_raw, device, self._sub)
            except torch.cuda.OutOfMemoryError:
                assert self._sub > 1, 'out of GPU memory with one window per forward'
                self._sub = max(1, self._sub // 2)
                torch.cuda.empty_cache()

    def base(self, key, raw, first, count, device):
        """Cached base forecast of windows [first, first+count) of one raw split."""
        parts = []
        for block in range(first // BLOCK_WINDOWS, (first + count - 1) // BLOCK_WINDOWS + 1):
            slot = (key, id(raw), block)
            if slot in self._cache:
                self._cache.move_to_end(slot)
            else:
                start = block * BLOCK_WINDOWS
                size = min(BLOCK_WINDOWS, g.n_windows(raw) - start)
                value = self.forward(key, np.array(g.get_windows(raw, start, size)[0], dtype=np.float32), device)
                self._cache[slot] = value
                self._cache_bytes += value.numel() * value.element_size()
                while self._cache_limit and self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                    _, dropped = self._cache.popitem(last=False)
                    self._cache_bytes -= dropped.numel() * dropped.element_size()
            lo = max(first, block * BLOCK_WINDOWS) - block * BLOCK_WINDOWS
            hi = min(first + count, (block + 1) * BLOCK_WINDOWS) - block * BLOCK_WINDOWS
            parts.append(self._cache[slot][lo:hi])
        return torch.cat(parts)


def key_of(arm):
    return (arm['base'], arm['ckpt'], arm['norm'], arm.get('variant'))


def arms_of(B, seed):
    B.seed = seed
    arms = [dict(name='same', base='stae', ckpt=B.year, norm=B.year, age=0, variant='full')]
    for y in B.ckpt_years[1:]:
        age = B.year - y
        arms.append(dict(name='stale%d' % age, base='stae', ckpt=y, norm=B.year, age=age, variant='full'))
        arms.append(dict(name='stale%do' % age, base='stae', ckpt=y, norm=y, age=age, variant='full'))
        if age == 5:
            arms.append(dict(name='known5', base='stae', ckpt=y, norm=B.year, age=age, variant='common'))
    arms.append(dict(name='nobb_zero', base='zero', ckpt=None, norm=B.year, age=None, variant=None))
    arms.append(dict(name='nobb_last', base='last', ckpt=None, norm=B.year, age=None, variant=None))
    return arms


def make_chunks(B, weights, adj, device, arms, pieces):
    """Same contract as e3_confirm.make_chunks; STAEFormer forecasts come from the bundle's block cache."""
    offsets = np.cumsum([0] + [c for _, _, c in pieces])
    T = int(offsets[-1])

    def segments(s):
        out, t = [], s.start
        while t < s.stop:
            k = int(np.searchsorted(offsets, t, side='right') - 1)
            raw, first, _ = pieces[k]
            take = min(s.stop, int(offsets[k + 1])) - t
            out.append((raw, first + t - int(offsets[k]), take))
            t += take
        return out

    def chunk_fn(s):
        segs = segments(s)
        x_raw = np.concatenate([np.array(g.get_windows(raw, first, take)[0], dtype=np.float32) for raw, first, take in segs])
        y = torch.cat([g.clean_y(g.get_windows(raw, first, take)[1]) for raw, first, take in segs]).to(device)
        memo, xs, bs = {}, [], []
        for arm in arms:
            key = key_of(arm)
            if key not in memo:
                x = g.normalize_x(x_raw, *B.norm[arm['norm']]).to(device)
                if arm['base'] == 'stae':
                    b = torch.cat([B.base(key, raw, first, take, device) for raw, first, take in segs]).to(device)
                elif arm['base'] == 'zero':
                    b = torch.zeros_like(y)
                else:
                    last = torch.nan_to_num(torch.from_numpy(x_raw[..., -1:]), nan=0., posinf=0., neginf=0.)
                    b = last.to(device).expand_as(y)
                memo[key] = (x.to(torch.float64), b.to(torch.float64))
            xs.append(memo[key][0])
            bs.append(memo[key][1])
        r = len(arms)
        return (torch.cat(xs, 1), torch.cat(bs, 1), y.repeat(1, r, 1).to(torch.float64),
                g.window_valid(y).repeat(1, r), None)
    return T, chunk_fn


def check_reference(B, seed, arms, weights, adj, device):
    """Standalone forecasts against the original model code's reference predictions stored by the export job."""
    index = B.z['ref_index']
    if len(index) == 0:
        return dict(reference='not stored for this graph size')
    raw = B.z['raw_test']
    x_raw = np.stack([np.array(g.get_windows(raw, int(i), 1)[0][0], dtype=np.float32) for i in index])
    report = {}
    for arm in arms:
        if arm['base'] != 'stae':
            continue
        tag = 'common' if arm['variant'] == 'common' else ('year' if arm['norm'] == B.year else 'own')
        name = 'ref/%d/y%d/%s' % (seed, arm['ckpt'], tag)
        if name not in B.z.files:
            report[arm['name']] = None
            continue
        ref = torch.as_tensor(B.z[name], device=device)
        mine = B.forward(key_of(arm), x_raw, device).to(device)
        if tag == 'common':
            mine = mine[:, B.known(arm['ckpt'], device)]
        worst, scale = float((mine - ref).abs().max()), float(ref.abs().max())
        assert worst <= 1e-3 * (1 + scale), ('standalone STAEFormer differs from the original model', arm['name'], worst, scale)
        report[arm['name']] = dict(max_abs_difference=worst, reference_abs_max=scale, windows=int(len(index)))
    return report


e3.Bundle = StaeBundle
e3.arms_of = arms_of
e3.make_chunks = make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    e3.main()
