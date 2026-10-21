"""E14 (g36): the identity-refresh experiment of g33 with Graph WaveNet as the backbone, under the g29 protocol.

Graph WaveNet holds two embeddings per sensor, E1 and E2 (N x 10 each). They are not an input feature: they generate
an adaptive adjacency softmax(relu(E1 E2^T)) that the backbone averages with the static adjacency of the dataset
before every graph convolution. "Refreshing the identities" of a frozen old Graph WaveNet therefore means refreshing
its learned graph. The repository keeps Graph WaveNet checkpoints for the last year only, so the per-year models are
trained by train_gwn.py with the repository's model class and recipe (backbone seeds 42-44 serve evaluation seeds 51-53).

Everything else is e10_refresh (g33) unchanged: the same arms, donors, embedding-only fit, stream, calibrator and
metrics; the table is [1, N, 20] = [E1 | E2]. With the mean embedding for every sensor (arms *_mean) the adaptive
adjacency is exactly uniform. Two arms are added for the static adjacency of the same-year model, as in g31 C3:

  same_perm   static adjacency with its nodes permuted (g28_common.permute_nodes, seed 2811), learned graph unchanged
  same_self   static adjacency replaced by self-loops only, learned graph unchanged

Data (raw splits, sensor identities, input statistics) are read from the g30 STAEFormer bundle of the target year, the
static adjacency from the g29 bundle; checkpoints from --ckpt (or A2TTA_GWN_CKPT): <dir>/<DATASET>/<DATASET>_<year>_s<seed>.npz.

The embedding-only fit is the one of g33 (e10_refresh.fit) run once per candidate learning rate (FIT_RATES); the table
with the lowest error on the holdout inside the fit windows is kept. That holdout already chooses the epoch in g33; no
validation or test value is read (PRESPEC.md, "Fit settings").
"""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import e3_confirm as e3
import e5_stae as e5
import e10_refresh as e10
import g28_common as g
import gwn_model as gm
import stae_model

PANEL = (2015, 2020, 2025)                                              # years with a trained Graph WaveNet
SEED_SHIFT = 9                                                          # evaluation seed 51 uses backbone seed 42
PERMUTATION_SEED = 2811                                                 # the permutation of g28 / g31 C3
STATIC = dict(perm='perm', self='self')                                 # variants that change the static adjacency, not the table
FIT_RATES = (1e-3, 3e-3, 1e-2, 3e-2)                                    # candidates of the embedding-only fit; 3e-3 is the rate of g33 / g35


def checkpoint_dir():
    if '--ckpt' in sys.argv:
        return Path(sys.argv[sys.argv.index('--ckpt') + 1]).resolve()
    return Path(os.environ['A2TTA_GWN_CKPT']).resolve()


class GwnBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.ckpt_years = [y for y in self.ckpt_years if y in PANEL]
        assert self.old in self.ckpt_years[1:]
        self._sub = 256 if self.n <= 800 else (64 if self.n <= 2600 else 16)     # windows per forward; halved on an out-of-memory error
        self._cache_limit = 0
        self._ckpt, self._mapped, self._static, self.ckpt_dir = {}, {}, {}, checkpoint_dir()

    def checkpoint(self, year):
        """(shared weights, table [N_year, 20], sensor ids of that year, training record) of the Graph WaveNet trained for `year`."""
        if year not in self._ckpt:
            name = '%s_%d_s%d.npz' % (self.meta['dataset'], year, self.seed - SEED_SHIFT)
            with np.load(self.ckpt_dir / self.meta['dataset'] / name, allow_pickle=False) as z:
                state = {k[len('gwn/'):]: torch.as_tensor(z[k]) for k in z.files if k.startswith('gwn/')}
                ids, info = [str(s) for s in z['sensor_ids']], json.loads(str(z['meta_json']))
            assert (info['dataset'], info['year'], info['seed']) == (self.meta['dataset'], year, self.seed - SEED_SHIFT)
            mean, std = self.norm[year]
            assert abs(info['mean'] - mean) <= 1e-6 * (1 + abs(mean)) and abs(info['std'] - std) <= 1e-6 * (1 + abs(std)), 'trained with other input statistics'
            shared, table = gm.split_state(state)
            assert table.shape == (len(ids), 2 * gm.ADP)
            self._ckpt[year] = (shared, table.to(torch.float32), ids, info)
            self.info.setdefault('gwn_checkpoints', {})[str(year)] = dict(file=name, best_epoch=info['best_epoch'], epochs_run=info['epochs_run'],
                                                                        lowest_validation_loss=info['lowest_validation_loss'],
                                                                        test_T12_MAE=info['metrics']['test_train_stats']['T12_MAE'])
        return self._ckpt[year]

    def model(self, ckpt_year, device):
        if ckpt_year not in self._models:
            self._models[ckpt_year] = gm.GwnModel(0.).load_shared(self.checkpoint(ckpt_year)[0]).to(device)
        return self._models[ckpt_year]

    def mapped(self, ckpt_year):
        """[E1 | E2] of a checkpoint in the sensor order of the target year, [1, N, 20] on the CPU; mean embedding for sensors it does not know."""
        if ckpt_year not in self._mapped:
            _, table, ids, _ = self.checkpoint(ckpt_year)
            position = {s: i for i, s in enumerate(ids)}
            index = np.array([position.get(str(s), -1) for s in self.z['sensor_ids']])
            mapped, known = stae_model.mapped_table(table[None], index)
            if ckpt_year == self.year:
                assert bool(known.all()) and np.array_equal(index, np.arange(self.n)), 'the same-year checkpoint was trained on other sensors'
            else:
                assert np.array_equal(known.numpy(), np.asarray(self.z['known/y%d' % ckpt_year], bool)), 'sensor identities differ from the bundle'
            self._mapped[ckpt_year] = mapped
        return self._mapped[ckpt_year]

    def table(self, ckpt_year, device):
        key = (ckpt_year, str(device))
        if key not in self._tables:
            self._tables[key] = self.mapped(ckpt_year).to(device)
        return self._tables[key]

    def old_tables(self, zo, ids_old):
        _, table, ids, _ = self.checkpoint(self.old)
        assert ids == ids_old, 'the old checkpoint was trained on other sensors than the old bundle holds'
        return table[None].numpy(), self.mapped(self.old).numpy()

    def static(self, kind, device):
        """Static adjacency of the target year: as given ('true'), node-permuted ('perm') or self-loops only ('self')."""
        key = (kind, str(device))
        if key not in self._static:
            true = e10.RefreshBundle.adjacency(self, torch.device('cpu'))
            assert true is not None and true.shape == (self.n, self.n), 'Graph WaveNet needs the adjacency of the target year (gcn bundle)'
            if kind == 'perm':
                true = torch.as_tensor(np.asarray(g.permute_nodes(true.numpy(), PERMUTATION_SEED)[0], np.float32))
            elif kind == 'self':
                true = torch.as_tensor(g.normalize_adj(np.eye(self.n)))
            self._static[key] = true.to(device)
        return self._static[key]

    def predict(self, model, x, table):
        return model.predict(x, table[0], self.static('true', x.device))

    def release(self, model, start):
        pass

    def fit(self, device, ckpt=None, start=None, tag='fit'):
        """The fit of e10 for every candidate learning rate; the table with the lowest holdout error is kept (the smaller rate on a tie)."""
        best, runs = None, {}
        for rate in FIT_RATES:
            e10.FIT['lr'] = rate
            table = e10.RefreshBundle.fit(self, device, ckpt, start, tag)
            run = self.info[tag]
            runs['%g' % rate] = run
            if best is None or min(run['holdout_mae']) < min(best[1]['holdout_mae']):
                best = (table, run)
        self.info[tag] = dict(best[1], selected_lr=best[1]['lr'], candidates=runs, seconds=sum(r['seconds'] for r in runs.values()))
        return best[0]

    def _forward(self, key, x_raw, device, sub):
        _, ckpt, norm, variant = key
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model = self.model(ckpt, device)
        table = (self.table(ckpt, device) if variant == 'full' or variant in STATIC else self.variant(ckpt, variant, device))[0]
        adj = self.static(STATIC.get(variant, 'true'), device)
        out = torch.cat([model(x[i:i + sub], table, adj) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))


def arms_of(B, seed):
    arms = e10.arms_of(B, seed)
    extra = [dict(name='same_' + v, base='stae', ckpt=B.year, norm=B.year, age=0, variant=v) for v in STATIC]
    cut = next(i for i, a in enumerate(arms) if a['base'] != 'stae')   # before the arms without a backbone
    return arms[:cut] + extra + arms[cut:]


def check_reference(B, seed, arms, weights, adj, device):
    """No stored reference forecasts: every checkpoint was compared with the project's model class when it was trained."""
    return {str(y): B.checkpoint(y)[3]['standalone_max_abs_difference'] for y in sorted({a['ckpt'] for a in arms if a['base'] == 'stae'})}


e3.Bundle = GwnBundle
e3.arms_of = arms_of
e3.make_chunks = e5.make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    if '--ckpt' in sys.argv:                                            # e3's parser does not know the option
        i = sys.argv.index('--ckpt')
        os.environ['A2TTA_GWN_CKPT'] = str(Path(sys.argv[i + 1]).resolve())
        del sys.argv[i:i + 2]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(e10.RefreshBundle.last.info, indent=2) + '\n')
