"""E13 (g35): the identity-refresh experiment of g33 with STID as the backbone, under the g29 protocol.

STID holds one 32-dimensional embedding per sensor (`backbone.node_embed`, [N, 32]) next to shared MLP weights. The
repository keeps STID checkpoints for the last year only, so the per-year models are trained by train_stid.py with the
repository's model class and recipe (backbone seeds 42-44 serve evaluation seeds 51-53, as for STAEFormer).

Everything else is e10_refresh (g33) unchanged: the same arms, donors, embedding-only fit, stream, calibrator and
metrics. Data (raw splits, sensor identities, input statistics) are read from the g30 STAEFormer bundle of the target
year; the STID checkpoints come from --ckpt (or A2TTA_STID_CKPT): <dir>/<DATASET>/<DATASET>_<year>_s<seed>.npz.
The node table is handled as [1, N, 32] so that the table code of e10 applies as it is.
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
import stae_model
import stid_model as sm

PANEL = (2015, 2020, 2025)                                              # years with a trained STID
SEED_SHIFT = 9                                                          # evaluation seed 51 uses backbone seed 42


def checkpoint_dir():
    if '--ckpt' in sys.argv:
        return Path(sys.argv[sys.argv.index('--ckpt') + 1]).resolve()
    return Path(os.environ['A2TTA_STID_CKPT']).resolve()


class StidBundle(e10.RefreshBundle):
    def __init__(self, path):
        super().__init__(path)
        self.ckpt_years = [y for y in self.ckpt_years if y in PANEL]
        assert self.old in self.ckpt_years[1:]
        self._sub, self._cache_limit = 256, 0                           # an MLP: large blocks, unbounded forecast cache
        self._ckpt, self._mapped, self.ckpt_dir = {}, {}, checkpoint_dir()

    def checkpoint(self, year):
        """(shared weights, node table [N_year, 32], sensor ids of that year, training record) of the STID trained for `year`."""
        if year not in self._ckpt:
            name = '%s_%d_s%d.npz' % (self.meta['dataset'], year, self.seed - SEED_SHIFT)
            with np.load(self.ckpt_dir / self.meta['dataset'] / name, allow_pickle=False) as z:
                state = {k[len('stid/'):]: torch.as_tensor(z[k]) for k in z.files if k.startswith('stid/')}
                ids, info = [str(s) for s in z['sensor_ids']], json.loads(str(z['meta_json']))
            assert (info['dataset'], info['year'], info['seed']) == (self.meta['dataset'], year, self.seed - SEED_SHIFT)
            mean, std = self.norm[year]
            assert abs(info['mean'] - mean) <= 1e-6 * (1 + abs(mean)) and abs(info['std'] - std) <= 1e-6 * (1 + abs(std)), 'trained with other input statistics'
            shared, table = sm.split_state(state)
            assert table.shape == (len(ids), 32)
            self._ckpt[year] = (shared, table.to(torch.float32), ids, info)
            self.info.setdefault('stid_checkpoints', {})[str(year)] = dict(file=name, best_epoch=info['best_epoch'], epochs_run=info['epochs_run'],
                                                                         lowest_validation_loss=info['lowest_validation_loss'],
                                                                         test_T12_MAE=info['metrics']['test_train_stats']['T12_MAE'])
        return self._ckpt[year]

    def model(self, ckpt_year, device):
        if ckpt_year not in self._models:
            self._models[ckpt_year] = sm.StidModel(0.).load_shared(self.checkpoint(ckpt_year)[0]).to(device)
        return self._models[ckpt_year]

    def mapped(self, ckpt_year):
        """Node table of a checkpoint in the sensor order of the target year, [1, N, 32] on the CPU; mean embedding for sensors it does not know."""
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

    def predict(self, model, x, table):
        return model.predict(x, table[0])

    def release(self, model, start):
        pass

    def _forward(self, key, x_raw, device, sub):
        _, ckpt, norm, variant = key
        x = g.normalize_x(x_raw, *self.norm[norm]).to(device)
        model = self.model(ckpt, device)
        table = (self.table(ckpt, device) if variant == 'full' else self.variant(ckpt, variant, device))[0]
        out = torch.cat([model(x[i:i + sub], table) for i in range(0, len(x), sub)])
        return out.to(torch.float32).to(self.cache_device(device))


def check_reference(B, seed, arms, weights, adj, device):
    """No stored reference forecasts: every checkpoint was compared with the project's model class when it was trained."""
    return {str(y): B.checkpoint(y)[3]['standalone_max_abs_difference'] for y in sorted({a['ckpt'] for a in arms if a['base'] == 'stae'})}


e3.Bundle = StidBundle
e3.arms_of = e10.arms_of
e3.make_chunks = e5.make_chunks
e3.check_reference = check_reference

if __name__ == '__main__':
    if '--ckpt' in sys.argv:                                            # e3's parser does not know the option
        i = sys.argv.index('--ckpt')
        os.environ['A2TTA_STID_CKPT'] = str(Path(sys.argv[i + 1]).resolve())
        del sys.argv[i:i + 2]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    e3.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    (out / 'e10_info.json').write_text(json.dumps(e10.RefreshBundle.last.info, indent=2) + '\n')
