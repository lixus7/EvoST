"""Export the checkpoint bundles that the EvoST experiments read. CPU only; data and checkpoints are only read.

Three kinds of bundle, one NumPy .npz file per network and year:

* ``stae``: maintenance bundle. It holds the target year's raw splits and the STAEFormer checkpoints of the target year
  and of the earlier panel years (2010, 2015, 2020, 2025), every identity table mapped to the target year's sensor
  order (layout ``g30-stae``). The maintenance studies read it from ``bundles/stae/``.
* ``graph``: the row-normalized adjacency and the stable sensor IDs of a year. Graph WaveNet and the geographic donor
  controls read it from ``bundles/gcn/`` next to ``bundles/stae/``.
* ``annual``: annual-benchmark bundle with the same-year checkpoints of evaluation seeds 51-55 only.

``digest`` writes the SHA-256 of every array of the bundles in a directory, and ``compare`` checks a new export
against such a digest file.

The functions follow the exporters that wrote the paper's bundles. On the original data and checkpoints, ten sampled exports
matched all 4,643 compared numerical and sensor-ID arrays byte for byte (PyTorch 2.2.1, NumPy 1.26.4); graph bundles are compared on
adj_norm and sensor_ids. This is an array-level check, not byte identity of the bundle files: metadata are compared separately
after removing paths and library versions, and two maintenance bundles differ in a diagnostic of the reference check. Inputs:

* ``--data-root``: ``<root>/<dataset in lower case>/{RawData,FastData,graph}/<year>...`` as in data/README.md.
  ``FastData`` must exist, because the split lengths and the target checks come from it.
* ``--model-root``: ``<root>/<DATASET>/retrain_staeformer_<dataset>-<seed>/<year>/<loss>.pkl``, the per-year
  checkpoints written by the baseline pipeline (scripts/extra_baselines_run.sh). Without a cases file, the file with
  the lowest loss in its name is used.
* ``--metadata-root`` (repeatable, searched in order): ``<root>/<dataset>/<year>/sensor_metadata.csv``.
  ``--tfnsw-nodes``: ``<dir>/<year>_nodes.csv`` for TFNSW years without a metadata file.
* ``--conf``: a STAEFormer configuration for the model settings. The default is the dataset's
  ``conf/<DATASET>/retrain_staeformer_<dataset>.json``, otherwise the PEMS03 file; all datasets share the STAEFormer
  settings.

Usage:
  python experiments/evost/export_bundles.py stae --dataset PEMS03 --out bundles/stae
  python experiments/evost/export_bundles.py graph --dataset PEMS03 --out bundles/gcn
  python experiments/evost/export_bundles.py annual --dataset PEMS03 --out bundles/annual \\
      [--cases experiments/evost/annual/cases-stage1.json --ckpt-root DIR]
  python experiments/evost/export_bundles.py digest bundles/stae --out digests.json
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE / 'annual'), str(REPO / 'src'), str(REPO)]
import numpy as np
import torch
import g28_common as g
import stae_model as sm
from model.model import STAEFORMER_Model
from utils.window_stats import window_input_stats

PANEL = (2010, 2015, 2020, 2025)            # checkpoint years of the maintenance bundles
MAINTENANCE_SEEDS = (51, 52, 53)
ANNUAL_SEEDS = (51, 52, 53, 54, 55)
SEED_SHIFT = 9                              # evaluation seed s reads the checkpoint trained with seed s - 9
FIT_TAIL = 2048                             # training windows kept in a bundle
REFERENCE_NODE_LIMIT = 3000                 # the original model's CPU attention materializes N x N per head and step
STATE_KEYS = 103                            # entries of a STAEFormer checkpoint
G30_RTOL, G30_ATOL = 1e-4, 1e-3             # standalone model against the original code
GRAPH_YEARS = (2015, 2020, 2025)
ANNUAL_VERSION = 'evost-public-export-1'


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------

class Inputs:
    def __init__(self, opt, dataset):
        low = dataset.lower()
        base = Path(opt.data_root) / low
        self.dataset, self.low = dataset, low
        self.raw, self.fast, self.graph = base / 'RawData', base / 'FastData', base / 'graph'
        self.model = Path(opt.model_root) / dataset
        self.metadata = [Path(r) for r in (opt.metadata_root or [REPO / 'data' / 'sensor_metadata'])]
        self.tfnsw_nodes = Path(opt.tfnsw_nodes) if opt.tfnsw_nodes else None
        if opt.conf:
            conf = Path(opt.conf)
            assert conf.is_file(), 'no configuration at %s' % conf
        else:
            conf = REPO / 'conf' / dataset / ('retrain_staeformer_%s.json' % low)
            if not conf.is_file():                         # all datasets share the STAEFormer settings
                conf = REPO / 'conf' / 'PEMS03' / 'retrain_staeformer_pems03.json'
        self.conf_path = conf
        self.conf = json.loads(conf.read_text())
        self.logname = 'retrain_staeformer_%s' % low

    def model_args(self, n):
        a = argparse.Namespace(**self.conf)
        a.graph_size = n
        return a


def npy_shape(path, key):
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return tuple(int(v) for v in shape)


def raw_splits(inp, year):
    fast = inp.fast / ('%d.npz' % year)
    raw_path = inp.raw / ('%d.npz' % year)
    lengths = {s: npy_shape(fast, s + '_x')[0] + g.SPAN for s in ('train', 'val', 'test')}
    with np.load(raw_path) as z:
        raw = np.asarray(z['x'])
    assert raw.ndim == 2 and raw.shape[0] >= sum(lengths.values()), (raw.shape, lengths)
    out, offset = {}, 0
    for s in ('train', 'val', 'test'):
        out[s] = raw[offset:offset + lengths[s]]
        offset += lengths[s]
    return out, lengths, fast, raw_path


def check_cached_target_anchors(cached_y, raw_split, x_len, y_len, split):
    """The first and last cached target window must match the raw split (the benchmark's order check)."""
    expected_first = np.nan_to_num(raw_split[len(raw_split) - 1 - y_len:len(raw_split) - 1], nan=0.0, posinf=0.0, neginf=0.0)
    expected_last = np.nan_to_num(raw_split[x_len:x_len + y_len], nan=0.0, posinf=0.0, neginf=0.0)
    if (cached_y.shape[0] == 0 or not np.allclose(cached_y[0], expected_first, rtol=0.0, atol=1e-5)
            or not np.allclose(cached_y[-1], expected_last, rtol=0.0, atol=1e-5)):
        raise RuntimeError('%s RawData/FastData window order mismatch' % split)


def check_targets(fast, splits):
    """Anchor check on every split; full equality on val/test and the train tail."""
    report = {}
    for s, raw in splits.items():
        with np.load(fast, allow_pickle=True) as z:
            cached = z[s + '_y']
        check_cached_target_anchors(cached, raw, g.X_LEN, g.Y_LEN, s)
        n = g.n_windows(raw)
        assert cached.shape[0] == n, (s, cached.shape, n)
        count = n if s != 'train' else min(n, 4096)
        start = n - count
        _, y = g.get_windows(raw, start, count)
        theirs = cached[::-1][start:start + count]
        worst = 0.
        for i in range(0, count, 256):
            mine = np.nan_to_num(y[i:i + 256], nan=0., posinf=0., neginf=0.)
            worst = max(worst, float(np.abs(mine - theirs[i:i + 256].transpose(0, 2, 1)).max()))
        assert worst <= 1e-5, (s, worst)
        report[s] = dict(windows=int(n), compared=int(count), max_abs_target_difference=worst)
        del cached, theirs
    return report


def raw_input_windows(raw_split, x_len, y_len):
    """The benchmark's input windows of a split, latest first."""
    n_windows = len(raw_split) - x_len - y_len
    if n_windows <= 0:
        raise ValueError('split length %d is too short for x=%d, y=%d' % (len(raw_split), x_len, y_len))
    starts = np.arange(n_windows - 1, -1, -1, dtype=np.int64)
    indices = starts[:, None] + np.arange(x_len, dtype=np.int64)[None, :]
    return raw_split[indices]


def check_window_order(test):
    n_test = g.n_windows(test)
    theirs = raw_input_windows(test, g.X_LEN, g.Y_LEN)
    for start in (0, max(n_test - 64, 0)):
        x, _ = g.get_windows(test, start, min(64, n_test - start))
        ref = theirs[::-1][start:start + len(x)].transpose(0, 2, 1)
        assert np.array_equal(np.nan_to_num(x), np.nan_to_num(ref)), 'window order'


def metadata_ids(path, n):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'sensor_index' not in rows[0]:
        raise RuntimeError('invalid sensor metadata: %s' % path)
    rows.sort(key=lambda row: int(row['sensor_index']))
    if [int(row['sensor_index']) for row in rows] != list(range(n)):
        raise RuntimeError('%s does not match graph order: expected indices 0..%d' % (path, n - 1))
    field = 'node_id' if 'node_id' in rows[0] else 'station_id' if 'station_id' in rows[0] else None
    if field is None:
        raise RuntimeError('%s must contain either a node_id or station_id column' % path)
    ids = [str(row[field]).strip() for row in rows]
    if len(ids) != n or any(not i for i in ids) or len(set(ids)) != n:
        raise RuntimeError('%s has %d rows/%d unique IDs; expected %d' % (path, len(ids), len(set(ids)), n))
    return ids


def has_metadata(inp, year):
    return any((root / inp.low / str(year) / 'sensor_metadata.csv').is_file() for root in inp.metadata)


def sensor_ids(inp, year, n):
    """Stable sensor IDs of a year in graph-row order, or (None, reason)."""
    for root in inp.metadata:
        path = root / inp.low / str(year) / 'sensor_metadata.csv'
        if path.is_file():
            return metadata_ids(path, n), str(path)
    if inp.dataset == 'TFNSW' and inp.tfnsw_nodes is not None:
        path = inp.tfnsw_nodes / ('%d_nodes.csv' % year)
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: int(r['idx']))
        assert [int(r['idx']) for r in rows] == list(range(n)), 'TFNSW node order'
        ids = [str(r['node']).strip() for r in rows]
        assert len(set(ids)) == n and all(ids)
        return ids, str(path)
    return None, 'no sensor_metadata.csv for %s %d' % (inp.dataset, year)


def checkpoint_years(inp, backbone_seed):
    root = inp.model / ('%s-%d' % (inp.logname, backbone_seed))
    if not root.is_dir():
        return []
    return sorted(int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit() and any(d.glob('*.pkl')))


def lowest_loss_checkpoint(inp, backbone_seed, year):
    folder = inp.model / ('%s-%d' % (inp.logname, backbone_seed)) / str(year)
    files = sorted(folder.glob('*.pkl'), key=lambda f: float(f.stem))
    assert files, 'no checkpoint in %s' % folder
    return files[0]


def load_state(path):
    # The benchmark saves full pickles ({'model_state_dict': ...}); load only checkpoints you trust.
    return torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']


# ---------------------------------------------------------------------------------------------------------------------
# Reference forecasts of the original model code
# ---------------------------------------------------------------------------------------------------------------------

class _XOnly:
    """Minimal stand-in for a PyG batch when a model's forward only reads .x."""
    __slots__ = ('x',)

    def __init__(self, x):
        self.x = x


def _backbone_infer(backbone, data, adj, N, chunk=16):
    """Inference call of the benchmark: STAEFormer batches are chunked; one window passes through unchanged."""
    if not hasattr(getattr(backbone, 'backbone', None), 'adaptive_embedding'):
        return backbone(data, adj)
    x = data.x
    B = x.shape[0] // N
    if B <= chunk:
        return backbone(data, adj)
    outs = []
    for i in range(0, B, chunk):
        outs.append(backbone(_XOnly(x[i * N:(i + chunk) * N]), adj))
    return torch.cat(outs, dim=0)


def original_predict(inp, state, table, x_scaled):
    """Reference forecast of the benchmark's STAEFORMER_Model for windows x_scaled [W, n, 12] with identity `table`."""
    n = table.shape[1]
    model = STAEFORMER_Model(inp.model_args(n))
    full = dict(state)
    full[sm.EMBEDDING] = table
    model.load_state_dict(full, strict=True)
    model.eval()
    dummy = torch.empty(n, 0)                          # the model reads only adj.shape[0]
    with torch.no_grad():
        return torch.stack([_backbone_infer(model, _XOnly(x), dummy, n, 1).reshape(n, g.Y_LEN) for x in x_scaled])


def scaled(x_raw, mean, std):
    """Reference input built as the benchmark builds it."""
    return torch.from_numpy(np.nan_to_num((x_raw - mean) / std, nan=0., posinf=0., neginf=0.).astype(np.float32, copy=False))


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def reference_record(got, ref):
    d = (got - ref).abs()
    scale = float(ref.abs().max())
    worst = float(d.max())
    return dict(max_abs_difference=finite_or_none(worst), reference_abs_max=finite_or_none(scale),
                g30_allclose=bool(torch.allclose(got, ref, rtol=G30_RTOL, atol=G30_ATOL)),
                g30_tolerance_use=finite_or_none(float((d / (G30_ATOL + G30_RTOL * ref.abs())).max())),
                runtime_rule_use=finite_or_none(worst / (1e-3 * (1 + scale))),
                finite=bool(torch.isfinite(got).all() and torch.isfinite(ref).all()))


def save_new(target, arrays, compressed=False):
    assert not target.exists(), 'refusing to overwrite %s' % target
    (np.savez_compressed if compressed else np.savez)(target, **arrays)
    return dict(file=target.name, sha256=g.sha256_file(target), bytes=target.stat().st_size)


# ---------------------------------------------------------------------------------------------------------------------
# Maintenance bundles
# ---------------------------------------------------------------------------------------------------------------------

def maintenance_targets(inp, seeds):
    """Panel years with a same-year checkpoint for every seed and at least one earlier panel checkpoint."""
    years = {seed: checkpoint_years(inp, seed - SEED_SHIFT) for seed in seeds}
    common = sorted(set.intersection(*(set(v) for v in years.values())))
    targets = {y: sorted((s for s in common if s < y and s in PANEL), reverse=True) for y in common if y in PANEL}
    return {y: s for y, s in targets.items() if s}


def export_stae_year(inp, year, stale_years, seeds, out):
    started = time.time()
    splits, lengths, fast, raw_path = raw_splits(inp, year)
    n = int(splits['train'].shape[1])
    target_report = check_targets(fast, splits)
    mean, std = window_input_stats(splits['train'], g.X_LEN, g.Y_LEN)
    norm = {year: dict(mean=float(mean), std=float(std))}
    ids, id_note = sensor_ids(inp, year, n)
    assert ids is not None, id_note
    others, index, known = {}, {}, {}
    for other in stale_years:
        o_splits, o_lengths, _, o_raw = raw_splits(inp, other)
        m, sd = window_input_stats(o_splits['train'], g.X_LEN, g.Y_LEN)
        norm[other] = dict(mean=float(m), std=float(sd))
        o_nodes = int(o_splits['train'].shape[1])
        o_ids, o_note = sensor_ids(inp, other, o_nodes)
        assert o_ids is not None, o_note
        assert len(set(o_ids)) == o_nodes, 'duplicate sensor identities in %d' % other
        position = {v: i for i, v in enumerate(o_ids)}
        index[other] = np.asarray([position.get(v, -1) for v in ids], np.int64)
        known[other] = index[other] >= 0
        others[str(other)] = dict(nodes=o_nodes, known=int(known[other].sum()), new_nodes=int((~known[other]).sum()),
                                  removed_nodes=int(o_nodes - known[other].sum()), raw=str(o_raw), sensor_ids=o_note)
        del o_splits
    assert len(set(ids)) == n, 'duplicate sensor identities'

    test = splits['test']
    n_test = g.n_windows(test)
    check_window_order(test)

    tail = splits['train'][-(FIT_TAIL + g.SPAN):]
    arrays = dict(raw_train=tail.astype(np.float32), raw_val=splits['val'].astype(np.float32), raw_test=test.astype(np.float32),
                  sensor_ids=np.asarray(ids))
    for s, source in (('train', tail), ('val', splits['val']), ('test', test)):
        assert np.allclose(np.nan_to_num(arrays['raw_' + s]), np.nan_to_num(source), rtol=1e-6, atol=1e-4)
    for other in stale_years:
        arrays['new_mask/y%d' % other] = ~known[other]
        arrays['known/y%d' % other] = known[other]

    with_reference = n <= REFERENCE_NODE_LIMIT
    ref_index = np.asarray(sorted({0, n_test - 1}), np.int64)
    arrays['ref_index'] = ref_index if with_reference else np.zeros(0, np.int64)
    x_ref_raw = np.stack([g.get_windows(test, int(i), 1)[0][0] for i in ref_index])
    dropout = float(inp.conf['dropout'])
    ckpts, worst_reference = {}, 0.
    for seed in seeds:
        backbone_seed = seed - SEED_SHIFT
        for ckpt_year in [year] + list(stale_years):
            path = lowest_loss_checkpoint(inp, backbone_seed, ckpt_year)
            state = load_state(path)
            embedding = state[sm.EMBEDDING].to(torch.float32)
            shared = {k: v.to(torch.float32) for k, v in state.items() if k != sm.EMBEDDING}
            assert all(k.startswith(sm.SHARED_PREFIXES) for k in shared), [k for k in shared if not k.startswith(sm.SHARED_PREFIXES)][:5]
            if ckpt_year == year:
                assert embedding.shape[1] == n, (embedding.shape, n)
                table, mask = embedding, torch.ones(n, dtype=torch.bool)
            else:
                assert embedding.shape[1] == others[str(ckpt_year)]['nodes'], (embedding.shape, others[str(ckpt_year)])
                table, mask = sm.mapped_table(embedding, index[ckpt_year])
                assert torch.equal(mask, torch.from_numpy(known[ckpt_year]))
            prefix = '%d/y%d' % (seed, ckpt_year)
            for k, v in shared.items():
                arrays['ckpt/%s/%s' % (prefix, k)] = v.numpy()
            arrays['emb/%s' % prefix] = table.numpy()
            if with_reference:
                mine = sm.StaeModel(dropout=dropout).load_shared(shared)
                variants = [('year', year, None)] + ([] if ckpt_year == year else [('own', ckpt_year, None)])
                if ckpt_year == year - 5 and bool(mask.any()):      # no common-sensor variant when no sensor survived
                    variants.append(('common', year, mask))
                for tag, norm_year, subset in variants:
                    xs = scaled(x_ref_raw, norm[norm_year]['mean'], norm[norm_year]['std'])
                    x32 = g.normalize_x(x_ref_raw.astype(np.float32), norm[norm_year]['mean'], norm[norm_year]['std'])
                    if subset is not None:
                        xs, x32, tab = xs[:, subset], x32[:, subset], table[:, subset]
                    else:
                        tab = table
                    ref = original_predict(inp, state, tab, xs)
                    got = mine(x32, tab)
                    assert torch.allclose(got, ref, rtol=G30_RTOL, atol=G30_ATOL), ('standalone STAEFormer differs', prefix, tag)
                    worst_reference = max(worst_reference, float((got - ref).abs().max()))
                    arrays['ref/%s/%s' % (prefix, tag)] = ref.numpy().astype(np.float32)
            ckpts[prefix] = dict(year=ckpt_year, backbone_seed=backbone_seed, path=str(path), sha256=g.sha256_file(path),
                                 embedding_nodes=int(embedding.shape[1]))
            del state, embedding, shared, table
    meta = dict(layout='g30-stae', dataset=inp.dataset, year=year, nodes=n, ckpt_years=[year] + list(stale_years),
                stale_years=list(stale_years), norm={str(k): v for k, v in norm.items()}, mean=float(mean), std=float(std),
                seeds=list(seeds), x_len=g.X_LEN, y_len=g.Y_LEN, split_lengths=lengths, train_tail_rows=int(tail.shape[0]),
                dropout=dropout, checkpoints=ckpts, target_check=target_report, stale=others,
                new_mask_years=sorted(int(k) for k in known), sensor_ids=id_note, reference_windows=int(len(arrays['ref_index'])),
                reference_max_abs_difference=worst_reference,
                fallback='sensors absent in a checkpoint year receive that checkpoint\'s mean embedding per input step',
                sources=dict(raw=str(raw_path), raw_sha256=g.sha256_file(raw_path), fast=str(fast)),
                raw_nonfinite_fraction={s: float(1 - np.isfinite(splits[s]).mean()) for s in splits},
                raw_zero_fraction={s: float((splits[s] == 0).mean()) for s in splits},
                exporter='experiments/evost/export_bundles.py stae', original_files_mutated=False,
                torch=torch.__version__, numpy=np.__version__)
    arrays['meta_json'] = np.asarray(json.dumps(meta))
    saved = save_new(out / ('%s_%d.npz' % (inp.dataset, year)), arrays)   # float weights do not compress
    return dict(status='completed', **saved, nodes=n, year=year, stale_years=list(stale_years), norm=meta['norm'],
                reference_windows=meta['reference_windows'], reference_max_abs_difference=worst_reference,
                seconds=round(time.time() - started, 1))


# ---------------------------------------------------------------------------------------------------------------------
# Graph bundles
# ---------------------------------------------------------------------------------------------------------------------

def export_graph_year(inp, year, out):
    started = time.time()
    graph_path = inp.graph / ('%d_adj.npz' % year)
    adj = np.asarray(np.load(graph_path)['x'], np.float64)
    n = adj.shape[0]
    assert adj.shape == (n, n) and np.isfinite(adj).all()
    raw_nodes = npy_shape(inp.raw / ('%d.npz' % year), 'x')[1]
    assert raw_nodes == n, (raw_nodes, n)
    ids, id_note = sensor_ids(inp, year, n)
    assert ids is not None, id_note
    off = adj.copy()
    np.fill_diagonal(off, 0.)
    arrays = dict(adj_norm=g.normalize_adj(adj), sensor_ids=np.asarray(ids))      # float64 division, float32 storage
    meta = dict(layout='g29-graph', dataset=inp.dataset, year=year, nodes=n, sensor_ids=id_note,
                adjacency=dict(symmetric=g.is_symmetric(adj), offdiag_nonzero=int((off != 0).sum()),
                               diag_nonzero=int((np.diag(adj) != 0).sum()), density=float((off != 0).mean()),
                               min=float(adj.min()), max=float(adj.max())),
                normalization='row-normalized: adj / (row sum + 1e-6), float64 division, float32 storage',
                sources=dict(graph=str(graph_path), graph_sha256=g.sha256_file(graph_path)),
                exporter='experiments/evost/export_bundles.py graph', torch=torch.__version__, numpy=np.__version__)
    arrays['meta_json'] = np.asarray(json.dumps(meta))
    saved = save_new(out / ('%s_%d.npz' % (inp.dataset, year)), arrays, compressed=True)
    return dict(status='completed', **saved, nodes=n, year=year, seconds=round(time.time() - started, 1))


# ---------------------------------------------------------------------------------------------------------------------
# Annual bundles
# ---------------------------------------------------------------------------------------------------------------------

def annual_items(inp, opt):
    """One item per year: the checkpoint of every evaluation seed and the sensor-ID policy."""
    years = set(opt.years or [])
    if opt.cases:
        assert opt.ckpt_root, '--cases needs --ckpt-root, the directory the cases paths are relative to'
        items = {}
        for cell in json.loads(Path(opt.cases).read_text()):
            if cell['dataset'] != inp.dataset or (years and int(cell['year']) not in years):
                continue
            item = items.setdefault(int(cell['year']), dict(dataset=inp.dataset, year=int(cell['year']), nodes=int(cell['nodes']),
                                                              sensor_id_policy=cell['sensor_id_policy'], seeds=[]))
            item['seeds'].append(dict(evaluation_seed=int(cell['evaluation_seed']), checkpoint_seed=int(cell['checkpoint_seed']),
                                      path=str(Path(opt.ckpt_root) / cell['rel']), rel=cell['rel'], sha256=cell['sha256']))
        return [items[y] for y in sorted(items)]
    seeds = opt.seeds or list(ANNUAL_SEEDS)
    found = {seed: set(checkpoint_years(inp, seed - SEED_SHIFT)) for seed in seeds}
    common = sorted(set.intersection(*found.values()))
    out = []
    for year in common:
        if years and year not in years:
            continue
        entries = []
        for seed in seeds:
            path = lowest_loss_checkpoint(inp, seed - SEED_SHIFT, year)
            entries.append(dict(evaluation_seed=seed, checkpoint_seed=seed - SEED_SHIFT, path=str(path),
                                rel=str(path.relative_to(inp.model)), sha256=None))
        out.append(dict(dataset=inp.dataset, year=year, nodes=None, sensor_id_policy='optional', seeds=entries))
    return out


def resolve_ids(inp, year, n, policy):
    if inp.dataset == 'TFNSW' or has_metadata(inp, year):
        ids, source = sensor_ids(inp, year, n)
        assert ids is not None, source
        assert len(ids) == n and len(set(ids)) == n, 'sensor identities'
        text = '\n'.join(ids)
        return ids, dict(status='available', policy=policy, source=source, count=n, sha256=hashlib.sha256(text.encode()).hexdigest())
    if policy in ('registered_missing', 'optional'):
        return [], dict(status='unavailable', policy=policy, searched=[str(r) for r in inp.metadata], count=0, sha256=None)
    raise RuntimeError('sensor metadata missing for %s %d and the absence is not registered' % (inp.dataset, year))


def export_annual_year(inp, item, out):
    started = time.time()
    year = int(item['year'])
    splits, lengths, fast, raw_path = raw_splits(inp, year)
    n = int(splits['train'].shape[1])
    adj_nodes = int(npy_shape(inp.graph / ('%d_adj.npz' % year), 'x')[0])
    assert n == adj_nodes and item['nodes'] in (None, n), (n, adj_nodes, item['nodes'])
    target_report = check_targets(fast, splits)
    mean, std = window_input_stats(splits['train'], g.X_LEN, g.Y_LEN)
    norm = dict(mean=float(mean), std=float(std))
    ids, id_record = resolve_ids(inp, year, n, item['sensor_id_policy'])

    test = splits['test']
    n_test = g.n_windows(test)
    check_window_order(test)

    tail = splits['train'][-(FIT_TAIL + g.SPAN):]
    arrays = dict(raw_train=tail.astype(np.float32), raw_val=splits['val'].astype(np.float32), raw_test=test.astype(np.float32),
                  sensor_ids=np.asarray(ids, dtype=str))
    for s, source in (('train', tail), ('val', splits['val']), ('test', test)):
        assert np.allclose(np.nan_to_num(arrays['raw_' + s]), np.nan_to_num(source), rtol=1e-6, atol=1e-4)

    with_reference = n <= REFERENCE_NODE_LIMIT
    ref_index = np.asarray(sorted({0, n_test - 1}), np.int64)
    arrays['ref_index'] = ref_index if with_reference else np.zeros(0, np.int64)
    x_ref_raw = np.stack([g.get_windows(test, int(i), 1)[0][0] for i in ref_index])
    dropout = float(inp.conf['dropout'])
    ckpts, references, seeds = {}, {}, []
    for entry in sorted(item['seeds'], key=lambda e: int(e['evaluation_seed'])):
        seed, backbone_seed = int(entry['evaluation_seed']), int(entry['checkpoint_seed'])
        assert backbone_seed == seed - SEED_SHIFT, entry
        path = Path(entry['path'])
        digest = g.sha256_file(path)
        assert entry['sha256'] in (None, digest), ('checkpoint differs from the cases file', entry['rel'], digest)
        state = load_state(path)
        assert len(state) == STATE_KEYS, (entry['rel'], len(state))
        embedding = state[sm.EMBEDDING].to(torch.float32)
        assert tuple(embedding.shape) == (12, n, 80), (entry['rel'], tuple(embedding.shape))
        shared = {k: v.to(torch.float32) for k, v in state.items() if k != sm.EMBEDDING}
        assert all(k.startswith(sm.SHARED_PREFIXES) for k in shared), [k for k in shared if not k.startswith(sm.SHARED_PREFIXES)][:5]
        prefix = '%d/y%d' % (seed, year)
        for k, v in shared.items():
            arrays['ckpt/%s/%s' % (prefix, k)] = v.numpy()
        arrays['emb/%s' % prefix] = embedding.numpy()
        if with_reference:
            mine = sm.StaeModel(dropout=dropout).load_shared(shared)
            xs = scaled(x_ref_raw, mean, std)                           # input built as the benchmark builds it
            x32 = g.normalize_x(x_ref_raw.astype(np.float32), mean, std)   # input built as the runner builds it
            ref = original_predict(inp, state, embedding, xs)
            got = mine(x32, embedding)
            references[prefix] = reference_record(got, ref)
            arrays['ref/%s/year' % prefix] = ref.numpy().astype(np.float32)
        ckpts[prefix] = dict(year=year, evaluation_seed=seed, backbone_seed=backbone_seed, rel=entry['rel'], path=str(path), sha256=digest,
                             manifest_sha256=entry['sha256'], bytes=path.stat().st_size, state_keys=len(state),   # read by e24_annual.py's gate
                             embedding_nodes=int(embedding.shape[1]))
        seeds.append(seed)
        del state, embedding, shared
    meta = dict(layout='g30-stae', annual=dict(version=ANNUAL_VERSION, same_year_only=True,
                                               checkpoints='cases file, path and SHA-256' if item['seeds'][0]['sha256'] else 'lowest loss per seed and year'),
                dataset=inp.dataset, year=year, nodes=n, ckpt_years=[year], stale_years=[], norm={str(year): norm}, mean=norm['mean'], std=norm['std'],
                seeds=seeds, x_len=g.X_LEN, y_len=g.Y_LEN, split_lengths=lengths, train_tail_rows=int(tail.shape[0]), dropout=dropout,
                checkpoints=ckpts, target_check=target_report, stale={}, new_mask_years=[], sensor_ids=id_record,
                reference_windows=int(len(arrays['ref_index'])), reference=references,
                reference_note=('original model code vs standalone, first and last test window' if with_reference else
                                'not stored above %d sensors (the CPU attention of the original code materialises N x N)' % REFERENCE_NODE_LIMIT),
                sources=dict(raw=str(raw_path), raw_sha256=g.sha256_file(raw_path), fast=str(fast), fast_sha256=g.sha256_file(fast),
                             adjacency_nodes=adj_nodes),
                raw_dtype=str(splits['train'].dtype),
                raw_nonfinite_fraction={s: float(1 - np.isfinite(splits[s]).mean()) for s in splits},
                raw_zero_fraction={s: float((splits[s] == 0).mean()) for s in splits},
                exporter='experiments/evost/export_bundles.py annual', original_files_mutated=False,
                torch=torch.__version__, numpy=np.__version__)
    arrays['meta_json'] = np.asarray(json.dumps(meta))
    saved = save_new(out / ('%s_%d.npz' % (inp.dataset, year)), arrays)
    return dict(status='completed', **saved, nodes=n, seeds=seeds, norm=norm, sensor_ids=id_record['status'],
                reference_windows=meta['reference_windows'],
                reference_g30_allclose=all(r['g30_allclose'] for r in references.values()) if references else None,
                seconds=round(time.time() - started, 1))


# ---------------------------------------------------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------------------------------------------------

PATH_KEYS = ('path', 'raw', 'fast', 'graph', 'source', 'searched')


def stable_meta(value):
    """The bundle metadata without machine-specific paths, library versions and exporter labels."""
    if isinstance(value, dict):
        return {k: stable_meta(v) for k, v in value.items()
                if k not in ('torch', 'numpy', 'exporter', 'annual') + PATH_KEYS and not (k == 'sensor_ids' and isinstance(v, str))}
    if isinstance(value, list):
        return [stable_meta(v) for v in value]
    return value


def digest_bundle(path):
    z = np.load(path, allow_pickle=False)
    arrays = {}
    for key in sorted(z.files):
        if key == 'meta_json':
            continue
        a = np.ascontiguousarray(z[key])
        arrays[key] = [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]
    return dict(arrays=arrays, meta=stable_meta(json.loads(str(z['meta_json']))))


def digest(opt):
    out = {}
    for folder in opt.folders:
        for path in sorted(Path(folder).glob('*.npz')):
            if path.name in out:
                raise ValueError('Duplicate bundle name: ' + path.name)
            out[path.name] = digest_bundle(path)
            print(path.name, len(out[path.name]['arrays']), 'arrays', flush=True)
    Path(opt.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


def differences(a, b, where=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += differences(a.get(k), b.get(k), '%s.%s' % (where, k) if where else k)
        return out
    return [] if a == b else [dict(key=where, reference=str(a)[:200], new=str(b)[:200])]


def compare(opt):
    """Compare new bundles with reference digests of the same bundles: every array, and the metadata without paths and
    versions, must agree; graph bundles are compared on adj_norm and sensor_ids, which both must be present."""
    reference, new = json.loads(Path(opt.reference).read_text()), json.loads(Path(opt.new).read_text())
    if not reference or not new:
        raise ValueError('Both digest files must contain at least one bundle.')
    if set(reference) != set(new):
        raise ValueError('The reference and new digest files cover different bundles: only in reference %s, only in new %s'
                         % (sorted(set(reference) - set(new)), sorted(set(new) - set(reference))))
    report, ok = {}, True
    for name in sorted(new):
        r, b = reference[name], new[name]
        graph = b['meta'].get('layout') == 'g29-graph'
        if graph:                                    # the original graph bundles hold more arrays; these two are what experiments read
            required = {'adj_norm', 'sensor_ids'}
            if set(b['arrays']) != required or not required.issubset(r['arrays']):
                raise ValueError('A graph bundle must contain adj_norm and sensor_ids: ' + name)
            keys = sorted(required)
        else:
            keys = sorted(set(r['arrays']) | set(b['arrays']))
        arrays = [dict(key=k, reference=r['arrays'].get(k), new=b['arrays'].get(k)) for k in keys if r['arrays'].get(k) != b['arrays'].get(k)]
        meta = [] if graph else differences(r['meta'], b['meta'])
        report[name] = dict(status='identical' if not arrays and not meta else 'different', arrays_compared=len(keys),
                            array_differences=arrays, meta_differences=meta)
        ok &= not arrays and not meta
        print(name, report[name]['status'], '| arrays compared', len(keys), '| array differences', len(arrays),
              '| metadata differences', len(meta), flush=True)
    Path(opt.out).write_text(json.dumps(dict(status='identical' if ok else 'different', bundles=report), indent=1))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------------

def run_export(opt):
    inp = Inputs(opt, opt.dataset)
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    started = time.time()
    if opt.kind == 'stae':
        seeds = tuple(opt.seeds or MAINTENANCE_SEEDS)
        targets = maintenance_targets(inp, seeds)
        jobs = [(str(y), lambda y=y, s=s: export_stae_year(inp, y, s, seeds, out)) for y, s in targets.items()
                if not opt.years or y in opt.years]
    elif opt.kind == 'graph':
        jobs = [(str(y), lambda y=y: export_graph_year(inp, y, out)) for y in (opt.years or GRAPH_YEARS)]
    else:
        jobs = [(str(item['year']), lambda item=item: export_annual_year(inp, item, out)) for item in annual_items(inp, opt)]
    assert jobs, 'nothing to export for %s' % opt.dataset
    done = dict(kind=opt.kind, dataset=opt.dataset, conf=str(inp.conf_path), years={})
    for key, job in jobs:
        try:
            done['years'][key] = job()
        except Exception as error:
            done['years'][key] = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
        print(json.dumps({key: {k: v for k, v in done['years'][key].items() if k != 'traceback'}}), flush=True)
    ok = all(v['status'] == 'completed' for v in done['years'].values())
    done.update(status='completed' if ok else 'partial', seconds=round(time.time() - started, 1), torch=torch.__version__, numpy=np.__version__)
    g.save_json(out / ('export-%s-%s.json' % (opt.kind, opt.dataset)), done)
    return 0 if ok else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='kind', required=True)
    for kind in ('stae', 'graph', 'annual'):
        p = sub.add_parser(kind)
        p.add_argument('--dataset', required=True, help='PEMS03 ... PEMS12 or TFNSW')
        p.add_argument('--out', required=True, help='output directory; existing bundles are never overwritten')
        p.add_argument('--years', type=int, nargs='+', help='restrict to these target years')
        p.add_argument('--data-root', default=str(REPO / 'data'))
        p.add_argument('--model-root', default=str(REPO / 'log'))
        p.add_argument('--metadata-root', action='append', help='repeatable; searched in order (default data/sensor_metadata)')
        p.add_argument('--tfnsw-nodes', help='directory of <year>_nodes.csv for TFNSW years without a metadata file')
        p.add_argument('--conf', help='STAEFormer configuration for the model settings')
        if kind != 'graph':
            p.add_argument('--seeds', type=int, nargs='+', help='evaluation seeds (default 51-53 for stae, 51-55 for annual)')
        if kind == 'annual':
            p.add_argument('--cases', help='cases file that fixes every checkpoint by path and SHA-256')
            p.add_argument('--ckpt-root', help='directory the paths of the cases file are relative to')
    p = sub.add_parser('digest')
    p.add_argument('folders', nargs='+')
    p.add_argument('--out', required=True)
    p = sub.add_parser('compare')
    p.add_argument('reference', help='digest file of the reference bundles')
    p.add_argument('new', help='digest file of the new bundles')
    p.add_argument('--out', required=True)
    opt = ap.parse_args(argv)
    return dict(digest=digest, compare=compare).get(opt.kind, run_export)(opt)


if __name__ == '__main__':
    sys.exit(main())
