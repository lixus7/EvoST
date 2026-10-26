"""Checks of the digest and compare commands of export_bundles.py on small synthetic bundles; no data needed.

Usage: python experiments/evost/test_export_bundles.py
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_bundles as E


def write_bundle(path, arrays, meta):
    np.savez(path, meta_json=np.asarray(json.dumps(meta)), **arrays)


def expect_error(argv, text):
    try:
        E.main(argv)
    except ValueError as error:
        assert text in str(error), error
        return
    raise AssertionError('no error for: ' + text)


def edited(source, target, change):
    d = json.loads(Path(source).read_text())
    change(d)
    Path(target).write_text(json.dumps(d))
    return str(target)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stae = dict(raw_train=np.arange(6, dtype=np.float32).reshape(3, 2), sensor_ids=np.asarray(['a', 'b']))
        graph = dict(adj_norm=np.eye(2, dtype=np.float32), sensor_ids=np.asarray(['a', 'b']))
        for side in ('ref', 'new'):
            (tmp / side).mkdir()
            write_bundle(tmp / side / 'X_2020.npz', stae, dict(layout='g30-stae', year=2020, sources=dict(raw='/data/%s/2020.npz' % side)))
            graph_arrays = dict(graph, raw_test=np.zeros((2, 2), np.float32)) if side == 'ref' else graph   # originals hold more arrays
            write_bundle(tmp / side / 'G_2020.npz', graph_arrays, dict(layout='g29' if side == 'ref' else 'g29-graph', year=2020))
        ref, new = str(tmp / 'ref.json'), str(tmp / 'new.json')
        assert E.main(['digest', str(tmp / 'ref'), '--out', ref]) == 0
        assert E.main(['digest', str(tmp / 'new'), '--out', new]) == 0
        report = str(tmp / 'report.json')

        # identical arrays; differing paths in the metadata are ignored
        assert E.main(['compare', ref, new, '--out', report]) == 0
        # a changed array or a changed metadata value is reported
        bad = edited(new, tmp / 'array.json', lambda d: d['X_2020.npz']['arrays']['raw_train'].__setitem__(2, '0' * 64))
        assert E.main(['compare', ref, bad, '--out', report]) == 1
        bad = edited(new, tmp / 'meta.json', lambda d: d['X_2020.npz']['meta'].__setitem__('year', 2021))
        assert E.main(['compare', ref, bad, '--out', report]) == 1
        # nothing to compare, or different sets of bundles
        expect_error(['compare', ref, edited(new, tmp / 'empty.json', lambda d: d.clear()), '--out', report], 'at least one bundle')
        expect_error(['compare', ref, edited(new, tmp / 'missing.json', lambda d: d.pop('G_2020.npz')), '--out', report],
                     'different bundles')
        # a graph bundle without sensor IDs
        expect_error(['compare', ref, edited(new, tmp / 'graph.json', lambda d: d['G_2020.npz']['arrays'].pop('sensor_ids')),
                      '--out', report], 'adj_norm and sensor_ids')
        # two folders with a bundle of the same name
        expect_error(['digest', str(tmp / 'ref'), str(tmp / 'new'), '--out', str(tmp / 'both.json')], 'Duplicate bundle name')
    print('export_bundles digest and compare: all checks passed')


if __name__ == '__main__':
    main()
