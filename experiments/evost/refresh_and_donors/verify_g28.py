"""Synthetic numerical and causality checks for the g28 code. Fails closed."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import g28_common as g


def check_windows():
    rng = np.random.RandomState(0)
    raw = rng.rand(80, 5).astype(np.float32)
    raw[7, 2] = np.nan
    x, y = g.get_windows(raw, 3, 20)
    for i in range(20):
        assert np.array_equal(x[i], raw[3 + i:3 + i + 12].T, equal_nan=True)
        assert np.array_equal(y[i], raw[3 + i + 12:3 + i + 24].T, equal_nan=True)
    assert g.n_windows(raw) == 80 - 24
    xn = g.normalize_x(x, 0.5, 2.)
    assert bool(torch.isfinite(xn).all()) and float(xn[4, 2, 0]) == 0.     # NaN input -> 0 after scaling
    return 'windows'


class _Conv(torch.nn.Module):          # transcription of the original BatchGCNConv(gcn=False)
    def __init__(self, i, o):
        super().__init__()
        self.weight_neigh = torch.nn.Linear(i, o, bias=True)
        self.weight_self = torch.nn.Linear(i, o, bias=False)

    def forward(self, x, adj):
        return self.weight_neigh(torch.matmul(adj, x)) + self.weight_self(x)


class _Reference(torch.nn.Module):     # transcription of the original TrafficStream_Model.forward
    def __init__(self):
        super().__init__()
        self.gcn1, self.gcn2 = _Conv(12, 64), _Conv(64, 12)
        self.tcn1 = torch.nn.Conv1d(1, 1, kernel_size=3, dilation=1, padding=1)
        self.fc = torch.nn.Linear(12, 12)
        self.activation = torch.nn.GELU()

    def forward(self, flat, adj):
        N = adj.shape[0]
        x = flat.reshape((-1, N, 12))
        x = torch.nn.functional.relu(self.gcn1(x, adj))
        x = x.reshape((-1, 1, 64))
        x = self.tcn1(x)
        x = x.reshape((-1, N, 64))
        x = self.gcn2(x, adj)
        x = x.reshape((-1, 12))
        x = x + flat
        return self.fc(self.activation(x))


def check_gcn(device):
    torch.manual_seed(1)
    model = _Reference().to(device).eval()
    assert set(model.state_dict()) == set(g.GCN_KEYS)
    adj = torch.rand(9, 9, device=device)
    adj = adj / (adj.sum(1, keepdim=True) + 1e-6)
    x = torch.randn(4, 9, 12, device=device)
    with torch.no_grad():
        ref = model(x.reshape(-1, 12), adj).reshape(4, 9, 12)
        mine = g.gcn_predict(dict(model.state_dict()), x, adj)
    assert torch.allclose(ref, mine, rtol=1e-5, atol=1e-5), float((ref - mine).abs().max())
    return 'gcn_functional_copy'


def check_rewiring():
    rng = np.random.RandomState(2)
    a = (rng.rand(60, 60) < .1).astype(np.float64)
    a = np.triu(a, 1)
    a = a + a.T
    a[np.arange(60), np.arange(60)] = 1.
    out, info = g.rewire(a, 5)
    assert info['feasible'] and info['changed_entries'] > 0 and g.is_symmetric(out)
    assert np.array_equal(np.diag(out), np.diag(a))
    w = np.triu(rng.rand(60, 60), 1) * np.triu(a, 1)
    w = w + w.T
    out, _ = g.rewire(w, 6)
    assert np.allclose(np.sort(out[out != 0]), np.sort(w[w != 0])) and g.is_symmetric(out)
    d = (rng.rand(40, 40) < .15).astype(np.float64)
    np.fill_diagonal(d, 0)
    out, info = g.rewire(d, 7)
    assert not info['symmetric'] and info['changed_entries'] > 0
    assert (out.sum(1) == d.sum(1)).all() and (out.sum(0) == d.sum(0)).all()
    mark = np.zeros(60, bool)
    mark[rng.permutation(60)[:12]] = True
    out, info = g.rewire(a, 8, restrict=mark)
    keep = ~(mark[:, None] | mark[None, :])
    assert np.array_equal(out[keep], a[keep]) and info['changed_entries'] > 0
    iso = g.isolate(a, mark)
    assert (g.support(iso)[mark].sum() == 0) and (g.support(iso)[:, mark].sum() == 0)
    assert np.array_equal(iso[np.ix_(~mark, ~mark)], a[np.ix_(~mark, ~mark)])
    perm, order = g.permute_nodes(a, 9)
    assert np.array_equal(perm, a[np.ix_(order, order)])
    k = g.knn_graph(rng.rand(30, 2) * 0.5 + np.array([-33., 151.]), 4)
    assert g.is_symmetric(k) and (g.support(k).sum(1) >= 4).all() and np.diag(k).sum() == 0
    return 'graph_controls'


def _data(T, n, seed, device):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(T, n, 12, generator=gen)
    b = 50 + 5 * torch.randn(T, n, 12, generator=gen)
    y = b + 3 * x[..., -1:] + torch.randn(T, n, 12, generator=gen)
    return x.to(device), b.to(device), y.to(device), torch.ones(T, n, dtype=torch.bool, device=device)


def check_rls(device):
    x, b, y, valid = _data(160, 4, 3, device)
    batch = g.NodeRLS(4, 12, 7., 2., 1., device)
    batch.fit(x, b, y, valid)
    seq = g.NodeRLS(4, 12, 7., 2., 1., device)
    for i in range(160):
        seq.update(seq.t(x[i]), seq.t(b[i]), seq.t(y[i]), valid[i])
    assert torch.allclose(batch.theta, seq.theta, rtol=1e-6, atol=1e-8)
    assert torch.allclose(batch.P, seq.P, rtol=1e-6, atol=1e-10) and float(batch.count[0]) == 160
    # direct ridge solution for one node
    f = g.features(batch.t(x[:, 0]), batch.t(b[:, 0]), 7.)
    direct = torch.linalg.solve(2. * torch.eye(29, dtype=torch.float64, device=device) + f.T @ f,
                                f.T @ ((batch.t(y[:, 0]) - batch.t(b[:, 0])) / 7.))
    assert torch.allclose(batch.theta[0], direct, rtol=1e-7, atol=1e-9)
    # masked fit ignores invalid node-windows
    part = valid.clone()
    part[:80, 1] = False
    masked = g.NodeRLS(4, 12, 7., 2., 1., device)
    masked.fit(x, b, y, part)
    alone = g.NodeRLS(4, 12, 7., 2., 1., device)
    alone.fit(x[80:], b[80:], y[80:], valid[80:])
    assert torch.allclose(masked.theta[1], alone.theta[1], rtol=1e-7, atol=1e-9) and float(masked.count[1]) == 80
    # per-node hyper-parameters equal separate models (virtual-node tiling)
    ridge = torch.tensor([1., 1., 100., 100.])
    forget = torch.tensor([1., .99, 1., .99])
    tiled = g.NodeRLS(4, 12, 7., ridge, forget, device)
    preds = {}
    g.rls_stream(tiled, 160, g.array_chunks(x[:, :1], b[:, :1], y[:, :1], valid[:, :1], None, device, 4), 12, 16,
                 lambda t, p, yy, c: preds.setdefault(t, p.clone()))
    for i in range(4):
        single = g.NodeRLS(1, 12, 7., float(ridge[i]), float(forget[i]), device)
        got = {}
        g.rls_stream(single, 160, g.array_chunks(x[:, :1], b[:, :1], y[:, :1], valid[:, :1], None, device), 12, 16,
                     lambda t, p, yy, c: got.setdefault(t, p.clone()))
        assert all(torch.allclose(preds[t][i], got[t][0], rtol=1e-9, atol=1e-9) for t in range(160))
    return 'rls_fit_update_mask_tiling'


def _stream(x, b, y, valid, device, delay, cadence, feedback=None):
    model = g.NodeRLS(x.shape[1], 12, 7., 1., .999, device)
    out = []
    g.rls_stream(model, len(x), g.array_chunks(x, b, y, valid, feedback, device), delay, cadence,
                 lambda t, p, yy, c: out.append(p.clone()))
    return torch.stack(out)


def check_causality(device):
    x, b, y, valid = _data(200, 3, 4, device)
    for delay, cadence in ((12, 64), (48, 64), (12, 1)):
        base = _stream(x, b, y, valid, device, delay, cadence)
        i0 = 70
        y2 = y.clone()
        y2[i0:] += 25.
        other = _stream(x, b, y2, valid, device, delay, cadence)
        same_until = i0 + delay                      # label of window i0 may first be used at i0 + delay
        assert torch.equal(base[:same_until], other[:same_until]), (delay, cadence)
        assert not torch.equal(base[same_until:], other[same_until:]), (delay, cadence)
        if cadence == 1:                             # freshest label age equals the delay exactly
            assert not torch.equal(base[same_until], other[same_until])
        else:                                        # batched: first use at the next multiple of the cadence
            first = -(-same_until // cadence) * cadence
            assert torch.equal(base[:first], other[:first]) and not torch.equal(base[first], other[first])
    # withheld feedback is never used
    fb = torch.ones(200, 3, dtype=torch.bool, device=device)
    fb[:, 2] = False
    held = _stream(x, b, y, valid, device, 12, 64, fb)
    assert torch.allclose(held[:, 2], b[:, 2].to(torch.float64))           # theta stays 0 -> backbone
    return 'label_causality_and_feedback'


def check_priors(device):
    x, b, y, valid = _data(300, 6, 5, device)
    donors = g.NodeRLS(6, 12, 7., 2., 1., device)
    donors.fit(x, b, y, valid)
    info, theta = g.donor_statistics(donors, np.arange(6))
    W = torch.full((2, 6), 1 / 6, dtype=torch.float64, device=device)
    eye = torch.eye(29, dtype=torch.float64, device=device)
    for kind in ('diag', 'pooled'):
        A, B = g.prior_terms(kind, W, info, theta)
        th, P = g.state_from_prior(2., 0., A, B, False)
        assert torch.equal(th, torch.zeros_like(th)) and torch.allclose(P, eye / 2.)
        th, P = g.state_from_prior(2., 16., A, B, True)
        assert torch.equal(th, torch.zeros_like(th)) and not torch.allclose(P, eye / 2.)
        assert abs(float(torch.diagonal(A, dim1=-2, dim2=-1).mean()) - 1.) < 1e-9
    # diag arm: explicit g27b formula
    A, B = g.prior_terms('diag', W, info, theta)
    diag = torch.diagonal(info, dim1=-2, dim2=-1).mean(0)
    prec = (diag / diag.mean()).clamp(1e-6, 1e4)
    mean = theta.mean(0)
    th, P = g.state_from_prior(2., 16., A, B, False)
    expect = (16. * prec[:, None] * mean) / (2. + 16. * prec[:, None])
    assert torch.allclose(th[0], expect, rtol=1e-9, atol=1e-12)
    assert torch.allclose(torch.diagonal(P[0]), 1. / (2. + 16. * prec), rtol=1e-9)
    # pooled arm with a very large strength equals ridge regression on the donors' union
    A, B = g.prior_terms('pooled', W, info, theta)
    th, _ = g.state_from_prior(2., 1e12, A, B, False)
    f = g.features(donors.t(x), donors.t(b), 7.).reshape(-1, 29)
    r = ((donors.t(y) - donors.t(b)) / 7.).reshape(-1, 12)
    union = torch.linalg.solve(6 * 2. * eye + f.T @ f, f.T @ r)
    assert torch.allclose(th[0], union, rtol=1e-5, atol=1e-7), float((th[0] - union).abs().max())
    # zero donor weight -> the node keeps the identity calibrator
    A, B = g.prior_terms('pooled', torch.zeros_like(W), info, theta)
    th, P = g.state_from_prior(2., 64., A, B, False)
    assert torch.equal(th, torch.zeros_like(th)) and torch.allclose(P, eye / 2.)
    return 'arrival_priors'


def check_metrics(device):
    acc = g.ErrAcc(2, device)
    y = torch.tensor([[[1.] * 12, [0.] * 12]], device=device)
    pred = torch.tensor([[[3.] * 12, [9.] * 12]], device=device)
    acc.add(pred, y)
    a = acc.arrays()
    m = g.summarize(a['abs'], a['sq'], a['cnt'])
    assert m['valid_targets'] == 12 and abs(m['H12_MAE'] - 2.) < 1e-12 and abs(m['Avg_MAE'] - 2.) < 1e-12
    assert g.summarize(a['abs'], a['sq'], a['cnt'], np.array([False, True])) is None
    return 'masked_metrics'


def check_g27b_regression(device, source):
    """Optional: the diag arm reproduces the original g27b inject_prior on the same state."""
    sys.path[:0] = [str(Path(source)), str(Path(source).parent.parent / 'graph-pilot25-20261007' / 'source')]
    from prior_core import inject_prior
    from graph_core import NodeRLS as Old
    x, b, y, valid = _data(300, 8, 6, device)
    old = Old(8, 12, 7., 2., 1., device)
    eligible = np.array([True] * 6 + [False] * 2)
    old.fit(dict(x=x.cpu().numpy(), b=b.cpu().numpy(), y=y.cpu().numpy()), eligible)
    adjacency = np.ones((8, 8)) - np.eye(8)
    new = g.NodeRLS(8, 12, 7., 2., 1., device)
    new.theta, new.P, new.count = old.theta.clone(), old.P.clone(), old.count.clone()
    assert inject_prior(old, np.array([6, 7]), adjacency, eligible, 16.) == 2
    info, theta = g.donor_statistics(new, np.arange(6))
    W = torch.full((2, 6), 1 / 6, dtype=torch.float64, device=device)
    th, P = g.state_from_prior(2., 16., *g.prior_terms('diag', W, info, theta), False)
    assert torch.allclose(th, old.theta[6:], rtol=1e-8, atol=1e-11), float((th - old.theta[6:]).abs().max())
    assert torch.allclose(P, old.P[6:], rtol=1e-8, atol=1e-11)
    return 'g27b_operator_regression'


def run(device, g27b_source=None):
    checks = [check_windows(), check_gcn(device), check_rewiring(), check_rls(device),
              check_causality(device), check_priors(device), check_metrics(device)]
    if g27b_source:
        checks.append(check_g27b_regression(device, g27b_source))
    return dict(status='passed', checks=checks, device=str(device), torch=torch.__version__, numpy=np.__version__)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--g27b-source', default=os.environ.get('G27B_SOURCE'))
    opt = ap.parse_args()
    dev = torch.device('cuda:0' if (opt.device == 'auto' and torch.cuda.is_available()) else
                       ('cpu' if opt.device == 'auto' else opt.device))
    result = run(dev, opt.g27b_source)
    if opt.out:
        g.save_json(opt.out, result)
    print(json.dumps(result), flush=True)
