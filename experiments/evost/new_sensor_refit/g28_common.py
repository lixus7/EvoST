"""g28 shared code. Standalone: needs only numpy and torch.

All quantities are computed causally. Raw splits are [T, N]; chronological
window i uses raw[i:i+12] as input and raw[i+12:i+24] as target, which is the
order verified against the original FastData by export_bundle.py.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

X_LEN = Y_LEN = 12
SPAN = X_LEN + Y_LEN
D_FEAT = 29


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_bundle(path):
    z = np.load(path, allow_pickle=False)
    return z, json.loads(str(z['meta_json']))


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def n_windows(raw):
    return raw.shape[0] - SPAN


def window_views(raw):
    """Zero-copy views: xv[i] = raw[i:i+12].T -> [N, 12]; target of i is xv[i+12]."""
    return np.lib.stride_tricks.sliding_window_view(raw, X_LEN, axis=0)


def get_windows(raw, start, count):
    """Chronological windows [start, start+count): x_raw, y_raw as [count, N, 12] (NaN kept)."""
    assert start >= 0 and count > 0 and start + count <= n_windows(raw), (start, count, raw.shape)
    v = window_views(raw)
    return v[start:start + count], v[start + X_LEN:start + X_LEN + count]


def normalize_x(x_raw, mean, std):
    x = (torch.from_numpy(np.array(x_raw, dtype=np.float32)) - float(mean)) / float(std)
    return torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)


def clean_y(y_raw):
    """Pipeline convention: non-finite targets become 0 and zeros are masked in evaluation."""
    return torch.nan_to_num(torch.from_numpy(np.array(y_raw, dtype=np.float32)),
                            nan=0., posinf=0., neginf=0.)


def target_valid(y):
    return torch.isfinite(y) & (y != 0)


# --------------------------------------------------------------------------
# Frozen GCN backbone: functional copy of TrafficStream_Model.forward
# --------------------------------------------------------------------------

GCN_KEYS = ('gcn1.weight_neigh.weight', 'gcn1.weight_neigh.bias', 'gcn1.weight_self.weight',
            'gcn2.weight_neigh.weight', 'gcn2.weight_neigh.bias', 'gcn2.weight_self.weight',
            'tcn1.weight', 'tcn1.bias', 'fc.weight', 'fc.bias')


def normalize_adj(a):
    a = np.asarray(a, np.float64)
    return (a / (a.sum(1, keepdims=True) + 1e-6)).astype(np.float32)


def gcn_predict(W, x, adj, tcn_padding=1, tcn_dilation=1):
    """x [B, N, 12] normalized inputs, adj [N, N] row-normalized -> [B, N, 12] in target units."""
    h = F.linear(torch.matmul(adj, x), W['gcn1.weight_neigh.weight'], W['gcn1.weight_neigh.bias'])
    h = F.relu(h + F.linear(x, W['gcn1.weight_self.weight']))
    B, N, C = h.shape
    h = F.conv1d(h.reshape(B * N, 1, C), W['tcn1.weight'], W['tcn1.bias'],
                 padding=tcn_padding, dilation=tcn_dilation).reshape(B, N, C)
    o = F.linear(torch.matmul(adj, h), W['gcn2.weight_neigh.weight'], W['gcn2.weight_neigh.bias'])
    o = o + F.linear(h, W['gcn2.weight_self.weight']) + x
    return F.linear(F.gelu(o), W['fc.weight'], W['fc.bias'])


def bundle_weights(z, seed, tag, device):
    prefix = 'ckpt/%d/%s/' % (seed, tag)
    return {k: torch.as_tensor(z[prefix + k], dtype=torch.float32, device=device) for k in GCN_KEYS}


# --------------------------------------------------------------------------
# Graph controls (numpy only)
# --------------------------------------------------------------------------

def support(a):
    s = np.asarray(a) != 0
    s = s.copy()
    np.fill_diagonal(s, False)
    return s


def is_symmetric(a):
    a = np.asarray(a, np.float64)
    return bool(np.allclose(a, a.T, rtol=0, atol=1e-9))


def rewire(a, seed, restrict=None, swaps_per_edge=5, tries_per_edge=40):
    """Degree-preserving rewiring of off-diagonal entries; each weight travels with its edge.

    Symmetric input: undirected double-edge swap, degree of every node preserved.
    Asymmetric input: directed swap of heads, in- and out-degree preserved.
    restrict: optional bool [N]; only edges incident to a marked node are moved and
    every moved edge stays incident to a marked node. Diagonal is untouched.
    """
    a = np.asarray(a, np.float64)
    n = a.shape[0]
    rng = np.random.RandomState(seed)
    sym = is_symmetric(a)
    s = support(a)
    if sym:
        iu, ju = np.nonzero(np.triu(s, 1))
    else:
        iu, ju = np.nonzero(s)
    edges = np.stack([iu, ju], 1).astype(np.int64)
    weights = a[iu, ju].copy()
    m = len(edges)
    if restrict is not None:
        restrict = np.asarray(restrict, bool)
        movable = np.flatnonzero(restrict[edges[:, 0]] | restrict[edges[:, 1]])
        if sym:
            # Orient movable edges with a marked endpoint first, so swapping tails keeps incidence.
            for e in movable:
                u, v = edges[e]
                if not restrict[u] or (restrict[v] and rng.rand() < .5):
                    edges[e] = (v, u)
    else:
        movable = np.arange(m)
    key = (lambda u, v: (min(u, v), max(u, v))) if sym else (lambda u, v: (u, v))
    present = {key(int(u), int(v)) for u, v in edges}
    target, done, tries = swaps_per_edge * len(movable), 0, 0
    limit = tries_per_edge * max(len(movable), 1)
    while done < target and tries < limit and len(movable) >= 2:
        tries += 1
        e1, e2 = movable[rng.randint(len(movable))], movable[rng.randint(len(movable))]
        if e1 == e2:
            continue
        u1, v1 = map(int, edges[e1])
        u2, v2 = map(int, edges[e2])
        if sym and restrict is None and rng.rand() < .5:
            u2, v2 = v2, u2
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue
        if restrict is not None and not sym:
            if not ((restrict[u1] or restrict[v2]) and (restrict[u2] or restrict[v1])):
                continue
        k1, k2 = key(u1, v2), key(u2, v1)
        if k1 in present or k2 in present:
            continue
        present.discard(key(u1, v1)); present.discard(key(u2, v2))
        present.add(k1); present.add(k2)
        edges[e1] = (u1, v2); edges[e2] = (u2, v1)
        done += 1
    out = np.zeros_like(a)
    out[edges[:, 0], edges[:, 1]] = weights
    if sym:
        out[edges[:, 1], edges[:, 0]] = weights
    out[np.arange(n), np.arange(n)] = np.diag(a)
    s2 = support(out)
    assert (s2.sum(1) == s.sum(1)).all() and (s2.sum(0) == s.sum(0)).all(), 'degree changed'
    assert s2.sum() == s.sum()
    changed = int((s2 != s).sum())
    if restrict is not None:
        untouched = ~(restrict[:, None] | restrict[None, :])
        assert np.array_equal(out[untouched], a[untouched]), 'edge outside the marked set moved'
    return out, dict(symmetric=sym, edges=int(m), movable=int(len(movable)), swaps=int(done),
                     tries=int(tries), changed_entries=changed, feasible=bool(done > 0))


def permute_nodes(a, seed):
    """Relabel nodes: every node receives another node's neighbourhood. Always feasible."""
    perm = np.random.RandomState(seed).permutation(a.shape[0])
    return np.asarray(a)[np.ix_(perm, perm)].copy(), perm


def isolate(a, mask):
    """Remove every off-diagonal entry in rows or columns of the marked nodes."""
    a = np.asarray(a, np.float64).copy()
    mask = np.asarray(mask, bool)
    d = np.diag(a).copy()
    a[mask, :] = 0
    a[:, mask] = 0
    a[np.arange(len(d)), np.arange(len(d))] = d
    return a


def knn_graph(coords, k=8):
    """Symmetric binary union-kNN on great-circle distance (same rule as g25)."""
    xy = np.radians(np.asarray(coords, np.float64))
    dlat = xy[:, None, 0] - xy[None, :, 0]
    dlon = xy[:, None, 1] - xy[None, :, 1]
    dist = np.sin(dlat / 2) ** 2 + np.cos(xy[:, None, 0]) * np.cos(xy[None, :, 0]) * np.sin(dlon / 2) ** 2
    np.fill_diagonal(dist, np.inf)
    n = len(xy)
    a = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in np.argsort(dist[i], kind='stable')[:min(k, n - 1)]:
            a[i, j] = a[j, i] = 1.
    return a


# --------------------------------------------------------------------------
# Feature RLS (same 29 features and recursion as the g25 NodeRLS)
# --------------------------------------------------------------------------

def features(x, b, scale):
    t = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)
    t = t - t.mean()
    mean = x.mean(-1, keepdim=True)
    stats = torch.cat((x[..., -1:], mean, x.std(-1, unbiased=False, keepdim=True),
                       ((x - mean) * t).sum(-1, keepdim=True) / (t * t).sum()), -1)
    return torch.cat((b / scale, x, stats, torch.ones_like(x[..., :1])), -1)


class NodeRLS:
    """Per-node multi-output RLS. theta [n, 29, h], P [n, 29, 29], float64.

    ridge and forget may be scalars or per-node vectors, so that several configurations
    or several prior arms can be streamed side by side as independent virtual nodes.
    """

    def __init__(self, n, h, scale, ridge, forget, device):
        self.scale, self.device = float(scale), device
        self.ridge = torch.as_tensor(ridge, dtype=torch.float64, device=device).expand(n).clone()
        self.forget = torch.as_tensor(forget, dtype=torch.float64, device=device).expand(n).clone()
        self.theta = torch.zeros(n, D_FEAT, h, dtype=torch.float64, device=device)
        self.P = torch.eye(D_FEAT, dtype=torch.float64, device=device).repeat(n, 1, 1) / self.ridge[:, None, None]
        self.count = torch.zeros(n, dtype=torch.float64, device=device)

    def t(self, a):
        return torch.as_tensor(a, dtype=torch.float64, device=self.device)

    def fit(self, x, b, y, valid=None, batch=64):
        """Exact ridge on [T, n, 12] arrays; valid [T, n] bool selects node-windows."""
        A = torch.linalg.inv(self.P)
        rhs = A @ self.theta
        used = torch.zeros_like(self.count)
        for s in range(0, len(x), batch):
            xx, bb, yy = self.t(x[s:s + batch]), self.t(b[s:s + batch]), self.t(y[s:s + batch])
            f = features(xx, bb, self.scale)
            r = (yy - bb) / self.scale
            if valid is not None:
                w = torch.as_tensor(valid[s:s + batch], device=self.device).to(torch.float64)
                f = f * w.unsqueeze(-1)
                used += w.sum(0)
            else:
                used += len(xx)
            A += torch.einsum('bni,bnj->nij', f, f)
            rhs += torch.einsum('bni,bnh->nih', f, r)
        A = (A + A.transpose(-1, -2)) * .5
        P = torch.linalg.inv(A)
        self.P = (P + P.transpose(-1, -2)) * .5
        self.theta = torch.linalg.solve(A, rhs)
        self.count = self.count + used
        assert bool(torch.isfinite(self.theta).all())

    def update(self, x, b, y, eligible):
        ids = torch.where(eligible)[0]
        if len(ids) == 0:
            return
        lam = self.forget[ids]
        f = features(x[ids], b[ids], self.scale)
        v = (self.P[ids] @ f.unsqueeze(-1)).squeeze(-1)
        den = lam + (f * v).sum(-1)
        gain = v / den.unsqueeze(-1)
        err = (y[ids] - b[ids]) / self.scale - torch.einsum('nd,ndh->nh', f, self.theta[ids])
        self.theta[ids] += gain.unsqueeze(-1) * err.unsqueeze(-2)
        P = (self.P[ids] - gain.unsqueeze(-1) * v.unsqueeze(-2)) / lam[:, None, None]
        self.P[ids] = (P + P.transpose(-1, -2)) * .5
        self.count[ids] = lam * self.count[ids] + 1

    def predict(self, x, b):
        return b + self.scale * torch.einsum('nd,ndh->nh', features(x, b, self.scale), self.theta)


def window_valid(y):
    """A node-window may be used for fitting when all 12 targets are finite and non-zero."""
    return target_valid(y).all(-1)


def array_chunks(x, b, y, valid, feedback=None, device='cpu', repeat=1):
    """chunk_fn over in-memory [T, n, 12] arrays; repeat tiles the node axis (virtual nodes)."""
    def to(v, dtype, s):
        v = torch.as_tensor(v[s] if torch.is_tensor(v) else np.ascontiguousarray(v[s]), device=device).to(dtype)
        return v if repeat == 1 else v.repeat(*((1, repeat) + (1,) * (v.dim() - 2)))

    def chunk_fn(s):
        return (to(x, torch.float64, s), to(b, torch.float64, s), to(y, torch.float64, s),
                to(valid, torch.bool, s), None if feedback is None else to(feedback, torch.bool, s))
    return chunk_fn


def rls_stream(model, T, chunk_fn, delay=12, cadence=64, on_window=None, chunk=512):
    """Causal prequential stream over T windows.

    chunk_fn(slice) returns (x, b, y, valid, feedback) tensors on the model device for the
    windows of that slice: x, b, y are [len, n, 12] float64, valid/feedback [len, n] bool
    (feedback may be None). At window t the model has absorbed exactly the windows
    i <= t' - delay, where t' is the latest multiple of `cadence` not exceeding t, and the
    forecast of t is issued before any label of window t is used. Labels are used only
    where valid (and feedback) hold. on_window(t, pred, y, count) receives the float64
    forecast, the targets and the per-node absorbed-label count at forecast time.
    """
    assert delay >= Y_LEN, 'a complete target window must have matured'
    used, cache = -1, {}

    def block(i):
        c = i // chunk
        if c not in cache:
            while len(cache) >= 2:          # updates trail forecasts by `delay`: keep two chunks
                del cache[min(cache)]
            cache[c] = chunk_fn(slice(c * chunk, min((c + 1) * chunk, T)))
        return cache[c], i - c * chunk

    for t in range(T):
        if t % cadence == 0:
            for i in range(used + 1, t - delay + 1):
                (xx, bb, yy, vv, ff), j = block(i)
                model.update(xx[j], bb[j], yy[j], vv[j] if ff is None else (vv[j] & ff[j]))
            used = max(used, t - delay)
        if on_window is not None:
            (xx, bb, yy, vv, ff), j = block(t)
            on_window(t, model.predict(xx[j], bb[j]), yy[j], model.count)


# --------------------------------------------------------------------------
# Error accumulation
# --------------------------------------------------------------------------

class ErrAcc:
    """Per-node, per-horizon sums of masked absolute and squared error."""

    def __init__(self, n, device, h=Y_LEN):
        self.abs = torch.zeros(n, h, dtype=torch.float64, device=device)
        self.sq = torch.zeros(n, h, dtype=torch.float64, device=device)
        self.cnt = torch.zeros(n, h, dtype=torch.float64, device=device)

    def add(self, pred, y):
        """pred, y [..., n, h]; leading dims are summed."""
        valid = target_valid(y)
        e = torch.where(valid, (pred - y).to(torch.float64), torch.zeros((), dtype=torch.float64, device=y.device))
        extra = tuple(range(e.dim() - 2))
        self.abs += e.abs().sum(extra) if extra else e.abs()
        self.sq += (e * e).sum(extra) if extra else e * e
        self.cnt += valid.to(torch.float64).sum(extra) if extra else valid.to(torch.float64)

    def arrays(self):
        return {k: getattr(self, k).cpu().numpy() for k in ('abs', 'sq', 'cnt')}


def summarize(abs_sum, sq_sum, cnt, node_mask=None):
    """Metrics for a node cohort from per-node per-horizon sums (numpy [n, h])."""
    if node_mask is not None:
        abs_sum, sq_sum, cnt = abs_sum[node_mask], sq_sum[node_mask], cnt[node_mask]
    a, s, c = abs_sum.sum(0), sq_sum.sum(0), cnt.sum(0)
    if c.sum() == 0:
        return None
    ca, cs, cc = np.cumsum(a), np.cumsum(s), np.cumsum(c)
    cum_mae = ca / np.maximum(cc, 1)
    out = dict(valid_targets=int(c.sum()), Avg_MAE=float(cum_mae.mean()))
    for h in (3, 6, 12):
        out['H%d_MAE' % h] = float(ca[h - 1] / max(cc[h - 1], 1))
        out['H%d_RMSE' % h] = float(np.sqrt(cs[h - 1] / max(cc[h - 1], 1)))
    return out


# --------------------------------------------------------------------------
# Arrival priors written into a new node's normal equations
# --------------------------------------------------------------------------

def donor_statistics(model, donor_ids):
    """Per-observation information matrices and normal-equation right-hand sides of donors."""
    ids = torch.as_tensor(donor_ids, dtype=torch.long, device=model.device)
    A = torch.linalg.inv(model.P[ids])
    A = (A + A.transpose(-1, -2)) * .5
    c = model.count[ids].clamp_min(1)
    info = A / c[:, None, None]
    return info, model.theta[ids]


def prior_terms(kind, W, info, theta):
    """Prior precision and right-hand side with unit mean diagonal.

    kind='diag'  : weighted mean of donor coefficients with a diagonal information
                   precision (the g27b operator).
    kind='pooled': weighted sum of donor normal equations, i.e. the ridge solution on
                   the donors' pooled evidence (full matrices).
    W [m, D] holds non-negative donor weights whose rows sum to one (or are all zero).
    """
    if kind == 'diag':
        prec = W @ torch.diagonal(info, dim1=-2, dim2=-1)
        prec = (prec / prec.mean(-1, keepdim=True).clamp_min(1e-12)).clamp(1e-6, 1e4)
        mean = torch.einsum('nm,mdh->ndh', W, theta)
        return torch.diag_embed(prec), prec.unsqueeze(-1) * mean
    assert kind == 'pooled', kind
    Abar = torch.einsum('nm,mij->nij', W, info)
    Bbar = torch.einsum('nm,mih->nih', W, info @ theta)
    norm = torch.diagonal(Abar, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    return Abar / norm[:, None, None], Bbar / norm[:, None, None]


def state_from_prior(ridge, tau, A_prior, B_prior, zero_mean):
    """(theta, P) of a node that has no own label yet; tau=0 or missing donors -> identity calibrator."""
    eye = torch.eye(D_FEAT, dtype=torch.float64, device=A_prior.device)
    A = ridge * eye + tau * A_prior
    A = (A + A.transpose(-1, -2)) * .5
    rhs = torch.zeros_like(B_prior) if zero_mean else tau * B_prior
    P = torch.linalg.inv(A)
    return torch.linalg.solve(A, rhs), (P + P.transpose(-1, -2)) * .5
