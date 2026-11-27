"""SpecGap demo on synthetic graphs (CPU, a few seconds, no downloads).

ID graphs are rings with a few chords (near-regular, like ring molecules). OOD graphs have the same numbers of
nodes and edges but grow around hubs by preferential attachment, so the gap between their two largest Laplacian
eigenvalues is larger and more varied (the situation of Fig. 1 in the paper). Node attributes are Gaussian for
both. A fixed, randomly initialised two-layer GCN-style map embeds every graph; SpecGap is applied to the
layer-1 node features (the best position in Tab. 5) and the adjusted features go through layer 2 (Eq. 6).
Graph embeddings are scored with an SSD-style Mahalanobis distance and with LOF, both fitted on ID training
graphs, and the AUROC (OOD = positive class) is printed without and with SpecGap.

    python demo.py
"""
import numpy as np
import torch
from scipy.stats import rankdata

from specgap import specgap, spectral_gap

D_IN, D_HID = 16, 32
rng = np.random.default_rng(0)
torch.manual_seed(0)
W1 = torch.randn(D_IN, D_HID) / D_IN ** 0.5           # fixed random weights of the GNN-like feature map
W2 = torch.randn(D_HID, D_HID) / D_HID ** 0.5


# ---------------------------------------------------------------------------------------------- synthetic graphs
def ring_graph(n, n_chords):
    """Cycle C_n plus random chords between nodes at ring distance >= 3 (edges as (i, j), i < j)."""
    edges = {(i, i + 1) for i in range(n - 1)} | {(0, n - 1)}
    while len(edges) < n + n_chords:
        a, b = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
        if min(b - a, n - b + a) >= 3:
            edges.add((a, b))
    return sorted(edges)


def hub_graph(n, n_edges):
    """Preferential-attachment tree, topped up with degree-biased edges until it has n_edges edges."""
    edges, deg = set(), np.zeros(n)
    for v in range(1, n):
        p = deg[:v] + 1.0
        u = int(rng.choice(v, p=p / p.sum()))
        edges.add((u, v))
        deg[u] += 1
        deg[v] += 1
    while len(edges) < n_edges:
        p = deg + 1.0
        a, b = sorted(int(v) for v in rng.choice(n, size=2, replace=False, p=p / p.sum()))
        if (a, b) not in edges:
            edges.add((a, b))
            deg[a] += 1
            deg[b] += 1
    return sorted(edges)


def sample(kind, count):
    """List of (edge_index, node features) pairs; ID and OOD graphs have matched node and edge counts."""
    graphs = []
    for _ in range(count):
        n, c = int(rng.integers(16, 29)), int(rng.integers(2, 6))
        edges = ring_graph(n, c) if kind == 'id' else hub_graph(n, n + c)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.from_numpy(rng.standard_normal((n, D_IN))).float()
        graphs.append((edge_index, x))
    return graphs


# ---------------------------------------------------------------------------------------------- feature map
def gcn_norm(edge_index, n):
    """GCN propagation matrix D^-1/2 (A + I) D^-1/2."""
    A = torch.eye(n)
    A[edge_index[0], edge_index[1]] = 1.0
    A[edge_index[1], edge_index[0]] = 1.0
    d = A.sum(1).rsqrt()
    return d[:, None] * A * d[None, :]


def embed(edge_index, x, kind=None):
    """Graph embedding of the fixed 2-layer map; kind=None: original, else SpecGap with that Laplacian."""
    P = gcn_norm(edge_index, x.shape[0])
    h = torch.tanh(P @ x @ W1)                                   # layer 1
    if kind is not None:
        h = specgap(h, edge_index=edge_index, kind=kind)         # X' = X - dl u v^T        (Eqs. 3-5)
    h = torch.tanh(P @ h @ W2)                                   # layer 2 on X'            (Eq. 6)
    return torch.cat([h.mean(0), h.max(0).values]).numpy()       # mean + max readout


# ---------------------------------------------------------------------------------------------- OOD scores
def ssd_score(train, test):
    """SSD-style score: Mahalanobis distance to the ID training embeddings (larger = more OOD)."""
    mu = train.mean(0)
    cov = np.cov(train, rowvar=False) + 1e-3 * np.eye(train.shape[1])
    d = test - mu
    return np.einsum('ij,jk,ik->i', d, np.linalg.inv(cov), d)


def lof_score(train, test, k=10):
    """Local outlier factor of test points w.r.t. the ID training embeddings (larger = more OOD)."""
    def knn(a, b, exclude_self):
        dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        if exclude_self:
            np.fill_diagonal(dist, np.inf)
        idx = np.argsort(dist, axis=1)[:, :k]
        return np.take_along_axis(dist, idx, axis=1), idx

    d_tr, i_tr = knn(train, train, True)
    k_dist = d_tr[:, -1]                                         # k-distance of every training point
    lrd_tr = 1.0 / (np.maximum(d_tr, k_dist[i_tr]).mean(1) + 1e-12)   # local reachability density
    d_te, i_te = knn(test, train, False)
    lrd_te = 1.0 / (np.maximum(d_te, k_dist[i_te]).mean(1) + 1e-12)
    return lrd_tr[i_te].mean(1) / lrd_te


def auroc(s_id, s_ood):
    """Area under the ROC curve with OOD as the positive class (Mann-Whitney U statistic)."""
    r = rankdata(np.concatenate([s_id, s_ood]))
    n_id, n_ood = len(s_id), len(s_ood)
    return (r[n_id:].sum() - n_ood * (n_ood + 1) / 2) / (n_id * n_ood)


# ---------------------------------------------------------------------------------------------- main
def main():
    id_train, id_test, ood_test = sample('id', 300), sample('id', 150), sample('ood', 150)
    print('SpecGap demo: %d ID train / %d ID test / %d OOD test synthetic graphs'
          % (len(id_train), len(id_test), len(ood_test)))
    for kind in ('unnormalized', 'normalized'):
        g_id = [spectral_gap(e, x.shape[0], kind=kind)[0] for e, x in id_test]
        g_ood = [spectral_gap(e, x.shape[0], kind=kind)[0] for e, x in ood_test]
        print('spectral gap, %-12s L:  ID %.3f +- %.3f   OOD %.3f +- %.3f'
              % (kind, np.mean(g_id), np.std(g_id), np.mean(g_ood), np.std(g_ood)))
    print('\nAUROC (OOD = positive)        SSD     LOF')
    with torch.no_grad():
        for label, kind in [('original', None), ('+ SpecGap, L = D - A', 'unnormalized'),
                            ('+ SpecGap, normalized L', 'normalized')]:
            tr, te_id, te_ood = (np.stack([embed(e, x, kind) for e, x in gs]) for gs in (id_train, id_test, ood_test))
            ssd = auroc(ssd_score(tr, te_id), ssd_score(tr, te_ood))
            lof = auroc(lof_score(tr, te_id), lof_score(tr, te_ood))
            print('%-28s  %.3f   %.3f' % (label, ssd, lof))


if __name__ == '__main__':
    main()
