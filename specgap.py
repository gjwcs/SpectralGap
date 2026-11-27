"""SpecGap: graph-level out-of-distribution detection via Laplacian eigenvalue gaps.

Concise implementation of

    Jiawei Gu, Ziyue Qiao, Zechao Li. SpectralGap: Graph-Level Out-of-Distribution Detection via
    Laplacian Eigenvalue Gaps. IJCAI 2025.

Equation numbers refer to the paper. For an undirected graph G = (V, E) with n nodes and node features
X in R^{n x d} (one row per node) taken after a layer of a trained GNN:

    A_ij = 1 if (i, j) in E, else 0                               (2)  adjacency
    L = D - A,  eigenvalues 0 = lambda_1 <= ... <= lambda_n       (1)  graph Laplacian
    dl = lambda_n - lambda_{n-1}                                  (3)  spectral gap
    v_{n-1} = X^T u_{n-1}                                         (4)  u_{n-1}: unit eigenvector of lambda_{n-1}
    X' = X - dl * u_{n-1} v_{n-1}^T                               (5)  gap-scaled correction
    h_{l+1} = f_l(X', A)                                          (6)  X' replaces X in the remaining layers

With ||u_{n-1}|| = 1, Eq. (5) equals (I - dl * u_{n-1} u_{n-1}^T) X: the component of the features along
u_{n-1} is scaled by (1 - dl) and every other direction is kept. SpecGap is post-hoc and parameter-free;
the trained GNN and the graph OOD scoring function (e.g. SSD or LOF) are used unchanged.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

__all__ = ['adjacency', 'laplacian', 'lanczos_top2', 'top2_eigenpairs', 'spectral_gap', 'specgap',
           'specgap_batch', 'SpecGap']

KINDS = ('unnormalized', 'normalized', 'signless')


# ---------------------------------------------------------------------------------------------- graph matrices
def adjacency(edge_index, num_nodes, sparse=False):
    """0/1 adjacency matrix A of an undirected graph (Eq. 2) from an edge list of shape (2, |E|).

    Edges are symmetrised, duplicate edges merged and self-loops removed. Returns a dense float64 torch
    tensor, or a SciPy CSR matrix if ``sparse=True`` (large graphs).
    """
    ei = torch.as_tensor(edge_index, dtype=torch.long).reshape(2, -1).cpu().numpy()
    keep = ei[0] != ei[1]
    row = np.concatenate([ei[0][keep], ei[1][keep]])
    col = np.concatenate([ei[1][keep], ei[0][keep]])
    A = sp.coo_matrix((np.ones(row.size), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0                                    # duplicates were summed by tocsr(); make them 1 again
    return A if sparse else torch.from_numpy(A.toarray())


def laplacian(adj, kind='unnormalized'):
    """Laplacian of a symmetric adjacency matrix (dense torch tensor or SciPy sparse matrix).

    kind='unnormalized'   L = D - A                          (Eq. 1, the definition used by the method)
    kind='normalized'     L = I - D^{-1/2} A D^{-1/2}        (best variant in the ablation, Tab. 6)
    kind='signless'       L = D + A                          (ablation, Tab. 6)

    Isolated nodes get an all-zero row and column in the normalized Laplacian.
    """
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}, got {kind!r}')
    if sp.issparse(adj):
        A = sp.csr_matrix(adj, dtype=np.float64)
        deg = np.asarray(A.sum(axis=1)).ravel()
        if kind == 'unnormalized':
            return (sp.diags(deg) - A).tocsr()
        if kind == 'signless':
            return (sp.diags(deg) + A).tocsr()
        inv = np.zeros_like(deg)
        inv[deg > 0] = deg[deg > 0] ** -0.5
        Dm = sp.diags(inv)
        return (sp.diags((deg > 0).astype(np.float64)) - Dm @ A @ Dm).tocsr()
    A = torch.as_tensor(adj, dtype=torch.float64)
    deg = A.sum(dim=1)
    if kind == 'unnormalized':
        return torch.diag(deg) - A
    if kind == 'signless':
        return torch.diag(deg) + A
    inv = torch.where(deg > 0, deg.clamp(min=1e-12).rsqrt(), torch.zeros_like(deg))
    return torch.diag((deg > 0).to(A.dtype)) - inv[:, None] * A * inv[None, :]


# ---------------------------------------------------------------------------------------------- eigensolvers
def lanczos_top2(L, tol=1e-10, max_iter=None, seed=0):
    """Two largest Ritz pairs of a symmetric matrix by the Lanczos iteration of Sec. 2.3 (Eqs. 8-19).

    Builds an orthonormal basis of the Krylov subspace K_k(L, q_1) (Eq. 8) from a random unit vector
    q_1 = r / ||r|| (Eq. 16) with the three-term recurrence (Eqs. 9-13), matrix-vector products on the
    (sparse) Laplacian, L q = D q - A q (Eq. 17), and full reorthogonalisation w <- w - Q_j (Q_j^T w)
    (Eq. 18). The Ritz values are the eigenvalues of the tridiagonal T_k (Eq. 14); the iteration stops when
    both top Ritz values change by less than ``tol`` (Eq. 19). Cost O(k nnz(L)) for the products.

    Returns (lambda_n, lambda_{n-1}, u_{n-1}) with u_{n-1} = Q_k y the unit Ritz vector of lambda_{n-1}
    (NumPy). Like any single-vector Krylov method it returns one copy of a repeated eigenvalue; use
    ``top2_eigenpairs(..., method='eigh')`` when exact multiplicities matter.
    """
    M = L if sp.issparse(L) else np.asarray(L, dtype=np.float64)
    n = M.shape[0]
    if n < 2:
        lam = float(M[0, 0]) if n == 1 else 0.0
        return lam, lam, np.zeros(n)
    k_max = n if max_iter is None else max(2, min(n, int(max_iter)))
    r = np.random.default_rng(seed).standard_normal(n)
    Q = np.zeros((n, k_max + 1))
    Q[:, 0] = r / np.linalg.norm(r)                                   # q_1 = r / ||r||            (Eq. 16)
    alpha, beta = [], []
    theta = Y = prev = None
    for j in range(k_max):
        w = np.asarray(M @ Q[:, j], dtype=np.float64).ravel()         # L q_j = D q_j - A q_j      (Eq. 17)
        if j > 0:
            w = w - beta[j - 1] * Q[:, j - 1]                         # w = L q_j - b_{j-1} q_{j-1} (Eq. 9)
        alpha.append(float(w @ Q[:, j]))                              # a_j = w^T q_j              (Eq. 10)
        w = w - alpha[j] * Q[:, j]                                    # w = w - a_j q_j            (Eq. 11)
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)                   # full reorthogonalisation   (Eq. 18)
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)     # tridiagonal T_k            (Eq. 14)
        theta, Y = np.linalg.eigh(T)                                  # Ritz values, ascending
        if j >= 1:
            cur = theta[-2:]
            if prev is not None and np.all(np.abs(cur - prev) < tol):  # stopping rule             (Eq. 19)
                break
            prev = cur
        b = float(np.linalg.norm(w))                                  # b_j = ||w||_2              (Eq. 12)
        if b < 1e-12:                                                 # invariant subspace: T_k is exact
            break
        beta.append(b)
        Q[:, j + 1] = w / b                                           # q_{j+1} = w / b_j          (Eq. 13)
    if len(alpha) < 2:                                                # start vector was an eigenvector
        dense = M.toarray() if sp.issparse(M) else M
        w_all, V = np.linalg.eigh(dense)
        return float(w_all[-1]), float(w_all[-2]), V[:, -2]
    u = Q[:, :len(alpha)] @ Y[:, -2]
    return float(theta[-1]), float(theta[-2]), u / np.linalg.norm(u)


def top2_eigenpairs(L, method='auto', dense_max_nodes=1000, tol=1e-10, seed=0):
    """Two largest eigenvalues lambda_n >= lambda_{n-1} of a symmetric Laplacian and the unit eigenvector u_{n-1}.

    method='eigh'      exact dense eigendecomposition with torch.linalg.eigh (small graphs, e.g. molecules)
    method='eigsh'     SciPy ARPACK ``eigsh(k=2, which='LA')``, an implicitly restarted Lanczos method
                       (large sparse graphs)
    method='lanczos'   the Lanczos iteration with full reorthogonalisation of Sec. 2.3 (:func:`lanczos_top2`)
    method='auto'      'eigh' if n <= dense_max_nodes, else 'eigsh'

    ``L`` is a dense torch tensor / NumPy array or a SciPy sparse matrix. Returns (lambda_n, lambda_{n-1},
    u_{n-1}) as two floats and a float64 torch tensor of length n. The sign of an eigenvector is arbitrary;
    Eq. (5) does not depend on it. If lambda_{n-1} is repeated, the eigenvector returned by the solver is used.
    """
    n = L.shape[0]
    if method == 'auto':
        method = 'eigh' if n <= dense_max_nodes else 'eigsh'
    if method not in ('eigh', 'eigsh', 'lanczos'):
        raise ValueError(f"method must be 'auto', 'eigh', 'eigsh' or 'lanczos', got {method!r}")
    if n < 2:                                          # a single node has no second eigenvalue: no correction
        lam = float(L[0, 0]) if n == 1 else 0.0
        return lam, lam, torch.zeros(n, dtype=torch.float64)
    if method == 'eigsh' and n <= 3:                   # ARPACK needs k < n; tiny graphs are solved exactly
        method = 'eigh'
    if method == 'eigh':
        dense = torch.as_tensor(L.toarray() if sp.issparse(L) else L, dtype=torch.float64).cpu()
        w, V = torch.linalg.eigh(dense)                # ascending eigenvalues
        return float(w[-1]), float(w[-2]), V[:, -2].contiguous()
    Ls = sp.csr_matrix(L, dtype=np.float64) if sp.issparse(L) else sp.csr_matrix(np.asarray(L, dtype=np.float64))
    if method == 'eigsh':
        v0 = np.random.default_rng(seed).standard_normal(n)
        try:
            w, V = spla.eigsh(Ls, k=2, which='LA', v0=v0)
            order = np.argsort(w)
            return float(w[order[1]]), float(w[order[0]]), torch.from_numpy(np.ascontiguousarray(V[:, order[0]]))
        except spla.ArpackNoConvergence:
            pass                                       # fall back to the paper's Lanczos iteration
    lam_n, lam_n1, u = lanczos_top2(Ls, tol=tol, seed=seed)
    return lam_n, lam_n1, torch.from_numpy(np.ascontiguousarray(u))


# ---------------------------------------------------------------------------------------------- SpecGap
def spectral_gap(edge_index=None, num_nodes=None, adj=None, kind='unnormalized', method='auto',
                 dense_max_nodes=1000, **kw):
    """Spectral gap dl = lambda_n - lambda_{n-1} (Eq. 3) and eigenvector u_{n-1} of one graph's Laplacian.

    Give the graph as ``edge_index`` (2, |E|) with ``num_nodes``, or as an adjacency matrix ``adj``
    (dense or SciPy sparse; every nonzero off-diagonal entry is an edge, Eq. 2). The gap depends only on the
    input graph, so it can be computed once per graph and passed to :func:`specgap` as ``eig``.
    Returns (dl, u_{n-1}) as a float and a float64 torch tensor.
    """
    if adj is None:
        if edge_index is None or num_nodes is None:
            raise ValueError('give edge_index and num_nodes, or adj')
        n = int(num_nodes)
        sparse = method in ('eigsh', 'lanczos') or (method == 'auto' and n > dense_max_nodes)
        A = adjacency(edge_index, n, sparse=sparse)
    elif sp.issparse(adj):
        A = sp.csr_matrix(adj, dtype=np.float64)
        A = sp.csr_matrix(abs(A) + abs(A.T) > 0, dtype=np.float64)
        A = (A - sp.diags(A.diagonal())).tocsr()
        A.eliminate_zeros()
    else:
        A = torch.as_tensor(adj, dtype=torch.float64).cpu()
        A = ((A != 0) | (A.T != 0)).to(torch.float64)
        A.fill_diagonal_(0.0)
    lam_n, lam_n1, u = top2_eigenpairs(laplacian(A, kind), method=method, dense_max_nodes=dense_max_nodes, **kw)
    return lam_n - lam_n1, u


def specgap(x, edge_index=None, adj=None, kind='unnormalized', method='auto', eig=None, return_gap=False, **kw):
    """SpecGap on the node features of one graph (Eqs. 3-5).

    x            (n, d) node features after a layer of a trained GNN, one row per node.
    edge_index   (2, |E|) edges of the input graph, or ``adj``: its n x n adjacency matrix.
    kind         Laplacian: 'unnormalized' L = D - A (Eq. 1, default), 'normalized' or 'signless' (Tab. 6).
    method       eigensolver, see :func:`top2_eigenpairs`.
    eig          optional precomputed (dl, u_{n-1}) from :func:`spectral_gap`, e.g. cached per graph.

    Returns X' = X - dl * u_{n-1} v_{n-1}^T with v_{n-1} = X^T u_{n-1}, and dl as well if ``return_gap``.
    The result is linear in ``x``, so gradients flow through it if needed.
    """
    if eig is None:
        with torch.no_grad():
            eig = spectral_gap(edge_index=edge_index, num_nodes=x.shape[0], adj=adj, kind=kind, method=method, **kw)
    dl, u = eig
    u = torch.as_tensor(u).to(device=x.device, dtype=x.dtype)
    v = x.transpose(0, 1) @ u                          # v_{n-1} = X^T u_{n-1}               (Eq. 4)
    out = x - float(dl) * torch.outer(u, v)            # X' = X - dl * u_{n-1} v_{n-1}^T     (Eq. 5)
    return (out, float(dl)) if return_gap else out


def specgap_batch(x, edge_index, batch=None, **kw):
    """SpecGap for a mini-batch of graphs stored as one disjoint union (the usual GNN batching).

    x            (N, d) node features of all graphs in the batch.
    edge_index   (2, E) edges with global node indices.
    batch        (N,) graph index of every node; ``None`` means a single graph.
    Every graph is corrected with its own spectral gap and eigenvector; ``kw`` goes to :func:`specgap`.
    """
    if batch is None:
        return specgap(x, edge_index=edge_index, **kw)
    batch = torch.as_tensor(batch, dtype=torch.long, device=x.device)
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
    local = torch.zeros_like(batch)
    out = torch.empty_like(x)
    for g in torch.unique(batch):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        local[idx] = torch.arange(idx.numel(), device=x.device)
        mask = batch[ei[0]] == g                       # edges of graph g (both end points lie in g)
        out[idx] = specgap(x[idx], edge_index=local[ei[:, mask]], **kw)
    return out


class SpecGap(torch.nn.Module):
    """Parameter-free SpecGap layer for a trained GNN (Eq. 6). Insert it after a GNN layer:

        h = gnn.layers[0](x, edge_index)              # node features of the first layer
        h = SpecGap()(h, edge_index, batch)           # X' = X - dl * u_{n-1} v_{n-1}^T per graph
        ...                                           # remaining layers, readout and OOD scorer unchanged

    In the paper's ablations, the first layer (Tab. 5) and the normalized Laplacian (Tab. 6) work best.
    """

    def __init__(self, kind='unnormalized', method='auto', dense_max_nodes=1000):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {kind!r}')
        self.kind, self.method, self.dense_max_nodes = kind, method, dense_max_nodes

    def forward(self, x, edge_index, batch=None):
        return specgap_batch(x, edge_index, batch, kind=self.kind, method=self.method,
                             dense_max_nodes=self.dense_max_nodes)

    def extra_repr(self):
        return f'kind={self.kind!r}, method={self.method!r}'
