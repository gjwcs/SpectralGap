# SpectralGap: Graph-Level Out-of-Distribution Detection via Laplacian Eigenvalue Gaps

Jiawei Gu, Ziyue Qiao, Zechao Li · IJCAI 2025 · [Paper](https://www.ijcai.org/proceedings/2025/584) · [arXiv](https://arxiv.org/abs/2505.15177)

In-distribution (ID) graphs of a dataset share a consistent gap between the two largest eigenvalues of their Laplacian, while out-of-distribution (OOD) graphs often show larger and more varied gaps. **SpecGap** turns this observation into a post-hoc feature adjustment: for each input graph it computes the spectral gap Δλ = λ<sub>n</sub> − λ<sub>n−1</sub> of L = D − A and the eigenvector u<sub>n−1</sub> of λ<sub>n−1</sub>, and subtracts the gap-scaled component X′ = X − Δλ u<sub>n−1</sub>v<sub>n−1</sub><sup>⊤</sup>, with v<sub>n−1</sub> = X<sup>⊤</sup>u<sub>n−1</sub>, from the node features X of a trained GNN. The adjusted features replace the originals in the remaining layers, and an existing graph OOD scoring function (e.g. SSD or LOF) is applied unchanged. SpecGap has no parameters and needs no retraining.

This repository provides a concise implementation of the method. The detailed version with the full experimental code will be released later.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy
- SciPy

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python demo.py
```

The demo builds synthetic ID graphs (rings with chords) and OOD graphs with the same numbers of nodes and edges but grown around hubs. It embeds them with a fixed random two-layer GCN-style map, applies SpecGap after the first layer, and prints the AUROC of an SSD-style Mahalanobis score and of LOF without and with SpecGap. It runs on the CPU in a few seconds and downloads nothing.

## Usage with a trained GNN

Graphs are given as `edge_index` (2 × E, global node indices) or as an adjacency matrix; a mini-batch is the usual disjoint union with a `batch` vector of graph indices. No graph library is required.

```python
import torch
from specgap import SpecGap

sg = SpecGap(kind="unnormalized")            # L = D - A (Eq. 1); kind="normalized" scored best in Tab. 6

@torch.no_grad()
def graph_embeddings(gnn, x, edge_index, batch):
    h = gnn.layers[0](x, edge_index)          # node features after the first layer of the trained GNN
    h = sg(h, edge_index, batch)              # X' = X - dl * u_{n-1} v_{n-1}^T, per graph (Eqs. 3-5)
    for layer in gnn.layers[1:]:              # X' replaces X in the remaining layers (Eq. 6)
        h = layer(h, edge_index)
    return gnn.readout(h, batch)              # graph-level embeddings

# Fit the existing OOD detector (e.g. SSD or LOF) on embeddings of ID training graphs and score the
# test graphs exactly as before; only the embeddings change.
```

For a single graph, `specgap(x, edge_index=edge_index)` from `specgap.py` returns the adjusted node features directly. The gap and the eigenvector depend only on the input graph, so they can be computed once with `spectral_gap(...)` and passed back as `eig=(dl, u)`.

## API

| Function | Description |
|---|---|
| `specgap(x, edge_index=None, adj=None, kind="unnormalized", method="auto", eig=None)` | SpecGap on one graph: returns X′ = X − Δλ u<sub>n−1</sub>v<sub>n−1</sub><sup>⊤</sup> (Eqs. 3–5) |
| `specgap_batch(x, edge_index, batch, **kw)` | the same for a mini-batch of graphs (disjoint union) |
| `SpecGap(kind, method)` | parameter-free `torch.nn.Module`: `SpecGap()(x, edge_index, batch=None)` |
| `spectral_gap(edge_index, num_nodes, adj=None, kind, method)` | Δλ = λ<sub>n</sub> − λ<sub>n−1</sub> and u<sub>n−1</sub> of one graph (Eq. 3) |
| `laplacian(adj, kind)` | `"unnormalized"` L = D − A (Eq. 1), `"normalized"` I − D<sup>−1/2</sup>AD<sup>−1/2</sup>, `"signless"` D + A (Tab. 6) |
| `top2_eigenpairs(L, method)` | `"eigh"`: exact dense solver (small graphs); `"eigsh"`: SciPy Lanczos/ARPACK on sparse L (large graphs); `"lanczos"`: the Lanczos iteration with full reorthogonalisation of Sec. 2.3 (Eqs. 8–19); `"auto"`: `eigh` up to 1000 nodes, else `eigsh` |

Node features have one row per node, so v<sub>n−1</sub> = X<sup>⊤</sup>u<sub>n−1</sub> is a vector over feature channels. With a unit eigenvector, Eq. 5 equals (I − Δλ u<sub>n−1</sub>u<sub>n−1</sub><sup>⊤</sup>)X: only the component of X along u<sub>n−1</sub> changes, scaled by 1 − Δλ. In the paper's ablations, applying SpecGap after the first GNN layer (Tab. 5), the normalized Laplacian (Tab. 6) and two eigenvalues, i.e. the gap itself (Fig. 3), give the best results.

## Citation

```bibtex
@inproceedings{gu2025spectralgap,
  title     = {SpectralGap: Graph-Level Out-of-Distribution Detection via Laplacian Eigenvalue Gaps},
  author    = {Gu, Jiawei and Qiao, Ziyue and Li, Zechao},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25},
  pages     = {5244--5252},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/584}
}
```

## Contact

Jiawei Gu · gjwcs@outlook.com
