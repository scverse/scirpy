from functools import lru_cache
from typing import Optional
import numpy as np
from ..util.graph import _get_igraph_from_adjacency
from ..util._negative_binomial import fit_nbinom
from .._compat import Literal
import scipy.stats

# TODO better name!
def clonotype_connectivity(
    adata,
    target_col="clone_id",
    permutation_test: Literal["approx", "exact"] = "approx",
    n_permutations: Optional[int] = None,
    inplace: bool = True,
):
    if n_permutations is None:
        n_permutations = 200 if permutation_test == "approx" else 10000

    connectivity = adata.obsp["connectivities"]
    g = _get_igraph_from_adjacency(connectivity)

    degree_per_cell = np.sum(connectivity > 0, axis=1).A1
    clonotype_per_cell = adata.obs[target_col].values
    clonotypes = np.unique(clonotype_per_cell)

    clonotype_subgraphs = {
        ct: g.subgraph(np.flatnonzero(clonotype_per_cell == ct)) for ct in clonotypes
    }

    clonotype_sizes = np.unique(g.vcount() for g in clonotype_subgraphs.values())

    max_connectivity_per_subgraph = {
        ct: min(
            # in theory, the maximum numer of edges is the full adjacency matrix
            # minus the diagonal
            graph.vcount() * (graph.vcount() - 1),
            # however, for larger subnetworks, the number of edges is actually
            # limited by the number of nearest neighbors. We take the actual
            # edges per cell from the connectivity matrix.
            np.sum(degree_per_cell[clonotype_per_cell == ct]),
        )
        / 2
        for ct, graph in clonotype_subgraphs.items()
    }

    score_per_clonotype = {
        ct: clonotype_subgraphs[ct].ecount() / max_connectivity_per_subgraph[ct]
        for ct in clonotypes
    }

    connectivity_scores = np.array()
    connectivity_pavluese = np.array()

    if inplace:
        # store in adata.obs
        pass
    else:
        return


@lru_cache(None)
def _background_edge_count(graph, subgraph_size, n, seed=0) -> np.ndarray:
    """Sample `n` random subgraphs of size `subgraph_size` from `graph` and
    return their edge count.

    We use lru_cache, as it is enough to compute this once per clonotype size.
    """
    np.random.seed(seed)
    if subgraph_size == 1:
        return np.zeros(n)

    return np.fromiter(
        graph.subgraph(
            np.random.choice(graph.vcount(), subgraph_size, replace=False)
        ).ecount(),
        dtype=int,
        count=n,
    )


def _connectivity_score(clonotype_subgraph) -> float:
    pass


def _pvalue_approx(graph, clonotype_subgraph, n) -> float:
    """Approximate permutation test.

    Get a small sample from the background distribution and fit a negative
    binomial distribution. Use the CDF of the fitted distribution to
    derive the pvalue"""
    if clonotype_subgraph.vcount() <= 2:
        return 1.0
    bg = _background_edge_count(graph, clonotype_subgraph.vcount(), n=n)
    n, p = fit_nbinom(bg)
    return 1 - scipy.stats.nbinom(n, p).cdf(clonotype_subgraph.ecount() - 1)


def _pvalue_exact(graph, clonotype_subgraph, n) -> float:
    """Exact permutation test

    See permutation test see http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/
    """
    p = (
        np.sum(
            _background_edge_count(graph, clonotype_subgraph.vcount(), n)
            >= clonotype_subgraph.ecount()
        )
        / n
    )

    # If the pvalue is 0 return 1 / n_permutations instead (the minimal "resolution")
    if p == 0:
        return 1 / n
    else:
        return p
