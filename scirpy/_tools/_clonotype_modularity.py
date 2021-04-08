from functools import lru_cache
from typing import Dict, Optional
import numpy as np
from ..util.graph import _get_igraph_from_adjacency
from ..util._negative_binomial import fit_nbinom
from .._compat import Literal
import scipy.stats
from collections import Counter

# TODO better name!
def clonotype_connectivity(
    adata,
    target_col="clone_id",
    permutation_test: Literal["approx", "exact"] = "approx",
    n_permutations: Optional[int] = None,
    key_added: str = "clonotype_connectivity",
    inplace: bool = True,
):
    if n_permutations is None:
        n_permutations = 200 if permutation_test == "approx" else 10000
    # compute_pvalue = _pvalue_approx if permutation_test == "approx" else _pvalue_exact

    # connectivity = adata.obsp["connectivities"]
    # g = _get_igraph_from_adjacency(connectivity)

    # degree_per_cell = np.sum(connectivity > 0, axis=1).A1
    # clonotype_per_cell = adata.obs[target_col].values
    # clonotypes = np.unique(clonotype_per_cell)
    # clonotype_sizes = set(Counter(clonotype_per_cell).values())

    # connectivity_scores = {}
    # connectivity_pvalues = {}
    # for clonotype in clonotypes:
    #     clonotype_mask = clonotype_per_cell == clonotype
    #     clonotype_subgraph = g.subgraph(np.flatnonzero(clonotype_mask))
    #     connectivity_scores[clonotype] = _connectivity_score(
    #         clonotype_subgraph, degree_per_cell[clonotype_mask]
    #     )
    #     connectivity_pvalues[clonotype] = compute_pvalue(
    #         g, clonotype_subgraph, n_permutations
    #     )

    # if inplace:
    #     adata.obs[key_added] = [connectivity_scores[ct] for ct in clonotype_per_cell]
    #     adata.obs[f"{key_added}_pvalue"] = [
    #         connectivity_pvalues[ct] for ct in clonotype_per_cell
    #     ]
    # else:
    #     return connectivity_scores, connectivity_pvalues


class _ClonotypeModularity:
    def __init__(self, clonotype_per_cell, connectivity, *, random_state=0):
        assert len(clonotype_per_cell) == connectivity.shape[0]
        self.graph = _get_igraph_from_adjacency(connectivity)
        clonotypes = np.unique(clonotype_per_cell)
        edges_per_cell = np.sum(connectivity > 0, axis=1).A1
        # the theoretical maximum of edges per clonotype (limited, as only the
        # k nearest neighbors are computed for each cell)
        self._max_edges = {
            clonotype: np.sum(edges_per_cell[clonotype_per_cell == clonotype])
            for clonotype in clonotypes
        }
        # Dissect the connectivity graph into one subgraph per clonotype.
        self._clonotype_subgraphs = {
            clonotype: self.graph.subgraph(
                np.flatnonzero(clonotype_per_cell == clonotype)
            )
            for clonotype in clonotypes
        }
        # Unique clonotype sizes
        self._clonotype_sizes = np.unique(
            [g.vcount() for g in self._clonotype_subgraphs]
        )
        self.random_state = random_state

    @property
    def n_cells(self):
        """The number of cells in the graph created from the connectivity matrix"""
        return self.graph.vcount()

    def get_scores(self) -> Dict[str, float]:
        """Return the connectivity score for all clonotypes.

        The connectivity score is defined as the number of actual edges
        in the clonotype subgraph, divided by the maximum number of possible
        edges in the subgraph.

        Returns
        -------
        Dictionary clonotype -> score
        """
        score_dict = dict()
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            if subgraph.vcount() == 1:
                score_dict[clonotype] = 0
            else:
                max_edges = (
                    min(
                        # in theory, the maximum numer of edges is the full adjacency matrix
                        # minus the diagonal
                        subgraph.vcount() * (subgraph.vcount() - 1),
                        # however, for larger subnetworks, the number of edges is actually
                        # limited by the number of nearest neighbors. We take the actual
                        # edges per cell from the connectivity matrix.
                        self._max_edges[clonotype],
                    )
                    / 2
                )
                score_dict[clonotype] = subgraph.ecount() / max_edges

        return score_dict

    def _background_edge_count(self, subgraph_size, n) -> np.ndarray:
        """Sample `n` random subgraphs of size `subgraph_size` from `graph` and
        return their edge count.
        """
        np.random.seed(self.random_state)
        if subgraph_size == 1:
            return np.zeros(n)

        return np.fromiter(
            (
                self.graph.subgraph(
                    np.random.choice(self.n_cells, subgraph_size, replace=False)
                ).ecount()
                for _ in range(n)
            ),
            dtype=int,
            count=n,
        )

    def get_approx_pvalues(self, n_permutations: int = 200) -> Dict[str, float]:
        """Compute pvalue for clonotype being more connected than random.

        Approximate permutation test.

        Get a small sample from the background distribution and fit a negative
        binomial distribution. Use the CDF of the fitted distribution to
        derive the pvalue.

        Returns
        -------
        Dictionary clonotype -> pvalue
        """
        distributions_per_size = {}
        for tmp_size in self._clonotype_sizes:
            bg = self._background_edge_count(tmp_size, n_permutations)
            n, p = fit_nbinom(bg)
            distributions_per_size[tmp_size] = scipy.stats.nbinom(n, p)

        pvalue_dict = dict()
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            if subgraph.vcount() <= 2:
                pvalue_dict[clonotype] = 1.0
            else:
                nb_dist = distributions_per_size[subgraph.vcount()]
                pvalue_dict[clonotype] = 1 - nb_dist.cdf(subgraph.ecount() - 1)

        return pvalue_dict

    def get_exact_pvalues(self, n_permutations: int = 10000) -> Dict[str, float]:
        """Compute pvalue for clonotype being more connected than random.

        Exact permutation test, see http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/

        The minimal achievable pvalue is 1/n_permutations, so this test
        only makes sense with large sampling rates.

        Returns
        -------
        Dictionary clonotype -> pvalue
        """
        distributions_per_size = {}
        for tmp_size in self._clonotype_sizes:
            distributions_per_size[tmp_size] = self._background_edge_count(
                tmp_size, n_permutations
            )

        pvalue_dict = dict()
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            p = (
                np.sum(distributions_per_size[subgraph.vcount()] >= subgraph.ecount())
                / n_permutations
            )
            # If the pvalue is 0 return 1 / n_permutations instead (the minimal "resolution")
            if p == 0:
                p = 1 / n_permutations
            pvalue_dict[clonotype] = p

        return pvalue_dict
