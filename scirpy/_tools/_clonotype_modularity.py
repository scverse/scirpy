from typing import Dict, Optional, Sequence, Tuple
import igraph
import numpy as np
from ..util.graph import _get_igraph_from_adjacency
from ..util._negative_binomial import fit_nbinom
from .._compat import Literal
import scipy.stats
import scipy.sparse
from collections import Counter
from statsmodels.stats.multitest import fdrcorrection
from ..util import tqdm


def clonotype_modularity(
    adata,
    target_col="clone_id",
    connectivity_key="connectivities",
    permutation_test: Literal["approx", "exact"] = "approx",
    n_permutations: Optional[int] = None,
    key_added: str = "clonotype_modularity",
    inplace: bool = True,
    fdr_correction: bool = True,
    random_state: int = 0,
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Identifies clonotypes or clonotype clusters that consist of cells that are
    more transcriptionally related than expected by chance.

    TODO explain graph approach

    This is loosely inspired by CoNGA [TODO cite], however, while CoNGA creates
    "conga clusters" based on cells that share edges in the TCR and transcriptomics
    neighborhood graph, `clonotype_modularity` uses given clonotype clusters
    and checks if the transcriptomics neighborhood graph is more connected than expected
    by chance.

    Parameters
    ----------
    adata
        annotated data matrix
    target_col
        Column in `adata.obs` containing the clonotype annotation.
    connectivity_key
        Key in`adata.obsp` containing the transcriptomics neighborhood graph
        connectivity matrix.
    permutation_test
        Whether to perform an approximate or exact permutation test. If the approximate
        permutation test is used, the result of fewer permutations is used to fit
        a negative binomial distribution, from which p-values are derived. TODO
    n_permutations
        Number of permutations used for the permutations test. Defaults to `200` for
        the approx test, and to `10000` for the exact test. Note that for the exact
        test, the minimum p-values achievable is 1/n, therefore, a lot of permutations
        are required.
    key_added
        Key under which the result will be stored in `adata.obs` if inplace is `True`.
    fdr_correction
        Whether to adjust the p-values for multiple testing using false-discovery-rate
        (FDR) correction.
    random_state
        random seed for permutation test

    Returns
    -------
    If `inplace` is False, returns two dictionaries mapping the clonotype id onto
    a single modularity score and p-value per clonotype. Otherwise, adds two columns
    to `adata.obs`:

       * "{key_added}": the modularity scores for each cell
       * "{key_added}_pvalue" or "{key_added}_fdr" with the raw p-values or false
         discovery rates, respectively, depending on the value of `fdr_correction`.
    """
    if n_permutations is None:
        n_permutations = 200 if permutation_test == "approx" else 10000

    clonotype_per_cell = adata.obs[target_col]

    cm = _ClonotypeModularity(
        clonotype_per_cell, adata.obsp[connectivity_key], random_state=random_state
    )
    cm.estimate_edges_background_distribution(n_permutations=n_permutations)

    modularity_scores = cm.get_scores()
    modularity_pvalues = (
        cm.get_approx_pvalues(n_permutations)
        if permutation_test == "approx"
        else cm.get_exact_pvalues(n_permutations)
    )

    if fdr_correction:
        modularity_pvalues = {
            k: v
            for k, v in zip(
                modularity_pvalues.keys(),
                fdrcorrection(list(modularity_pvalues.values()))[1],
            )
        }

    if inplace:
        adata.obs[key_added] = [modularity_scores[ct] for ct in clonotype_per_cell]
        suffix = "fdr" if not fdr_correction else "pvalue"
        adata.obs[f"{key_added}_{suffix}"] = [
            modularity_pvalues[ct] for ct in clonotype_per_cell
        ]
    else:
        return modularity_scores, modularity_pvalues


class _ClonotypeModularity:
    def __init__(
        self,
        clonotype_per_cell: Sequence[str],
        connectivity: scipy.sparse.csr_matrix,
        *,
        random_state: int = 0,
    ):
        """Class to compute the clonotype modularity.

        Parameters
        ----------
        clonotype_per_cell
            Array containing the clonotype for each cell (usually a column from
            adata.obs)
        connectivity
            pairwise connectivity matrix of the transcriptomics neighborhood graph.
            Must be aligned with `clonotype_per_cell`
        random_state
            Random seed for permutation tests
        """
        assert len(clonotype_per_cell) == connectivity.shape[0]
        self.graph = _get_igraph_from_adjacency(connectivity)
        self.clonotype_per_cell = clonotype_per_cell
        self.random_state = random_state
        self.edges_background_distribution = None

    @property
    def n_cells(self):
        """The number of cells in the graph created from the connectivity matrix"""
        return self.graph.vcount()

    def estimate_edges_background_distribution(self, n_permutations=200):
        """Get the distribution of #edges per clonotype under a random model"""
        np.random.seed(self.random_state)
        clonotype_sizes = Counter(self.clonotype_per_cell)
        self.edges_background_distribution = {ct: [] for ct in clonotype_sizes}
        for i in tqdm(range(n_permutations)):
            # Potentially `Degree_Sequence` and `rewire are` options here to generate
            # random graphs with the same degree distribution. Sequence_Game creates
            # a new graph with a given degree distribution.  When using rewire, the
            # number of rewiring trials must be large enough that the original graph
            # structure becomes irrelevant. 10*|E| seems to be commonly considered "enough"
            #
            # My tests have shown that for the graph structure we expect here,
            # `Degree_Sequence` is about 10 times faster than rewiring with 10*|E|
            # trials.
            #
            # See also the discussion here:
            # https://igraph-help.nongnu.narkive.com/LJfvfzKz/rewire-vs-degree-sequence-game
            g = igraph.Graph.Degree_Sequence(self.graph.degree(), method="no_multiple")
            for subgraph_size in clonotype_sizes:
                subgraph = g.subgraph(
                    np.random.choice(self.n_cells, subgraph_size, replace=False)
                )
                self.edges_background_distribution[subgraph_size].append(
                    subgraph.ecount()
                )

    def get_scores(self) -> Dict[str, float]:
        """Return the connectivity score for all clonotypes.

        The connectivity score is defined as the number of actual edges
        in the clonotype subgraph, divided by the maximum number of possible
        edges in the subgraph.

        Returns
        -------
        Dictionary clonotype -> score
        """
        # TODO consider using the difference compared to the expected number
        # of edges instead. (as implemented here as an inefficient proof of concept).
        # Potentially need to clean up the old code (using number of possible edges)
        # instead.
        # --> https://stats.stackexchange.com/questions/508577/given-a-graph-test-if-some-vertices-are-more-connected-than-the-background
        # --> https://en.wikipedia.org/wiki/Modularity_(networks)

        if self.edges_background_distribution is None:
            raise ValueError("Need to run `estimate_background_distribution` first. ")

        score_dict = {}
        for clonotype in np.unique(self.clonotype_per_cell):
            score_dict[clonotype] = (
                self.graph.subgraph(
                    np.flatnonzero(self.clonotype_per_cell == clonotype)
                ).ecount()
                + 1
            ) / (np.mean(self.edges_background_distribution[clonotype]) + 1)

        # distributions_per_size = {}
        # for tmp_size in self._clonotype_sizes:
        #     bg = self._background_edge_count(tmp_size, 200)
        #     distributions_per_size[tmp_size] = np.mean(bg)

        # score_dict = dict()
        # for clonotype, subgraph in self._clonotype_subgraphs.items():
        #     if subgraph.vcount() == 1:
        #         score_dict[clonotype] = 0
        #     else:
        #         score_dict[clonotype] = (
        #             subgraph.ecount() - distributions_per_size[subgraph.vcount()]
        #         ) / subgraph.vcount()
        #         continue
        #         max_edges = (
        #             min(
        #                 # in theory, the maximum numer of edges is the full adjacency matrix
        #                 # minus the diagonal
        #                 subgraph.vcount() * (subgraph.vcount() - 1),
        #                 # however, for larger subnetworks, the number of edges is actually
        #                 # limited by the number of nearest neighbors. We take the actual
        #                 # edges per cell from the connectivity matrix.
        #                 self._max_edges[clonotype],
        #             )
        #             / 2
        #         )
        #         score_dict[clonotype] = subgraph.ecount() / max_edges

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
