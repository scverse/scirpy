from collections.abc import Sequence
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.stats
from mudata import MuData
from scanpy import logging
from statsmodels.stats.multitest import fdrcorrection

from scirpy.util import DataHandler, _is_na, tqdm
from scirpy.util._negative_binomial import fit_nbinom
from scirpy.util.graph import _get_igraph_from_adjacency


@DataHandler.inject_param_docs()
def clonotype_modularity(
    adata: DataHandler.TYPE,
    target_col="clone_id",
    connectivity_key="gex:connectivities",
    permutation_test: Literal["approx", "exact"] = "approx",
    n_permutations: int | None = None,
    key_added: str = "clonotype_modularity",
    inplace: bool = True,
    fdr_correction: bool = True,
    random_state: int = 0,
    airr_mod: str = "airr",
) -> tuple[dict[str, float], dict[str, float]] | None:
    """\
    Identifies clonotypes or clonotype clusters consisting of cells that are
    more transcriptionally related than expected by chance by computing the
    :term:`Clonotype modularity`.

    For each clonotype, we compare the number of edges connecting the cells belonging
    to that clonotype in the transcriptomics neighborhood graph with
    the number of edges expeced by chance in a subgraph of the same size.

    We define the *connectivity score* as the log2 of the ratio of actual to
    expected edges. A pseudocount of 1 is added to cope with small subgraphs with 0
    expected edges. Intuitively, a clonotype modularity of `1` means that there are
    twice as many edges in the neighborhood graph than expected by chance.

    .. math::

        \\text{{connectivity score}} = \\log_2 \\frac{{
                |E|_{{\\text{{actual}}}} + 1
            }}{{
                |E|_{{\\text{{expected}}}} + 1
            }}

    For each unique clonotype size, the expected number of edges is derived by
    randomly sampling `n_permutation` subgraphs from the transcriptomics neighborhood
    graph. This background distribution is also used to calculate p-values for the
    connectivity scores. By choosing `permutation_test="approx"`, a negative
    binomial distribution is fitted to the background distribution and used to
    calculate p-values.

    The `clonotype_modularity` function inspired by CoNGA :cite:`Schattgen2021`,
    however, while CoNGA creates "conga clusters" based on cells that share edges in
    the TCR and transcriptomics neighborhood graph, `clonotype_modularity` uses
    predefined clonotype clusters and checks if within those clusters, the transcriptomics
    neighborhood graph is more connected than expected by chance.

    Parameters
    ----------
    {adata}
    target_col
        Column in `adata.obs` containing the clonotype annotation.
    connectivity_key
        Key in`adata.obsp` containing the transcriptomics neighborhood graph
        connectivity matrix.
    permutation_test
        Whether to perform an approximate or exact permutation test. If the approximate
        permutation test is used, the result of fewer permutations is used to fit
        a negative binomial distribution, from which p-values are derived.
    n_permutations
        Number of permutations used for the permutations test. Defaults to `1000` for
        the approx test, and to `10000` for the exact test. Note that for the exact
        test, the minimum achievable p-values is `1/n`.
    {inplace}
    {key_added}
    fdr_correction
        Whether to adjust the p-values for multiple testing using false-discovery-rate
        (FDR) correction.
    random_state
        random seed for permutation test
    {airr_mod}

    Returns
    -------
    If `inplace` is False, returns two dictionaries mapping the clonotype id onto
    a single modularity score and p-value per clonotype. Otherwise, adds two columns
    to `adata.obs`

       * `adata.obs["{{key_added}}"]`: the modularity scores for each cell
       * `adata.obs["{{key_added}}_pvalue"]` or `adata.obs["{{key_added}}_fdr"]` with the
         raw p-values or false discovery rates, respectively, depending on the value
         of `fdr_correction`.

    and a dictionary to `adata.uns`

       * `adata.uns["{{key_added}}"]`: A dictionary holding the parameters this
         function was called with.
    """
    params = DataHandler(adata, airr_mod)
    if n_permutations is None:
        n_permutations = 1000 if permutation_test == "approx" else 10000

    clonotype_per_cell = params.get_obs(target_col)
    cells_with_valid_clonotype = clonotype_per_cell[~_is_na(clonotype_per_cell.values)].index
    data_subset = params.data[cells_with_valid_clonotype.values, :]
    try:
        connectivities = data_subset.obsp[connectivity_key]
    except KeyError:
        if isinstance(params.data, MuData):
            # try again by getting connectivities from modality
            gex_mod, connectivity_key = connectivity_key.split(":")
            # Since we are now taking the connectivities from the GEX modality,
            # we need to subset valid cells to the intersection with the GEX modality
            cells_with_valid_clonotype = cells_with_valid_clonotype[
                cells_with_valid_clonotype.isin(data_subset.mod[gex_mod].obs_names)
            ]
            data_subset = data_subset[cells_with_valid_clonotype.values, :]
            connectivities = data_subset.mod[gex_mod].obsp[connectivity_key]
        else:
            # for backwards compatibility, try default value without 'gex'
            connectivities = data_subset.obsp[connectivity_key.replace("gex:", "")]

    logging.info("Initalizing clonotype subgraphs...")  # type: ignore

    cm = _ClonotypeModularity(
        clonotype_per_cell[cells_with_valid_clonotype].values,  # type: ignore
        connectivities,  # type: ignore
        random_state=random_state,
    )
    start = logging.info("Computing background distributions...")  # type: ignore
    cm.estimate_edges_background_distribution(n_permutations=n_permutations)
    logging.debug("Finished computing background distributions", time=start)  # type:ignore

    logging.debug("Computing modularity scores")  # type: ignore
    modularity_scores = cm.get_scores()
    logging.debug("Computing modularity pvalues")  # type: ignore
    modularity_pvalues = cm.get_approx_pvalues() if permutation_test == "approx" else cm.get_exact_pvalues()

    if fdr_correction:
        modularity_pvalues = dict(
            zip(
                modularity_pvalues.keys(),
                fdrcorrection(list(modularity_pvalues.values()))[1],
                strict=False,
            )
        )

    if inplace:
        params.set_obs(key_added, [modularity_scores.get(ct, np.nan) for ct in clonotype_per_cell])
        # remove the entries from previous run, should they exist
        # results can be inconsisten otherwise (old dangling "fdr" values when only
        # pvalues are calculated)
        for suffix in ["fdr", "pvalue"]:
            for d in ["mdata", "adata"]:
                try:
                    del getattr(params, d).obs[f"{key_added}_{suffix}"]
                except (AttributeError, KeyError):
                    pass

        suffix = "fdr" if fdr_correction else "pvalue"
        params.set_obs(
            f"{key_added}_{suffix}",
            [modularity_pvalues.get(ct, np.nan) for ct in clonotype_per_cell],
        )
        params.data.uns[key_added] = {
            "target_col": target_col,
            "permutation_test": permutation_test,
            "n_permutations": n_permutations,
            "fdr_correction": fdr_correction,
        }
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
        clonotypes = np.unique(clonotype_per_cell)
        # Dissect the connectivity graph into one subgraph per clonotype.
        self._clonotype_subgraphs = {
            clonotype: self.graph.subgraph(np.flatnonzero(clonotype_per_cell == clonotype))
            for clonotype in tqdm(clonotypes)
        }
        # Unique clonotype sizes
        self._clonotype_sizes = np.unique([g.vcount() for g in self._clonotype_subgraphs.values()])
        self.random_state = random_state
        self.edges_background_distribution = None

    @property
    def n_cells(self):
        """The number of cells in the graph created from the connectivity matrix"""
        return self.graph.vcount()

    def _get_background_distribution(self, i):
        """Helper function to get random background distribution for a single iteration"""
        res = []
        np.random.seed(self.random_state + i)
        for subgraph_size in self._clonotype_sizes:
            subgraph = self.graph.subgraph(np.random.choice(self.n_cells, subgraph_size, replace=False))
            res.append(subgraph.ecount())
        return np.array(res)

    def estimate_edges_background_distribution(self, n_permutations=1000):
        """Compute the distribution of #edges per subgraph under a random model.

        The calculation needs to be performed for each subgraph with a given size.
        """
        self.n_permutations = n_permutations

        # TODO parallelize. Need to do this more cleverly, currently the memory
        # consumption is too high as everything is copied to each worker.
        # This is also the bottleneck of the process, s.t. naively parallelizing
        # does not lead to a speed gain.

        background_distribution = np.vstack(
            list(
                map(
                    self._get_background_distribution,
                    tqdm(range(n_permutations)),
                )
            )
        )

        # convert n_iter x clonotype_sizes array to dictionary
        self.edges_background_distribution = {
            s: background_distribution[:, i] for i, s in enumerate(self._clonotype_sizes)
        }

    def get_scores(self) -> dict[str, float]:
        """Return the connectivity score for all clonotypes.

        The connectivity score is ispired by network modularity and
        defined as the ratio of actual edges in a subgraph to the expected number
        of edges according to the background distribution.

        The classical definition of network modularity does not work as it always
        depends on a partitioning. Using clonotype vs. rest of network as a partitioning
        does not make sense since the value would also depend on the connectivity
        of the rest of the network rather than the clonotype only.

        See Also
        --------
         * https://stats.stackexchange.com/questions/508577/given-a-graph-test-if-some-vertices-are-more-connected-than-the-background
         * https://en.wikipedia.org/wiki/Modularity_(networks)

        Returns
        -------
        Dictionary clonotype -> score
        """
        if self.edges_background_distribution is None:
            raise ValueError("Need to run `estimate_background_distribution` first. ")

        score_dict = {}
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            score_dict[clonotype] = np.log2(
                (subgraph.ecount() + 1) / (np.mean(self.edges_background_distribution[subgraph.vcount()]) + 1)
            )

        return score_dict

    def get_approx_pvalues(self) -> dict[str, float]:
        """Compute pvalue for clonotype being more connected than random.

        Approximate permutation test.

        Get a small sample from the background distribution and fit a negative
        binomial distribution. Use the CDF of the fitted distribution to
        derive the pvalue.

        Returns
        -------
        Dictionary clonotype -> pvalue
        """
        if self.edges_background_distribution is None:
            raise ValueError("Need to run `estimate_background_distribution` first. ")

        distributions_per_size = {}
        for tmp_size in self._clonotype_sizes:
            bg = self.edges_background_distribution[tmp_size]
            n, p = fit_nbinom(np.array(bg))
            distributions_per_size[tmp_size] = scipy.stats.nbinom(n, p)

        pvalue_dict = {}
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            if subgraph.vcount() <= 2:
                pvalue_dict[clonotype] = 1.0
            else:
                nb_dist = distributions_per_size[subgraph.vcount()]
                # restrict pvalues to float precision
                pvalue_dict[clonotype] = max(
                    1 - nb_dist.cdf(subgraph.ecount() - 1),
                    np.finfo(np.float32).tiny,  # type: ignore
                )

        return pvalue_dict

    def get_exact_pvalues(self) -> dict[str, float]:
        """Compute pvalue for clonotype being more connected than random.

        Exact permutation test, see http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/

        The minimal achievable pvalue is 1/n_permutations, so this test
        only makes sense with large sampling rates.

        Returns
        -------
        Dictionary clonotype -> pvalue
        """
        if self.edges_background_distribution is None:
            raise ValueError("Need to run `estimate_background_distribution` first. ")

        pvalue_dict = {}
        for clonotype, subgraph in self._clonotype_subgraphs.items():
            p = (
                np.sum(np.array(self.edges_background_distribution[subgraph.vcount()]) >= subgraph.ecount())
                / self.n_permutations
            )
            # If the pvalue is 0 return 1 / n_permutations instead (the minimal "resolution")
            if p == 0:
                p = 1 / self.n_permutations
            pvalue_dict[clonotype] = p

        return pvalue_dict
