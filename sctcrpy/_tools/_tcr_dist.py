import parasail
from multiprocessing import Pool
import itertools
from anndata import AnnData
from typing import Union, Collection, List
from .._compat import Literal
import pandas as pd
import numpy as np
from scanpy import logging
from sklearn.metrics import pairwise_distances
import numpy.testing as npt
from .._util import _is_na, _is_symmetric
import abc
import textwrap
from io import StringIO
import umap
from scipy.sparse import coo_matrix
from Levenshtein import distance as levenshtein_dist
import scipy.spatial
import igraph as ig


def _get_sparse_matrix_from_indices_distances_umap(
    knn_indices, knn_dists, n_obs, n_neighbors
):
    """This is from scanpy.neighbors [Wolf18]_."""
    rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

    for i in range(knn_indices.shape[0]):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_obs,
    n_neighbors,
    *,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from scanpy.neighbors [Wolf18]_ which again has taken it 
    from umap.fuzzy_simplicial_set [McInnes18]_.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """
    from umap.umap_ import fuzzy_simplicial_set

    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    distances = _get_sparse_matrix_from_indices_distances_umap(
        knn_indices, knn_dists, n_obs, n_neighbors
    )

    return distances, connectivities.tocsr()


def _dist_to_connectivities(
    dist_mat: np.array, n_neighbors: int, *, random_state: int = 0
):
    """Convert a distance matrix into a sparse, nearest-neighbor distance 
    matrix and a sparse adjacencey matrix using umap.nearest_neighbors 
    and a fuzzy-simlicital-set embedding"""
    knn_indices, knn_dists, forest = umap.umap_.nearest_neighbors(
        dist_mat,
        n_neighbors=n_neighbors,
        metric="precomputed",
        metric_kwds=dict(),
        angular=False,
        random_state=random_state,
    )

    dist, connectivities = _compute_connectivities_umap(
        knn_indices, knn_dists, n_obs=dist_mat.shape[0], n_neighbors=n_neighbors
    )

    return dist, connectivities


class _DistanceCalculator(abc.ABC):
    def __init__(self, n_jobs: Union[int, None] = None):
        """
        Parameters
        ----------
        n_jobs
            Number of jobs to use for the pairwise distance calculation. 
            If None, use all jobs. 
        """
        self.n_jobs = n_jobs

    @abc.abstractmethod
    def calc_dist_mat(self, seqs: np.ndarray) -> np.ndarray:
        """Calculate a symmetric, pairwise distance matrix of all sequences in `seq`.
        Distances are non-negative values"""
        pass


class _IdentityDistanceCalculator(_DistanceCalculator):
    """Calculate the distance between TCR based on the identity 
    of sequences. I.e. 0 = sequence identical, 1 = sequences not identical
    """

    def calc_dist_mat(self, seqs: np.ndarray) -> np.ndarray:
        return 1 - np.identity(len(seqs))


class _LevenshteinDistanceCalculator(_DistanceCalculator):
    """Calculates the Levenshtein (i.e. edit-distance) between sequences. """

    def calc_dist_mat(self, seqs: np.ndarray) -> np.ndarray:
        dist = scipy.spatial.distance.pdist(
            seqs.reshape(-1, 1), metric=lambda x, y: levenshtein_dist(x[0], y[0])
        )
        return scipy.spatial.distance.squareform(dist)


class _KideraDistanceCalculator(_DistanceCalculator):
    KIDERA_FACTORS = textwrap.dedent(
        """
        A -1.56 -1.67 -0.97 -0.27 -0.93 -0.78 -0.20 -0.08 0.21 -0.48
        R 0.22 1.27 1.37 1.87 -1.70 0.46 0.92 -0.39 0.23 0.93
        N 1.14 -0.07 -0.12 0.81 0.18 0.37 -0.09 1.23 1.10 -1.73
        D 0.58 -0.22 -1.58 0.81 -0.92 0.15 -1.52 0.47 0.76 0.70
        C 0.12 -0.89 0.45 -1.05 -0.71 2.41 1.52 -0.69 1.13 1.10
        Q -0.47 0.24 0.07 1.10 1.10 0.59 0.84 -0.71 -0.03 -2.33
        E -1.45 0.19 -1.61 1.17 -1.31 0.40 0.04 0.38 -0.35 -0.12
        G 1.46 -1.96 -0.23 -0.16 0.10 -0.11 1.32 2.36 -1.66 0.46
        H -0.41 0.52 -0.28 0.28 1.61 1.01 -1.85 0.47 1.13 1.63
        I -0.73 -0.16 1.79 -0.77 -0.54 0.03 -0.83 0.51 0.66 -1.78
        L -1.04 0.00 -0.24 -1.10 -0.55 -2.05 0.96 -0.76 0.45 0.93
        K -0.34 0.82 -0.23 1.70 1.54 -1.62 1.15 -0.08 -0.48 0.60
        M -1.40 0.18 -0.42 -0.73 2.00 1.52 0.26 0.11 -1.27 0.27
        F -0.21 0.98 -0.36 -1.43 0.22 -0.81 0.67 1.10 1.71 -0.44
        P 2.06 -0.33 -1.15 -0.75 0.88 -0.45 0.30 -2.30 0.74 -0.28
        S 0.81 -1.08 0.16 0.42 -0.21 -0.43 -1.89 -1.15 -0.97 -0.23
        T 0.26 -0.70 1.21 0.63 -0.10 0.21 0.24 -1.15 -0.56 0.19
        W 0.30 2.10 -0.72 -1.57 -1.16 0.57 -0.48 -0.40 -2.30 -0.60
        Y 1.38 1.48 0.80 -0.56 -0.00 -0.68 -0.31 1.03 -0.05 0.53
        V -0.74 -0.71 2.04 -0.40 0.50 -0.81 -1.07 0.06 -0.46 0.65
    """
    )

    def __init__(self, n_jobs: Union[int, None] = None):
        """Class to generate pairwise distances between amino acid sequences
        based on kidera factors.

        Parameters
        ----------
        n_jobs
            Number of jobs to use for the pairwise distance calculation. 
            If None, use all jobs. 
        """
        self.kidera_factors = pd.read_csv(
            StringIO(self.KIDERA_FACTORS), sep=" ", header=None, index_col=0
        )
        self.n_jobs = -1 if n_jobs is None else n_jobs

    def _make_kidera_vectors(self, seqs: Collection) -> np.ndarray:
        """Convert each AA-sequence into a vector of kidera factors. 
        Sums over the kidera factors for each amino acid. """
        return np.vstack(
            [
                np.mean(
                    np.vstack([self.kidera_factors.loc[c, :].values for c in seq]),
                    axis=0,
                )
                for seq in seqs
            ]
        )

    def calc_dist_mat(self, seqs: Collection) -> np.ndarray:
        kidera_vectors = self._make_kidera_vectors(seqs)
        return pairwise_distances(
            kidera_vectors, metric="euclidean", n_jobs=self.n_jobs
        )


class _AlignmentDistanceCalculator(_DistanceCalculator):
    def __init__(
        self,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 1,
        n_jobs: Union[int, None] = None,
    ):
        """Class to generate pairwise alignment distances
        
        High-performance sequence alignment through parasail library [Daily2016]_

        Parameters
        ----------
        subst_mat
            Name of parasail substitution matrix
        gap_open
            Gap open penalty
        gap_extend
            Gap extend penatly
        n_jobs
            Number of processes to use. Will be passed to :meth:`multiprocessing.Pool`
        """
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.n_jobs = n_jobs

    def _align_row(self, seqs: np.ndarray, i_row: int) -> np.ndarray:
        """Generates a row of the triangular distance matrix. 
        
        Aligns `seqs[i_row]` with all other sequences in `seqs[i_row:]`. 

        Parameters
        ----------
        seqs
            Array of amino acid sequences 
        i_row
            Index of the row in the final distance matrix. Determines the target sequence. 

        Returns
        -------
        The i_th row of the final score matrix. 
        """
        subst_mat = parasail.Matrix(self.subst_mat)
        target = seqs[i_row]
        profile = parasail.profile_create_16(target, subst_mat)
        result = np.empty(len(seqs))
        result[:] = np.nan
        for j, s2 in enumerate(seqs[i_row:], start=i_row):
            r = parasail.nw_scan_profile_16(profile, s2, self.gap_open, self.gap_extend)
            result[j] = r.score

        return result

    def _calc_norm_factors(self, score_mat: np.ndarray) -> np.ndarray:
        """Calculate normalization factors to normaliza a score matrix between 0 and 1. 
        
        We define the normalization factors as the minimum of the self-alignment score
        of each pair of sequences. The refers to the max. possible score of an alignment
        between the two sequences. 
        """
        self_scores = np.diag(score_mat)
        a1, a2 = np.meshgrid(self_scores, self_scores)
        norm_factors = np.minimum(a1, a2)

        assert _is_symmetric(norm_factors), "Matrix not symmetric"

        return norm_factors

    def _score_to_dist(self, score_mat: np.ndarray) -> np.ndarray:
        """Convert an alignment score matrix into a distance between 0 and 1.
        This is achieved by dividing the alignment score with a normalization
        factor that refers to the maximum possible alignment score between
        two sequences. """
        assert np.all(
            np.argmax(score_mat, axis=1) == np.diag_indices_from(score_mat)
        ), """Max value not on the diagonal"""

        norm_factors = self._calc_norm_factors(score_mat)
        # normalize
        dist_mat = score_mat / norm_factors

        # upper bound is 1 already, set lower bound to 0
        dist_mat[dist_mat < 0] = 0

        # inverse (= turn into distance)
        dist_mat = 1 - dist_mat

        assert np.min(dist_mat) >= 0
        assert np.max(dist_mat) <= 1

        return dist_mat

    def _calc_score_mat(self, seqs: Collection) -> np.ndarray:
        """Calculate the alignment scores of all-against-all pairwise
        sequence alignments"""
        p = Pool(self.n_jobs)
        rows = p.starmap(self._align_row, zip(itertools.repeat(seqs), range(len(seqs))))

        score_mat = np.vstack(rows)
        assert score_mat.shape[0] == score_mat.shape[1]

        # mirror matrix at diagonal (https://stackoverflow.com/a/42209263/2340703)
        i_lower = np.tril_indices(score_mat.shape[0], -1)
        score_mat[i_lower] = score_mat.T[i_lower]

        assert _is_symmetric(score_mat), "Matrix not symmetric"

        return score_mat

    def calc_dist_mat(self, seqs: Collection) -> np.ndarray:
        """Calculate the distances between amino acid sequences based on
        of all-against-all pairwise sequence alignments.

        Parameters
        ----------
        seqs
            Array of amino acid sequences

        Returns
        -------
        Symmetric, square matrix of normalized alignment distances. 
        """
        score_mat = self._calc_score_mat(seqs)
        dist_mat = self._score_to_dist(score_mat)
        return dist_mat


def _dist_for_chain(
    adata, chain: Literal["TRA", "TRB"], distance_calculator: _DistanceCalculator
) -> List[np.ndarray]:
    """Compute distances for all combinations of 
    (TRx1,TRx1), (TRx1,TRx2), (TRx2, TRx1), (TRx2, TRx2). 
    
    Parameters
    ----------
    adata
        Annotated data matrix
    chain
        Chain to work on. Can be "TRA" or "TRB". Will include both 
        primary (TRA_1_cdr3) and secondary (TRA_2_cdr3) chains. 
    distance_calculator
        Class implementing a calc_dist_mat(seqs) function 
        that computes pariwise distances between all cdr3 sequences. 
    """

    chains = ["{}_{}".format(chain, i) for i in ["1", "2"]]
    tr_seqs = {k: adata.obs["{}_cdr3".format(k)].values for k in chains}
    unique_seqs = np.hstack(list(tr_seqs.values()))
    unique_seqs = np.unique(unique_seqs[~_is_na(unique_seqs)]).astype(str)
    # reverse mapping of amino acid sequence to index in distance matrix
    seq_to_index = {seq: i for i, seq in enumerate(unique_seqs)}

    logging.debug("Started computing {} pairwise distances.".format(chain))
    dist_mat = distance_calculator.calc_dist_mat(unique_seqs)
    logging.info("Finished computing {} pairwise distances.".format(chain))

    # indices of cells in adata that have a CDR3 sequence.
    cells_with_chain = {k: np.where(~_is_na(tr_seqs[k]))[0] for k in chains}
    # indices of the corresponding sequences in the distance matrix.
    seq_inds = {
        k: np.array([seq_to_index[tr_seqs[k][i]] for i in cells_with_chain[k]])
        for k in chains
    }

    # assert that the indices are correct...
    for k in chains:
        npt.assert_equal(unique_seqs[seq_inds[k]], tr_seqs[k][~_is_na(tr_seqs[k])])

    # compute cell x cell matrix for each combination of chains
    cell_mats = list()
    for chain1, chain2 in [(1, 1), (1, 2), (2, 2)]:
        chain1, chain2 = "{}_{}".format(chain, chain1), "{}_{}".format(chain, chain2)
        cell_mat = np.full([adata.n_obs] * 2, np.nan)

        # 2d indices in the cell matrix
        # This is several orders of magnitudes faster than using nested for loops.
        i_cm_0, i_cm_1 = np.meshgrid(cells_with_chain[chain1], cells_with_chain[chain2])
        # 2d indices of the sequences in the distance matrix
        i_dm_0, i_dm_1 = np.meshgrid(seq_inds[chain1], seq_inds[chain2])

        cell_mat[i_cm_0, i_cm_1] = dist_mat[i_dm_0, i_dm_1]

        if chain1 == chain2:
            # TRX1:TRX2 is not supposed to be symmetric
            assert _is_symmetric(cell_mat), "matrix not symmetric"

        cell_mats.append(cell_mat)
        if chain1 != chain2:
            cell_mats.append(cell_mat.T)

    return cell_mats


def tcr_neighbors(
    adata: AnnData,
    *,
    metric: Literal["alignment", "kidera", "identity", "levenshtein"] = "alignment",
    n_jobs: [int, None] = None,
    inplace: bool = True,
) -> Union[None, dict]:
    """Compute the TCRdist on CDR3 sequences. 
    The equivalent of scanpy.pp.neighbors for TCR sequences. 


    Parameters
    ----------
    metric
        Distance metric to use. `alignment` will calculate an alignment distance
        based on normalized BLOSUM62 scores. `kidera` calculates a distance 
        based on kidera factors. `identity` results in `0` for an identical sequence, 
        `1` for different sequence. 

    """
    if metric == "alignment":
        dist_calc = _AlignmentDistanceCalculator(n_jobs=n_jobs)
    elif metric == "kidera":
        dist_calc = _KideraDistanceCalculator(n_jobs=n_jobs)
    elif metric == "identity":
        dist_calc = _IdentityDistanceCalculator()
    elif metric == "levenshtein":
        dist_calc = _LevenshteinDistanceCalculator()
    else:
        raise ValueError("Invalid distance metric.")

    tra_dists = _dist_for_chain(adata, "TRA", dist_calc)
    trb_dists = _dist_for_chain(adata, "TRB", dist_calc)

    if inplace:
        if "sctcrpy" not in adata.uns:
            adata.uns["sctcrpy"] = dict()
        adata.uns["sctcrpy"]["tra_neighbors"] = tra_dists[0]
        adata.uns["sctcrpy"]["trb_neighbors"] = trb_dists[0]
    else:
        return tra_dists, trb_dists


def get_igraph_from_adjacency(adj, edge_type=None, directed=None):
    """Get igraph graph from adjacency matrix.
    Better than Graph.Adjacency for sparse matrices
    """

    g = ig.Graph(directed=False)
    g.add_vertices(adj.shape[0])  # this adds adjacency.shape[0] vertices

    sources, targets = np.triu(adj, k=1).nonzero()
    weights = adj[sources, targets].astype("float")
    g.add_edges(list(zip(sources, targets)))

    g.es["weight"] = weights
    if edge_type is not None:
        g.es["type"] = edge_type

    if g.vcount() != adj.shape[0]:
        logging.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


def _merge_graphs(g1, g2):
    g = g1.copy()
    g.add_edges(g2.get_edgelist())
    assert g.ecount() == g1.ecount() + g2.ecount()
    g.es[: g1.ecount()]["type"] = g1.es["type"]
    g.es[g1.ecount() :]["type"] = g2.es["type"]
    return g


def define_clonotypes(
    adata, strategy: Literal["all", "any", "lenient"] = "any", inplace=True
):
    """Define clonotypes based on cdr3 identity.
    
    For now, uses primary TRA and TRB only. 

    Parameters
    ----------
    strategy:
        "all" - both TRA and TRB need to match
        "any" - either TRA or TRB need to match
        "lenient" - both TRA and TRB need to match, however it is tolerated if for a 
          given cell pair, no TRA or TRB sequence is available. 

    """
    tra_dists, trb_dists = tcr_neighbors(adata, metric="identity", inplace=False)

    if strategy == "any":
        adj_tra = tra_dists[0] == 0
        adj_trb = trb_dists[0] == 0

        g_tra = get_igraph_from_adjacency(adj_tra, "TRA")
        g_trb = get_igraph_from_adjacency(adj_trb, "TRB")

        g = _merge_graphs(g_tra, g_trb)

    elif strategy == "all":
        adj = (tra_dists[0] == 0) & (trb_dists[0] == 0)

        g = get_igraph_from_adjacency(adj)
    else:
        raise ValueError("Unknown strategy. ")

    # find all connected partitions that are
    # connected by at least one edge
    partitions = g.components(mode="weak")

    if not inplace:
        return partitions.membership, partitions.graph
    else:
        if "sctcrpy" not in adata.uns:
            adata.uns["sctcrpy"] = dict()
        # TODO this cannot be saved with adata.write_h5ad
        adata.uns["sctcrpy"]["clonotype_graph"] = partitions.graph
        assert len(partitions.membership) == adata.obs.shape[0]
        adata.obs["clonotype"] = partitions.membership
        adata.obs["clonotype_size"] = adata.obs.groupby("clonotype")[
            "clonotype"
        ].transform("count")


def clonotype_network(
    adata, *, layout="fr", key_added="X_clonotype_network", min_size=1
):
    """Build the clonotype network for plotting
    
    Parameters
    ----------
    min_size
        Only show clonotypes with at least `min_size` cells.
    """
    if (
        "clonotype" not in adata.obs.columns
        or "clonotype_size" not in adata.obs.columns
    ):
        raise ValueError("You need to run define_clonotypes first.")

    graph = adata.uns["sctcrpy"]["clonotype_graph"]
    subgraph_idx = np.where(adata.obs["clonotype_size"].values >= min_size)[0]
    graph = graph.subgraph(subgraph_idx)
    layout_ = graph.layout(layout)
    adata.uns["sctcrpy"]["clonotype_subgraph"] = graph
    adata.uns["sctcrpy"]["clonotype_layout"] = layout_
    adata.uns["sctcrpy"]["clonotype_subgraph_idx"] = subgraph_idx
