import parasail
from multiprocessing import Pool
import itertools
from anndata import AnnData
from typing import Union, Collection, List, Tuple
from .._compat import Literal
import numpy as np
from scanpy import logging
import numpy.testing as npt
from .._util import _is_na, _is_symmetric, _reduce_nonzero
import abc
from Levenshtein import distance as levenshtein_dist
import scipy.spatial
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
from functools import reduce


class _DistanceCalculator(abc.ABC):
    DTYPE = "uint8"

    def __init__(self, cutoff: float, n_jobs: Union[int, None] = None):
        """
        Parameters
        ----------
        cutoff:
            Will eleminate distances > cutoff to make efficient 
            use of sparse matrices. 
        n_jobs
            Number of jobs to use for the pairwise distance calculation. 
            If None, use all jobs. 
        """
        if cutoff > 255:
            raise ValueError(
                "Using a cutoff > 255 is not possible due to the `uint8` dtype used"
            )
        self.cutoff = cutoff
        self.n_jobs = n_jobs

    @abc.abstractmethod
    def calc_dist_mat(self, seqs: np.ndarray) -> csr_matrix:
        """Calculate a symmetric, pairwise distance matrix of all sequences in `seq`.

         * Only returns distances <= cutoff
         * Distances are non-negative values.
         * The resulting matrix is offsetted by 1 to allow efficient use
           of sparse matrices ($d' = d+1$).
           I.e. 0 -> d > cutoff; 1 -> d == 0; 2 -> d == 1; ...
        """
        pass


class _IdentityDistanceCalculator(_DistanceCalculator):
    """Calculate the distance between TCR based on the identity 
    of sequences. I.e. 0 = sequence identical, 1 = sequences not identical
    """

    def __init__(self, cutoff: float = 0, n_jobs: Union[int, None] = None):
        """For this DistanceCalculator, per definition, the cutoff = 0. 
        The `cutoff` argument is ignored. """
        super().__init__(cutoff, n_jobs)

    def calc_dist_mat(self, seqs: np.ndarray) -> csr_matrix:
        """The offsetted matrix is the identity matrix."""
        return scipy.sparse.identity(len(seqs), dtype=self.DTYPE, format="csr")


class _LevenshteinDistanceCalculator(_DistanceCalculator):
    """Calculates the Levenshtein (i.e. edit-distance) between sequences. """

    def _compute_row(self, seqs: np.ndarray, i_row: int) -> coo_matrix:
        """Compute a row of the upper diagnomal distance matrix"""
        target = seqs[i_row]

        def coord_generator():
            for j, s2 in enumerate(seqs[i_row:], start=i_row):
                d = levenshtein_dist(target, s2)
                if d <= self.cutoff:
                    yield d + 1, j

        d, col = zip(*coord_generator())
        row = np.zeros(len(col), dtype="int")
        return coo_matrix((d, (row, col)), dtype=self.DTYPE, shape=(1, seqs.size))

    def calc_dist_mat(self, seqs: np.ndarray) -> csr_matrix:
        p = Pool(self.n_jobs)
        rows = p.starmap(
            self._compute_row, zip(itertools.repeat(seqs), range(len(seqs)))
        )

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        score_mat = score_mat.tocsr()
        assert score_mat.shape[0] == score_mat.shape[1]

        # mirror matrix at diagonal (https://stackoverflow.com/a/42209263/2340703)
        i_lower = np.tril_indices(score_mat.shape[0], -1)
        score_mat[i_lower] = score_mat.T[i_lower]

        assert _is_symmetric(score_mat), "Matrix not symmetric"

        return score_mat


class _AlignmentDistanceCalculator(_DistanceCalculator):
    """Calculates distance between sequences based on pairwise sequence alignment. 

    The distance between two sequences is defined as $S_{1,2}^{max} - S_{1,2}$ 
    where $S_{1,2} $ is the alignment score of sequences 1 and 2 and $S_{1,2}^{max}$ 
    is the max. achievable alignment score of sequences 1 and 2 defined as 
    $\\min(S_{1,1}, S_{2,2})$. 
    """

    def __init__(
        self,
        cutoff: float,
        n_jobs: Union[int, None] = None,
        *,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 1,
    ):
        """Class to generate pairwise alignment distances
        
        High-performance sequence alignment through parasail library [Daily2016]_

        Parameters
        ----------
        cutoff
            see `_DistanceCalculator`
        n_jobs
            see `_DistanceCalculator`
        subst_mat
            Name of parasail substitution matrix
        gap_open
            Gap open penalty
        gap_extend
            Gap extend penatly
        """
        super().__init__(cutoff, n_jobs)
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def _align_row(
        self, seqs: np.ndarray, self_alignment_scores: np.array, i_row: int
    ) -> np.ndarray:
        """Generates a row of the triangular distance matrix. 
        
        Aligns `seqs[i_row]` with all other sequences in `seqs[i_row:]`. 

        Parameters
        ----------
        seqs
            Array of amino acid sequences
        self_alignment_scores
            Array containing the scores of aligning each sequence in `seqs` 
            with itself. This is used as a reference value to turn 
            alignment scores into distances. 
        i_row
            Index of the row in the final distance matrix. Determines the target sequence. 

        Returns
        -------
        The i_th row of the final score matrix. 
        """
        subst_mat = parasail.Matrix(self.subst_mat)
        target = seqs[i_row]
        profile = parasail.profile_create_16(target, subst_mat)

        def coord_generator():
            for j, s2 in enumerate(seqs[i_row:], start=i_row):
                r = parasail.nw_scan_profile_16(
                    profile, s2, self.gap_open, self.gap_extend
                )
                max_score = np.min(self_alignment_scores[[i_row, j]])
                d = max_score - r.score
                if d <= self.cutoff:
                    yield d + 1, j

        d, col = zip(*coord_generator())
        row = np.zeros(len(col), dtype="int")
        return coo_matrix((d, (row, col)), dtype=self.DTYPE, shape=(1, len(seqs)))

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
        # first, calculate self-alignments. We need them as refererence values
        # to turn scores into dists
        self_alignment_scores = np.array(
            [
                parasail.nw_scan_16(
                    s,
                    s,
                    self.gap_open,
                    self.gap_extend,
                    parasail.Matrix(self.subst_mat),
                ).score
                for s in seqs
            ]
        )

        p = Pool(self.n_jobs)
        rows = p.starmap(
            self._align_row,
            zip(
                itertools.repeat(seqs),
                itertools.repeat(self_alignment_scores),
                range(len(seqs)),
            ),
        )

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        score_mat = score_mat.tocsr()
        assert score_mat.shape[0] == score_mat.shape[1]

        # mirror matrix at diagonal (https://stackoverflow.com/a/42209263/2340703)
        i_lower = np.tril_indices(score_mat.shape[0], -1)
        score_mat[i_lower] = score_mat.T[i_lower]

        assert _is_symmetric(score_mat), "Matrix not symmetric"

        return score_mat


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
        cell_mat = csr_matrix((adata.n_obs, adata.n_obs))

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


def tcr_dist(
    adata: AnnData,
    *,
    metric: Literal["alignment", "identity", "levenshtein"] = "alignment",
    cutoff: float = 2,
    n_jobs: Union[int, None] = None,
) -> Tuple[csr_matrix, csr_matrix]:
    """Computes the distances between CDR3 sequences 

    Parameters
    ----------
    metric
        Distance metric to use. `alignment` will calculate an alignment distance
        based on normalized BLOSUM62 scores. 
        `identity` results in `0` for an identical sequence, `1` for different sequence. 
        `levenshtein` is the Levenshtein edit distance between two strings. 
    cutoff
        Only store distances < cutoff in the sparse distance matrix
    j_jobs
        Number of CPUs to use for alignment and levenshtein distance. 
        Default: use all CPUS. 

    Returns
    -------
    tra_dists, trb_dists

    """
    if metric == "alignment":
        dist_calc = _AlignmentDistanceCalculator(cutoff=cutoff, n_jobs=n_jobs)
    elif metric == "identity":
        dist_calc = _IdentityDistanceCalculator(cutoff=cutoff)
    elif metric == "levenshtein":
        dist_calc = _LevenshteinDistanceCalculator(cutoff=cutoff, n_jobs=n_jobs)
    else:
        raise ValueError("Invalid distance metric.")

    tra_dists = _dist_for_chain(adata, "TRA", dist_calc)
    trb_dists = _dist_for_chain(adata, "TRB", dist_calc)

    return tra_dists, trb_dists


def _reduce_chains(
    dists: csr_matrix, mode: Literal["primary_only", "all"]
) -> csr_matrix:
    """Combine distances between primary and secondary cdr3 chains. 

    Possible modes are
     * "primary only": use only the primary chain
     * "all": use the minimal distance between any of the primary and secondary chains. 
     """
    if mode == "primary_only":
        return dists[0]
    elif mode == "all":
        return reduce(_reduce_nonzero, dists)
    else:
        raise ValueError("Unknown value for `mode`")


def _reduce_dists(
    tra_dist, trb_dist, mode: Literal["TRA", "TRB", "all", "any"]
) -> csr_matrix:
    """Combine TRA and TRB distances into a single distance matrix based on `mode`. 

    modes: 
     * "TRA": use only the TRA distance
     * "TRB": use only the TRB distance 
     * "any": Either TRA or TRB needs to have a distance < cutoff. The 
        resulting distance will be the minimum distance of either TRA or TRB. 
     * "all": Both TRA and TRB need to have a distance < cutoff. The 
        resulting distance will be the maximum distance of either TRA or TRB. 
    """
    if mode == "TRA":
        return tra_dist
    elif mode == "TRB":
        return trb_dist
    elif mode == "any":
        return _reduce_nonzero(tra_dist, trb_dist)
    elif mode == "all":
        # multiply == logical and for boolean (sparse) matrices
        return tra_dist.maximum(trb_dist).multiply(tra_dist > 0).multiply(trb_dist > 0)
    else:
        raise ValueError("Unknown mode. ")


def _dist_to_connectivities(dist, cutoff):
    """Convert a (sparse) distance matrix to a weighted adjacency matrix. 

    Parameters
    ----------
    dist
        distance matrix. already contains `0` entries for edges 
        that are not connected. 
    cutoff
        cutoff that was used to filter the distance matrix. 
        Will be used to normalize the distances and refers
        to the maximum possible value in dist. 
    """
    assert isinstance(dist, csr_matrix)
    # actual distances
    connectivities = dist.copy()
    d = connectivities.data - 1
    # structure of the matrix stayes the same, we can safely change the data only
    connectivities.data = (cutoff - d) / cutoff
    return connectivities


def tcr_neighbors(
    adata: AnnData,
    *,
    metric: Literal["identity", "alignment", "levenshtein"] = "alignment",
    cutoff: int = 2,
    strategy: Literal["TRA", "TRB", "all", "any"] = "any",
    chains: Literal["primary_only", "all"] = "primary_only",
    key_added: str = "neighbors",
    inplace: bool = True,
    n_jobs: Union[int, None] = None,
) -> Union[Tuple[csr_matrix, csr_matrix], None]:
    """Construct a cell x cell neighborhood graph based on CDR3 sequence
    similarity. 

    Parameters
    ----------
    metric
        "identity" = Calculate 0/1 distance based on sequence identity. Equals a 
            cutoff of 0. 
        "alignment" - Calculate distance using pairwise sequence alignment 
            and BLOSUM62 matrix
        "levenshtein" - Levenshtein edit distance
    cutoff
        Two cells with a distance <= the cutoff will be connected. 
        If cutoff = 0, the CDR3 sequences need to be identical. In this 
        case, no alignment is performed. 
    strategy:
        "TRA" - only consider TRA sequences
        "TRB" - only consider TRB sequences
        "all" - both TRA and TRB need to match
        "any" - either TRA or TRB need to match
    chains:
        Use only primary chains ("TRA_1", "TRB_1") or all four chains? 
        When considering all four chains, at least one of them needs
        to match. 
    key_added:
        dict key under which the result will be stored in `adata.uns["sctcrpy"]`
        when `inplace` is True. 
    inplace:
        If True, store the results in adata.uns. If False, returns
        the results. 
    n_jobs:
        Number of cores to use for alignment and levenshtein distance. 
    
    Returns
    -------
    connectivities
        weighted adjacency matrix
    dist
        cell x cell distance matrix with the distances as computed according to `metric`
        offsetted by 1 to make use of sparse matrices. 
    """
    if cutoff == 0:
        metric = "identity"
    tra_dists, trb_dists = tcr_dist(adata, metric=metric, n_jobs=n_jobs, cutoff=cutoff)

    tra_dist, trb_dist = (
        _reduce_chains(tra_dists, chains),
        _reduce_chains(trb_dists, chains),
    )

    assert _is_symmetric(tra_dist)
    assert _is_symmetric(trb_dist)

    dist = _reduce_dists(tra_dist, trb_dist, strategy)
    connectivities = _dist_to_connectivities(dist, cutoff=cutoff)

    if not inplace:
        return connectivities, dist
    else:
        if "sctcrpy" not in adata.uns:
            adata.uns["sctcrpy"] = dict()
        adata.uns["sctcrpy"][key_added] = dict()
        adata.uns["sctcrpy"][key_added]["params"] = {
            "metric": metric,
            "cutoff": cutoff,
            "strategy": strategy,
            "chains": chains,
        }
        adata.uns["sctcrpy"][key_added]["connectivities"] = connectivities
        adata.uns["sctcrpy"][key_added]["distances"] = connectivities
