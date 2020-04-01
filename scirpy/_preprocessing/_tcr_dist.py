import parasail
from .._util._multiprocessing import EnhancedPool as Pool
import itertools
from anndata import AnnData
from typing import Union, Collection, List, Tuple, Dict
from .._compat import Literal
import numpy as np
from scanpy import logging
import numpy.testing as npt
from .._util import _is_na, _is_symmetric, _reduce_nonzero
import abc
from Levenshtein import distance as levenshtein_dist
import scipy.spatial
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
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
    def calc_dist_mat(self, seqs: np.ndarray) -> coo_matrix:
        """Calculate the upper diagnoal, pairwise distance matrix of all 
        sequences in `seq`.

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

    def calc_dist_mat(self, seqs: np.ndarray) -> coo_matrix:
        """The offsetted matrix is the identity matrix."""
        return scipy.sparse.identity(len(seqs), dtype=self.DTYPE, format="coo")


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
        rows = p.starmap_progress(
            self._compute_row,
            zip(itertools.repeat(seqs), range(len(seqs))),
            chunksize=200,
            total=len(seqs),
        )
        p.close()

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        assert score_mat.shape[0] == score_mat.shape[1]

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

    def calc_dist_mat(self, seqs: Collection) -> coo_matrix:
        """Calculate the distances between amino acid sequences based on
        of all-against-all pairwise sequence alignments.

        Parameters
        ----------
        seqs
            Array of amino acid sequences

        Returns
        -------
        Upper diagonal distance matrix of normalized alignment distances. 
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
        rows = p.starmap_progress(
            self._align_row,
            zip(
                itertools.repeat(seqs),
                itertools.repeat(self_alignment_scores),
                range(len(seqs)),
            ),
            chunksize=200,
            total=len(seqs),
        )
        p.close()

        score_mat = scipy.sparse.vstack(rows)
        score_mat.eliminate_zeros()
        assert score_mat.shape[0] == score_mat.shape[1]

        return score_mat


def _seq_to_cell_idx(
    unique_seqs: np.ndarray, cdr_seqs: np.ndarray
) -> Dict[int, List[int]]:
    """
    Compute sequence to cell index for a single chain (e.g. `TRA_1`). 

    Maps cell_idx -> [list, of, seq_idx]. 
    Useful to build a cell x cell matrix from a seq x seq matrix. 

    Computes magic lookup indexes in linear time

    Parameters
    ----------
    unique_seqs
        Pool of all unique cdr3 sequences (length = #unique cdr3 sequences)
    cdr_seqs
        CDR3 sequences for the current chain (length = #cells)

    Returns
    -------
    Sequence2Cell mapping    
    """
    # 1) reverse mapping of amino acid sequence to index in sequence-distance matrix
    seq_to_index = {seq: i for i, seq in enumerate(unique_seqs)}

    # 2) indices of cells in adata that have a CDR3 sequence.
    cells_with_chain = np.where(~_is_na(cdr_seqs))[0]

    # 3) indices of the corresponding sequences in the distance matrix.
    seq_inds = {
        chain_id: seq_to_index[cdr_seqs[chain_id]] for chain_id in cells_with_chain
    }

    # 4) list of cell-indices in the cell distance matrix for each sequence
    seq_to_cell = {seq_id: list() for seq_id in seq_to_index.values()}
    for cell_id in cells_with_chain:
        seq_id = seq_inds[cell_id]
        seq_to_cell[seq_id].append(cell_id)

    return seq_to_cell


def _dist_for_chain(
    adata,
    chain: Literal["TRA", "TRB"],
    distance_calculator: _DistanceCalculator,
    merge_chains=Literal["primary_only", "all"],
) -> List[np.ndarray]:
    """Computes the cell x cell distance matrix for either TRA or TRB chains. 

    The option merge_chains specifies how primary/secondary chain are handled. 
    
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
    merge_chains
        Whether to consider only the most abundant pair of TCR sequences, 
        or all. When `all` the distance is reduced to the minimal distance of 
        all receptors. 
    """
    chain_inds = ["1"] if merge_chains == "primary_only" else ["1", "2"]
    chains = ["{}_{}".format(chain, i) for i in chain_inds]
    cdr_seqs = {k: adata.obs["{}_cdr3".format(k)].values for k in chains}
    unique_seqs = np.hstack(list(cdr_seqs.values()))
    unique_seqs = np.unique(unique_seqs[~_is_na(unique_seqs)]).astype(str)
    seq_to_cell = {k: _seq_to_cell_idx(unique_seqs, cdr_seqs[k]) for k in chains}
    logging.debug("Finished computing indices")

    dist_mat = distance_calculator.calc_dist_mat(unique_seqs)
    logging.info("Finished computing {} pairwise distances.".format(chain))

    # compute cell x cell distance matrix from seq x seq matrix
    def _seq_to_cell(dist_mat, seq_to_cell, merge_chains):
        """Build coordinates for cell x cell distance matrix"""
        if merge_chains == "primary_only":
            k = chain + "_1"
            for row, col, value in zip(dist_mat.row, dist_mat.col, dist_mat.data):
                for cell_row in seq_to_cell[k][row]:
                    for cell_col in seq_to_cell[k][col]:
                        # build the full matrix from triagular one:
                        yield value, cell_row, cell_col
                        yield value, cell_col, cell_row
        elif merge_chains == "all":
            raise NotImplementedError()
        else:
            raise ValueError("Unknown value for `merge_chains`")

    values, rows, cols = zip(*_seq_to_cell(dist_mat, seq_to_cell, merge_chains))
    dist_mat = coo_matrix((values, (rows, cols)))
    dist_mat.eliminate_zeros()
    return dist_mat.to_csr()


def tcr_dist(
    adata: AnnData,
    *,
    metric: Literal["alignment", "identity", "levenshtein"] = "alignment",
    cutoff: float = 2,
    merge_chains: Literal["primary_only", "all"] = "primary_only",
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

    if cutoff == 0:
        return dist

    connectivities = dist.copy()

    # actual distances
    d = connectivities.data - 1

    # structure of the matrix stayes the same, we can safely change the data only
    connectivities.data = (cutoff - d) / cutoff
    return connectivities


def tcr_neighbors(
    adata: AnnData,
    *,
    metric: Literal["identity", "alignment", "levenshtein"] = "alignment",
    cutoff: int = 2,
    strategy: Literal["TRA", "TRB", "all", "any"] = "all",
    merge_chains: Literal["primary_only", "all"] = "primary_only",
    key_added: str = "tcr_neighbors",
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
    merge_chains:
        Use only primary chains ("TRA_1", "TRB_1") or all four chains? 
        When considering all four chains, at least one of them needs
        to match. 
    key_added:
        dict key under which the result will be stored in `adata.uns["scirpy"]`
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
    tra_dist, trb_dist = tcr_dist(
        adata, metric=metric, n_jobs=n_jobs, cutoff=cutoff, merge_chains=merge_chains
    )

    dist = _reduce_dists(tra_dist, trb_dist, strategy)
    dist.eliminate_zeros()
    logging.debug("Finished reducing dists across chains.")

    connectivities = _dist_to_connectivities(dist, cutoff=cutoff)
    connectivities.eliminate_zeros()
    logging.debug("Finished converting distances to connectivities. ")

    if not inplace:
        return connectivities, dist
    else:
        adata.uns[key_added] = dict()
        adata.uns[key_added]["params"] = {
            "metric": metric,
            "cutoff": cutoff,
            "strategy": strategy,
            "merge_chains": merge_chains,
        }
        adata.uns[key_added]["connectivities"] = connectivities
        adata.uns[key_added]["distances"] = connectivities
