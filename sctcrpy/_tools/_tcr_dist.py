import parasail
from multiprocessing import Pool
import itertools
from anndata import AnnData
from typing import Union, Collection
import pandas as pd
import numpy as np
from scanpy import logging
from sklearn.metrics import pairwise_distances
import numpy.testing as npt
from .._util import _is_na, _is_symmetric
import abc
import textwrap
from io import StringIO


class _DistanceCalculator(abc.ABC):
    @abc.abstractmethod
    def calc_dist_mat(self, seqs: np.ndarray) -> np.ndarray:
        """Calculate a symmetric, pairwise distance matrix of all sequences in `seq`.
        Distances are non-negative values"""
        pass


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

    def __init__(self, n_jobs=-1):
        """Class to generate pairwise distances between amino acid sequences
        based on kidera factors.
        """
        self.kidera_factors = pd.read_csv(
            StringIO(self.KIDERA_FACTORS), sep=" ", header=None, index_col=0
        )
        self.n_jobs = n_jobs

    def _make_kidera_vectors(self, seqs: Collection) -> np.ndarray:
        """Convert each AA-sequence into a vector of kidera factors. 
        Sums over the kidera factors for each amino acid. """
        return np.vstack(
            np.sum(np.vstack(self.kidera_factors.loc[c, :].values for c in seq), axis=0)
            for seq in seqs
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
        n_jobs=None,
    ):
        """Class to generate pairwise alignment distances
        
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

    def calc_dist_mat(self, seqs: Collection) -> np.ndarray:
        """Calculate the scores of all-against-all pairwise sequence alignments.

        Parameters
        ----------
        seqs
            Array of amino acid sequences

        Returns
        -------
        Symmetric, square matrix of pairwise alignment scores
        """
        p = Pool(self.n_jobs)
        rows = p.starmap(self._align_row, zip(itertools.repeat(seqs), range(len(seqs))))

        score_mat = np.vstack(rows)
        assert score_mat.shape[0] == score_mat.shape[1]

        # mirror matrix at diagonal (https://stackoverflow.com/a/42209263/2340703)
        i_lower = np.tril_indices(score_mat.shape[0], -1)
        score_mat[i_lower] = score_mat.T[i_lower]

        assert _is_symmetric(score_mat), "Matrix not symmetric"

        return score_mat


def _calc_norm_factors(score_mat: np.ndarray) -> np.ndarray:
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


def _score_to_dist(score_mat: np.ndarray) -> np.ndarray:
    """Convert an alignment score matrix into a distance between 0 and 1."""
    assert np.all(
        np.argmax(score_mat, axis=1) == np.diag_indices_from(score_mat)
    ), """Max value not on the diagonal"""

    norm_factors = _calc_norm_factors(score_mat)
    # normalize
    dist_mat = score_mat / norm_factors

    # upper bound is 1 already, set lower bound to 0
    dist_mat[dist_mat < 0] = 0

    # inverse (= turn into distance)
    dist_mat = 1 - dist_mat

    assert np.min(dist_mat) >= 0
    assert np.max(dist_mat) <= 1

    return dist_mat


def _dist_for_chain(adata, chain):
    """Compute distances for all combinations of 
    (TRX1,TRX1), (TRX1,TRX2), (TRX2, TRX1), (TRX2, TRX2)"""
    aligner = _Aligner()

    chains = ["{}_{}".format(chain, i) for i in ["1", "2"]]
    tr_seqs = {k: adata.obs["{}_cdr3".format(k)].values for k in chains}
    unique_seqs = np.hstack(list(tr_seqs.values()))
    unique_seqs = np.unique(unique_seqs[~_is_na(unique_seqs)]).astype(str)
    # reverse mapping of amino acid sequence to index in distance matrix
    seq_to_index = {seq: i for i, seq in enumerate(unique_seqs)}

    logging.debug("Started computing {} alignments.".format(chain))
    score_mat = aligner.make_score_mat(unique_seqs)
    logging.info("Finished computing {} alignments.".format(chain))
    dist_mat = _score_to_dist(score_mat)

    # indices of cells in adata that have a CDR3 sequence.
    cells_with_chain = {k: np.where(~_is_na(tr_seqs[k]))[0] for k in chains}
    # indices of the corresponding sequences in the distance matrix.
    seq_inds = {
        k: np.array([seq_to_index[tr_seqs[k][i]] for i in cells_with_chain[k]])
        for k in chains
    }

    # assert that the indices are right...
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

    return cell_mats


def tcr_dist(
    adata: AnnData,
    *,
    inplace: bool = True,
    reduction_same_chain=np.fmin,
    reduction_other_chain=np.fmin
) -> Union[None, dict]:
    """Compute the TCRdist on CDR3 sequences. 

    High-performance sequence alignment through parasail library [Daily2016]_

    Parameters
    ----------
    adata
    subst_mat
    gap_open
    gap_extend
    """

    tra_dists = _dist_for_chain(adata, "TRA")
    trb_dists = _dist_for_chain(adata, "TRB")

    return reduction_other_chain.reduce(
        [reduction_same_chain.reduce(tra_dists), reduction_same_chain.reduce(trb_dists)]
    )
