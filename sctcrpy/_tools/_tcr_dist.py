import scipy.sparse
import parasail
from multiprocessing import Pool
import itertools
from anndata import AnnData
from typing import Union
import pandas as pd
import numpy as np
from scanpy import logging
import numpy.testing as npt
from .._util import _is_na, _is_symmetric


class _Aligner:
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

    def make_score_mat(self, seqs: np.ndarray) -> np.ndarray:
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


def tcr_dist(adata: AnnData, *, inplace: bool = True) -> Union[None, dict]:
    """Compute the TCRdist on CDR3 sequences. 

    High-performance sequence alignment through parasail library [Daily2016]_

    Parameters
    ----------
    adata
    subst_mat
    gap_open
    gap_extend
    """

    aligner = _Aligner()

    chain = "TRA"
    chains = ["{}_{}".format(chain, i) for i in ["1", "2"]]
    tr_seqs = {k: adata.obs["{}_cdr3".format(k)].values for k in chains}
    unique_seqs = np.hstack(list(tr_seqs.values()))
    unique_seqs = np.unique(unique_seqs[~_is_na(unique_seqs)]).astype(str)
    # reverse mapping of amino acid sequence to index in distance matrix
    seq_to_index = {seq: i for i, seq in enumerate(unique_seqs)}

    logging.debug("Started computing alignments.")
    score_mat = aligner.make_score_mat(unique_seqs)
    logging.info("Finished computing alignments.")
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

    cell_mats = dict()
    for chain1, chain2 in itertools.combinations_with_replacement(chains, 2):
        k = "{}_{}".format(chain1, chain2)
        cell_mats[k] = np.full([adata.n_obs] * 2, np.nan)
        i_cm_0, i_cm_1 = np.meshgrid(cells_with_chain[chain1], cells_with_chain[chain2])
        i_dm_0, i_dm_1 = np.meshgrid(seq_inds[chain1], seq_inds[chain2])

        cell_mats[k][i_cm_0, i_cm_1] = dist_mat[i_dm_0, i_dm_1]

        if chain1 == chain2:
            assert _is_symmetric(cell_mats[k]), "matrix not symmetric"

    return cell_mats
