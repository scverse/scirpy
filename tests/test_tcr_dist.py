import pytest
from scirpy._preprocessing._tcr_dist import (
    _AlignmentDistanceCalculator,
    _DistanceCalculator,
    _IdentityDistanceCalculator,
    _LevenshteinDistanceCalculator,
    _dist_for_chain,
    _reduce_dists,
    _dist_to_connectivities,
    _seq_to_cell_idx,
)
import numpy as np
import numpy.testing as npt
import scirpy as st
import scipy.sparse
from scirpy._util import _reduce_nonzero
from functools import reduce
from .fixtures import adata_cdr3


@pytest.fixture
def adata_cdr3_mock_distance_calculator():
    class MockDistanceCalculator(_DistanceCalculator):
        def __init__(self, n_jobs=None):
            pass

        def calc_dist_mat(self, seqs):
            """Don't calculate distances, but return the
            hard-coded distance matrix needed for the test"""
            npt.assert_equal(seqs, ["AAA", "AHA"])
            return scipy.sparse.coo_matrix(np.array([[1, 4], [0, 1]]))

    return MockDistanceCalculator()


def test_identity_dist():
    identity = _IdentityDistanceCalculator()
    npt.assert_almost_equal(
        identity.calc_dist_mat(["ARS", "ARS", "RSA"]).toarray(),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


def test_levenshtein_compute_row():
    levenshtein1 = _LevenshteinDistanceCalculator(1)
    seqs = np.array(["A", "AAA", "AA"])
    row0 = levenshtein1._compute_row(seqs, 0)
    row2 = levenshtein1._compute_row(seqs, 2)
    assert row0.getnnz() == 2
    assert row2.getnnz() == 1
    npt.assert_equal(row0.toarray(), [[1, 0, 2]])
    npt.assert_equal(row2.toarray(), [[0, 0, 1]])


def test_levensthein_dist():
    levenshtein10 = _LevenshteinDistanceCalculator(10)
    levenshtein1 = _LevenshteinDistanceCalculator(1, n_jobs=1)
    npt.assert_almost_equal(
        levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"])).toarray(),
        np.array([[1, 2, 3, 3], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )
    npt.assert_almost_equal(
        levenshtein1.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"])).toarray(),
        np.array([[1, 2, 0, 0], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )


def test_align_row():
    aligner = _AlignmentDistanceCalculator(cutoff=255)
    aligner10 = _AlignmentDistanceCalculator(cutoff=10)
    seqs = ["AWAW", "VWVW", "HHHH"]
    self_alignment_scores = np.array([30, 30, 32])
    row0 = aligner._align_row(seqs, self_alignment_scores, 0)
    row2 = aligner._align_row(seqs, self_alignment_scores, 2)
    npt.assert_equal(row0.toarray(), [[1, 9, 39]])
    npt.assert_equal(row2.toarray(), [[0, 0, 1]])

    row0_10 = aligner10._align_row(seqs, self_alignment_scores, 0)
    npt.assert_equal(row0_10.toarray(), [[1, 9, 0]])


def test_alignment_dist():
    with pytest.raises(ValueError):
        _AlignmentDistanceCalculator(3000)
    aligner = _AlignmentDistanceCalculator(cutoff=255, n_jobs=1)
    aligner10 = _AlignmentDistanceCalculator(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    npt.assert_almost_equal(
        res.toarray(), np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]])
    )

    res = aligner10.calc_dist_mat(seqs)
    npt.assert_almost_equal(res.toarray(), np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]]))


def test_seq_to_cell_idx():
    unique_seqs = np.array(["AAA", "ABA", "CCC", "XXX", "AA"])
    cdr_seqs = np.array(["AAA", "CCC", "ABA", "CCC", np.nan, "AA", "AA"])
    result = _seq_to_cell_idx(unique_seqs, cdr_seqs)
    assert result == {0: [0], 1: [2], 2: [1, 3], 3: [], 4: [5, 6]}


def test_dist_for_chain_identity(adata_cdr3):
    identity = _IdentityDistanceCalculator()

    cell_mat = _dist_for_chain(adata_cdr3, "TRA", identity, merge_chains="primary_only")
    npt.assert_equal(
        cell_mat.toarray(),
        np.array(
            [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0] * 5, [1, 0, 0, 1, 0], [0] * 5,]
        ),
    )

    cell_mat_all = _dist_for_chain(adata_cdr3, "TRA", identity, merge_chains="all")
    npt.assert_equal(
        cell_mat_all.toarray(),
        np.array(
            [[1, 0, 0, 1, 1], [1, 1, 0, 0, 0], [0] * 5, [1, 0, 0, 1, 1], [0] * 5,]
        ),
    )


def test_dist_for_chain(adata_cdr3, adata_cdr3_mock_distance_calculator):
    cell_mat = _dist_for_chain(
        adata_cdr3,
        "TRA",
        adata_cdr3_mock_distance_calculator,
        merge_chains="primary_only",
    )
    print(cell_mat.toarray())
    assert cell_mat.nnz == 9
    npt.assert_equal(
        cell_mat.toarray(),
        np.array(
            [[1, 4, 0, 1, 0], [4, 1, 0, 4, 0], [0] * 5, [1, 4, 0, 1, 0], [0] * 5,]
        ),
    )

    cell_mat_all = _dist_for_chain(
        adata_cdr3, "TRA", adata_cdr3_mock_distance_calculator, merge_chains="all"
    )
    assert cell_mat_all.nnz == 16
    npt.assert_equal(
        cell_mat_all.toarray(),
        np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 4, 4],
                [0] * 5,
                [1, 4, 0, 1, 1],
                [1, 4, 0, 1, 1],
            ]
        ),
    )


def test_tcr_dist(adata_cdr3):
    for metric in ["alignment", "identity", "levenshtein"]:
        tra_dists, trb_dists = st.pp.tcr_dist(adata_cdr3, metric=metric)
        assert len(tra_dists) == len(trb_dists) == 4
        for tra_dist, trb_dist in zip(tra_dists, trb_dists):
            assert (
                tra_dist.shape == trb_dist.shape == (adata_cdr3.n_obs, adata_cdr3.n_obs)
            )


def test_reduce_dists():
    A = scipy.sparse.csr_matrix(
        [[0, 1, 3, 0], [1, 0, 0, 0], [0, 7, 0, 8], [0, 0, 0, 0]]
    )
    B = scipy.sparse.csr_matrix(
        [[0, 2, 0, 0], [0, 0, 0, 0], [0, 6, 0, 9], [0, 1, 0, 0]]
    )
    expected_all = np.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 7, 0, 9], [0, 0, 0, 0]])
    expected_any = np.array([[0, 1, 3, 0], [1, 0, 0, 0], [0, 6, 0, 8], [0, 1, 0, 0]])
    A.eliminate_zeros()
    B.eliminate_zeros()

    npt.assert_equal(_reduce_dists(A, B, "TRA").toarray(), A.toarray())
    npt.assert_equal(_reduce_dists(A, B, "TRB").toarray(), B.toarray())
    npt.assert_equal(_reduce_dists(A, B, "all").toarray(), expected_all)
    npt.assert_equal(_reduce_dists(A, B, "any").toarray(), expected_any)


def test_dist_to_connectivities():
    D = scipy.sparse.csr_matrix(
        [[0, 1, 1, 5], [0, 0, 2, 8], [1, 5, 0, 2], [10, 0, 0, 0]]
    )
    C = _dist_to_connectivities(D, 10)
    assert C.nnz == D.nnz
    npt.assert_equal(
        C.toarray(),
        np.array([[0, 1, 1, 0.6], [0, 0, 0.9, 0.3], [1, 0.6, 0, 0.9], [0.1, 0, 0, 0]]),
    )

    D = scipy.sparse.csr_matrix(
        [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    )
    C = _dist_to_connectivities(D, 0)
    assert C.nnz == D.nnz
    npt.assert_equal(
        C.toarray(), D.toarray(),
    )


def test_tcr_neighbors(adata_cdr3):
    conn, dist = st.pp.tcr_neighbors(adata_cdr3, inplace=False)
    assert conn.shape == dist.shape == (5, 5)

    st.pp.tcr_neighbors(
        adata_cdr3,
        metric="levenshtein",
        cutoff=3,
        strategy="TRA",
        chains="all",
        key_added="nbs",
    )
    assert adata_cdr3.uns["nbs"]["connectivities"].shape == (5, 5)
    assert adata_cdr3.uns["nbs"]["distances"].shape == (5, 5)
