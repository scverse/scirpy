import pytest
from sctcrpy._tools._tcr_dist import (
    _AlignmentDistanceCalculator,
    _KideraDistanceCalculator,
    _DistanceCalculator,
    _IdentityDistanceCalculator,
    _dist_for_chain,
    tcr_neighbors,
)
import numpy as np
import pandas as pd
import numpy.testing as npt
from anndata import AnnData


@pytest.fixture
def aligner():
    return _AlignmentDistanceCalculator()


@pytest.fixture
def kidera():
    return _KideraDistanceCalculator()


@pytest.fixture
def identity():
    return _IdentityDistanceCalculator()


@pytest.fixture
def adata_cdr3():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "AHA"],
            ["cell2", "AHA", "nan"],
            ["cell3", "nan", "nan"],
            ["cell4", "AAA", "AAA"],
            ["cell5", "nan", "AAA"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_cdr3_mock_distance_calculator():
    class MockDistanceCalculator(_DistanceCalculator):
        def __init__(self, n_jobs=None):
            pass

        def calc_dist_mat(self, seqs):
            """Don't calculate distances, but return the
            dist matrix passed to the constructor."""
            npt.assert_equal(seqs, ["AAA", "AHA"])
            return np.array([[0, 1], [1, 0]])

    return MockDistanceCalculator()


def test_identity_dist(identity):
    npt.assert_almost_equal(
        identity.calc_dist_mat(["ARS", "ARS", "RSA"]),
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    )


def test_chain_dist_identity(adata_cdr3, identity):
    cell_mats = _dist_for_chain(adata_cdr3, "TRA", identity)
    tra1_tra1, tra1_tra2, tra2_tra1, tra2_tra2 = cell_mats

    npt.assert_equal(
        tra1_tra1,
        np.array(
            [
                [0, 1, np.nan, 0, np.nan],
                [1, 0, np.nan, 1, np.nan],
                [np.nan] * 5,
                [0, 1, np.nan, 0, np.nan],
                [np.nan] * 5,
            ]
        ),
    )


def test_kidera_vectors(kidera):
    AR_KIDERA_VECTORS = np.array(
        [
            [-1.56, 0.22],
            [-1.67, 1.27],
            [-0.97, 1.37],
            [-0.27, 1.87],
            [-0.93, -1.7],
            [-0.78, 0.46],
            [-0.2, 0.92],
            [-0.08, -0.39],
            [0.21, 0.23],
            [-0.48, 0.93],
        ]
    ).T
    npt.assert_almost_equal(kidera._make_kidera_vectors(["A", "R"]), AR_KIDERA_VECTORS)
    npt.assert_almost_equal(
        kidera._make_kidera_vectors(["AAA", "RRR"]), AR_KIDERA_VECTORS
    )


def test_kidera_dist(kidera):
    npt.assert_almost_equal(
        kidera.calc_dist_mat(["ARS", "ARS", "RSA", "SRA"]), np.zeros((4, 4))
    )


def test_align_row(aligner):
    seqs = np.array(["AWAW", "VWVW", "HHHH"])
    row0 = aligner._align_row(seqs, 0)
    row2 = aligner._align_row(seqs, 2)
    npt.assert_equal(row0, [2 * 4 + 2 * 11, 2 * 0 + 2 * 11, 2 * -2 + 2 * -2])
    npt.assert_equal(row2, [np.nan, np.nan, 4 * 8])


def test_calc_norm_factors(aligner):
    score_mat = np.array([[15, 9, -2], [9, 11, 3], [-2, 3, 5]])
    norm_factors = aligner._calc_norm_factors(score_mat)
    npt.assert_equal(norm_factors, np.array([[15, 11, 5], [11, 11, 5], [5, 5, 5]]))


def test_score_to_dist(aligner):
    with pytest.raises(AssertionError):
        # Violates assumption that max value is on diagonal
        score_mat = np.array([[15, 12, -2], [12, 11, 3], [-2, 3, 5]])
        aligner._score_to_dist(score_mat)

    score_mat = np.array([[15, 9, -2], [9, 11, 3], [-2, 3, 5]])
    dist_mat = aligner._score_to_dist(score_mat)
    npt.assert_almost_equal(
        dist_mat, np.array([[0, 0.181, 1], [0.181, 0, 0.4], [1, 0.4, 0]]), decimal=2
    )


def test_alignment_score(aligner):
    seqs = np.array(["AAAA", "HHHH"])
    res = aligner._calc_score_mat(seqs)
    npt.assert_equal(res, np.array([[4 * 4, 4 * -2], [4 * -2, 4 * 8]]))


def test_alignment_dist(aligner):
    seqs = np.array(["AAAA", "AAHA"])
    res = aligner.calc_dist_mat(seqs)
    npt.assert_almost_equal(res, np.array([[0, 1 - 10 / 16], [1 - 10 / 16, 0]]))


def test_dist_for_chain(adata_cdr3, adata_cdr3_mock_distance_calculator):
    """The _dist_for_chain function returns four matrices for 
    all combinations of tra1_tra1, tra1_tra2, tra2_tra1, tra2_tra2. 
    Tests if these matrices are correct. """
    cell_mats = _dist_for_chain(adata_cdr3, "TRA", adata_cdr3_mock_distance_calculator)
    tra1_tra1, tra1_tra2, tra2_tra1, tra2_tra2 = cell_mats

    npt.assert_equal(
        tra1_tra1,
        np.array(
            [
                [0, 1, np.nan, 0, np.nan],
                [1, 0, np.nan, 1, np.nan],
                [np.nan] * 5,
                [0, 1, np.nan, 0, np.nan],
                [np.nan] * 5,
            ]
        ),
    )

    npt.assert_equal(
        tra1_tra2,
        np.array(
            [
                [1, np.nan, np.nan, 0, 0],
                [0, np.nan, np.nan, 1, 1],
                [np.nan] * 5,
                [1, np.nan, np.nan, 0, 0],
                [np.nan] * 5,
            ]
        ),
    )

    npt.assert_equal(
        tra2_tra1,
        np.array(
            [
                [1, 0, np.nan, 1, np.nan],
                [np.nan] * 5,
                [np.nan] * 5,
                [0, 1, np.nan, 0, np.nan],
                [0, 1, np.nan, 0, np.nan],
            ]
        ),
    )

    npt.assert_equal(
        tra2_tra2,
        np.array(
            [
                [0, np.nan, np.nan, 1, 1],
                [np.nan] * 5,
                [np.nan] * 5,
                [1, np.nan, np.nan, 0, 0],
                [1, np.nan, np.nan, 0, 0],
            ]
        ),
    )

    npt.assert_equal(
        np.fmin.reduce(cell_mats),
        np.array(
            [
                [0, 0, np.nan, 0, 0],
                [0, 0, np.nan, 1, 1],
                [np.nan] * 5,
                [0, 1, np.nan, 0, 0],
                [0, 1, np.nan, 0, 0],
            ]
        ),
    )
