import pytest
from sctcrpy._tools._tcr_dist import (
    _AlignmentDistanceCalculator,
    _KideraDistanceCalculator,
    tcr_dist,
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
def adata_cdr3():
    obs = pd.DataFrame(
        [
            ["cell1", "AAAAA", "WWWWW"],
            ["cell2", "AAAVV", "WWWYY"],
            ["cell3", "HHAHH", "PPWPP"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


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
        kidera._make_kidera_vectors(["AAA", "RRR"]), 3 * AR_KIDERA_VECTORS
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


def test_alignment_dist(aligner):
    seqs = np.array(["AAAA", "HHHH"])
    res = aligner.calc_dist_mat(seqs)
    npt.assert_equal(res, np.array([[4 * 4, 4 * -2], [4 * -2, 4 * 8]]))


def test_dist_for_chain():
    assert False


def test_tcr_dist(adata_cdr3):
    assert False
