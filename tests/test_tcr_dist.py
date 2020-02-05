import pytest
from sctcrpy._tools._tcr_dist import (
    _Aligner,
    _calc_norm_factors,
    _score_to_dist,
    tcr_dist,
)
import numpy as np
import numpy.testing as npt


@pytest.fixture
def aligner():
    return _Aligner()


def test_align_row(aligner):
    seqs = np.array(["AWAW", "VWVW", "HHHH"])
    row0 = aligner._align_row(seqs, 0)
    row2 = aligner._align_row(seqs, 2)
    npt.assert_equal(row0, [2 * 4 + 2 * 11, 2 * 0 + 2 * 11, 2 * -2 + 2 * -2])
    npt.assert_equal(row2, [np.nan, np.nan, 4 * 8])


def test_make_score_mat(aligner):
    seqs = np.array(["AAAA", "HHHH"])
    res = aligner.make_score_mat(seqs)
    npt.assert_equal(res, np.array([[4 * 4, 4 * -2], [4 * -2, 4 * 8]]))


def test_calc_norm_factors():
    score_mat = np.array([[15, 9, -2], [9, 11, 3], [-2, 3, 5]])
    norm_factors = _calc_norm_factors(score_mat)
    npt.assert_equal(norm_factors, np.array([[15, 11, 5], [11, 11, 5], [5, 5, 5]]))


def test_score_to_dist():
    with pytest.raises(AssertionError):
        # Violates assumption that max value is on diagonal
        score_mat = np.array([[15, 12, -2], [12, 11, 3], [-2, 3, 5]])
        _score_to_dist(score_mat)

    score_mat = np.array([[15, 9, -2], [9, 11, 3], [-2, 3, 5]])
    dist_mat = _score_to_dist(score_mat)
    npt.assert_almost_equal(
        dist_mat, np.array([[0, 0.181, 1], [0.181, 0, 0.4], [1, 0.4, 0]]), decimal=2
    )
