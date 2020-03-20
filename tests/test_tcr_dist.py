import pytest
from sctcrpy._tools._tcr_dist import (
    _AlignmentDistanceCalculator,
    _DistanceCalculator,
    _IdentityDistanceCalculator,
    _LevenshteinDistanceCalculator,
    _dist_for_chain,
    tcr_dist,
)
import numpy as np
import pandas as pd
import numpy.testing as npt
from anndata import AnnData
import sctcrpy as st


@pytest.fixture
def aligner():
    return _AlignmentDistanceCalculator()


@pytest.fixture
def identity():
    return _IdentityDistanceCalculator()


@pytest.fixture
def levenshtein():
    return _LevenshteinDistanceCalculator()


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


def test_levensthein_dist(levenshtein):
    npt.assert_almost_equal(
        levenshtein.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"])),
        np.array([[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 1], [2, 1, 1, 0]]),
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
        dist_mat, np.array([[0, 2, 7], [2, 0, 2], [7, 2, 0]]), decimal=2
    )


def test_alignment_score(aligner):
    seqs = np.array(["AAAA", "HHHH"])
    res = aligner._calc_score_mat(seqs)
    npt.assert_equal(res, np.array([[4 * 4, 4 * -2], [4 * -2, 4 * 8]]))


def test_alignment_dist(aligner):
    seqs = np.array(["AAAA", "AAHA"])
    res = aligner.calc_dist_mat(seqs)
    npt.assert_almost_equal(res, np.array([[0, 6], [6, 0]]))


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


def test_define_clonotypes_no_graph():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "AAAA", "nan", "nan", "nan"],
            ["cell2", "nan", "nan", "nan", "nan"],
            ["cell3", "AAAA", "nan", "nan", "nan"],
            ["cell4", "AAAA", "BBBB", "nan", "nan"],
            ["cell5", "nan", "nan", "CCCC", "DDDD"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3", "TRB_1_cdr3", "TRB_2_cdr3"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    res = st.tl._define_clonotypes_no_graph(adata, inplace=False)
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res,
        ["clonotype_1", np.nan, "clonotype_1", "clonotype_0", "clonotype_2"],
    )

    res_primary_only = st.tl._define_clonotypes_no_graph(
        adata, flavor="primary_only", inplace=False
    )
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res_primary_only,
        ["clonotype_0", np.nan, "clonotype_0", "clonotype_0", "clonotype_1"],
    )

    # test inplace
    st.tl._define_clonotypes_no_graph(adata, key_added="clonotype_")
    npt.assert_equal(res, adata.obs["clonotype_"].values)
