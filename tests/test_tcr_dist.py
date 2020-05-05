import pytest
from scirpy.tcr_dist import (
    AlignmentDistanceCalculator,
    DistanceCalculator,
    IdentityDistanceCalculator,
    LevenshteinDistanceCalculator,
    TcrNeighbors,
)
import numpy as np
import numpy.testing as npt
import scirpy as st
import scipy.sparse
from .fixtures import adata_cdr3
from anndata import AnnData


@pytest.fixture
def adata_cdr3_mock_distance_calculator():
    class MockDistanceCalculator(DistanceCalculator):
        def __init__(self, n_jobs=None):
            pass

        def calc_dist_mat(self, seqs):
            """Don't calculate distances, but return the
            hard-coded distance matrix needed for the test"""
            mat_seqs = np.array(["AAA", "AHA", "KK", "KKK", "KKY", "LLL"])
            mask = np.isin(mat_seqs, seqs)
            return scipy.sparse.coo_matrix(
                np.array(
                    [
                        [1, 4, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 10, 10, 0],
                        [0, 0, 0, 1, 5, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )[mask, :][:, mask]
            )

    return MockDistanceCalculator()


def test_identity_dist():
    identity = IdentityDistanceCalculator()
    npt.assert_almost_equal(
        identity.calc_dist_mat(["ARS", "ARS", "RSA"]).toarray(),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


def test_levenshtein_compute_row():
    levenshtein1 = LevenshteinDistanceCalculator(1)
    seqs = np.array(["A", "AAA", "AA"])
    row0 = levenshtein1._compute_row(seqs, 0)
    row2 = levenshtein1._compute_row(seqs, 2)
    assert row0.getnnz() == 2
    assert row2.getnnz() == 1
    npt.assert_equal(row0.toarray(), [[1, 0, 2]])
    npt.assert_equal(row2.toarray(), [[0, 0, 1]])


def test_levensthein_dist():
    levenshtein10 = LevenshteinDistanceCalculator(10)
    levenshtein1 = LevenshteinDistanceCalculator(1, n_jobs=1)
    npt.assert_almost_equal(
        levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"])).toarray(),
        np.array([[1, 2, 3, 3], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )
    npt.assert_almost_equal(
        levenshtein1.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"])).toarray(),
        np.array([[1, 2, 0, 0], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )


def test_align_row():
    aligner = AlignmentDistanceCalculator(cutoff=255)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
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
        AlignmentDistanceCalculator(3000)
    aligner = AlignmentDistanceCalculator(cutoff=255, n_jobs=1)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    npt.assert_almost_equal(
        res.toarray(), np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]])
    )

    res = aligner10.calc_dist_mat(seqs)
    npt.assert_almost_equal(res.toarray(), np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]]))


def test_tcr_dist(adata_cdr3):
    for metric in ["alignment", "identity", "levenshtein"]:
        unique_seqs = np.array(["AAA", "ARA", "AFFFFFA", "FAFAFA", "FFF"])
        dist_mat = st.tcr_dist.tcr_dist(unique_seqs, metric=metric, cutoff=8, n_jobs=2)
        assert dist_mat.shape == (5, 5)


def test_seq_to_cell_idx():
    unique_seqs = np.array(["AAA", "ABA", "CCC", "XXX", "AA"])
    cdr_seqs = np.array(["AAA", "CCC", "ABA", "CCC", np.nan, "AA", "AA"])
    result = TcrNeighbors._seq_to_cell_idx(unique_seqs, cdr_seqs)
    assert result == {0: [0], 1: [2], 2: [1, 3], 3: [], 4: [5, 6]}


def test_build_index_dict(adata_cdr3):
    tn = TcrNeighbors(
        adata_cdr3, receptor_arms="TRA", dual_tcr="primary_only", sequence="nt"
    )
    tn._build_index_dict()
    npt.assert_equal(
        tn.index_dict,
        {
            "TRA": {
                "chain_inds": [1],
                "unique_seqs": ["GCGAUGGCG", "GCGGCGGCG", "GCUGCUGCU"],
                "seq_to_cell": {1: {0: [1], 1: [0], 2: [3]}},
            }
        },
    )

    tn = TcrNeighbors(adata_cdr3, receptor_arms="all", dual_tcr="all", sequence="aa")
    tn._build_index_dict()
    print(tn.index_dict)
    npt.assert_equal(
        tn.index_dict,
        {
            "TRA": {
                "chain_inds": [1, 2],
                "unique_seqs": ["AAA", "AHA"],
                "seq_to_cell": {1: {0: [0, 3], 1: [1]}, 2: {0: [3, 4], 1: [0]},},
                "chains_per_cell": np.array([2, 1, 0, 2, 1]),
            },
            "TRB": {
                "chain_inds": [1, 2],
                "unique_seqs": ["AAA", "KK", "KKK", "KKY", "LLL"],
                "seq_to_cell": {
                    1: {0: [], 1: [1], 2: [], 3: [0], 4: [3, 4]},
                    2: {0: [3], 1: [], 2: [0, 1], 3: [], 4: []},
                },
                "chains_per_cell": np.array([2, 2, 0, 2, 1]),
            },
        },
    )

    tn2 = TcrNeighbors(adata_cdr3, receptor_arms="any", dual_tcr="any", sequence="aa")
    tn2._build_index_dict()
    print(tn2.index_dict)
    npt.assert_equal(
        tn2.index_dict,
        {
            "TRA": {
                "chain_inds": [1, 2],
                "unique_seqs": ["AAA", "AHA"],
                "seq_to_cell": {1: {0: [0, 3], 1: [1]}, 2: {0: [3, 4], 1: [0]},},
            },
            "TRB": {
                "chain_inds": [1, 2],
                "unique_seqs": ["AAA", "KK", "KKK", "KKY", "LLL"],
                "seq_to_cell": {
                    1: {0: [], 1: [1], 2: [], 3: [0], 4: [3, 4]},
                    2: {0: [3], 1: [], 2: [0, 1], 3: [], 4: []},
                },
            },
        },
    )


def test_compute_distances1(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single chain with identity distance
    tn = TcrNeighbors(
        adata_cdr3,
        metric="identity",
        cutoff=0,
        receptor_arms="TRA",
        dual_tcr="primary_only",
        sequence="aa",
    )
    tn.compute_distances()
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0] * 5, [1, 0, 0, 1, 0], [0] * 5,]
        ),
    )


def test_compute_distances2(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single receptor arm with multiple chains and identity distance
    tn = TcrNeighbors(
        adata_cdr3, metric="identity", cutoff=0, receptor_arms="TRA", dual_tcr="any",
    )
    tn.compute_distances()
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0],
                [0] * 5,
                [1, 0, 0, 1, 1],
                [1, 0, 0, 1, 1],
            ]
        ),
    )


def test_compute_distances3(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single chain with custom distance
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="TRA",
        dual_tcr="primary_only",
    )
    tn.compute_distances()
    assert tn.dist.nnz == 9
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [[1, 4, 0, 1, 0], [4, 1, 0, 4, 0], [0] * 5, [1, 4, 0, 1, 0], [0] * 5,]
        ),
    )


def test_compute_distances4(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single receptor arm with multiple chains and custom distance
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="TRA",
        dual_tcr="any",
    )
    tn.compute_distances()

    assert tn.dist.nnz == 16
    npt.assert_equal(
        tn.dist.toarray(),
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


def test_compute_distances5(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single receptor arm with multiple chains and custom distance
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="TRA",
        dual_tcr="all",
    )
    tn.compute_distances()

    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 0, 0, 4, 0],
                [0, 1, 0, 0, 4],
                [0, 0, 0, 0, 0],
                [4, 0, 0, 1, 0],
                [0, 4, 0, 0, 1],
            ]
        ),
    )


def test_compute_distances6(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test both receptor arms, primary chain only
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_tcr="primary_only",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 13, 0, 0, 0],
                [13, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )


def test_compute_distances7(adata_cdr3, adata_cdr3_mock_distance_calculator):
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_tcr="primary_only",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 4, 0, 1, 0],
                [4, 1, 0, 4, 0],
                [0, 0, 0, 0, 0],
                [1, 4, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ),
    )


def test_compute_distances8(adata_cdr3, adata_cdr3_mock_distance_calculator):
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_tcr="all",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 10, 0, 4, 0],
                [10, 1, 0, 0, 4],
                [0, 0, 0, 0, 0],
                [4, 0, 0, 1, 0],
                [0, 4, 0, 0, 1],
            ]
        ),
    )


def test_compute_distances9(adata_cdr3, adata_cdr3_mock_distance_calculator):
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_tcr="any",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 1, 0, 1, 1],
                [1, 1, 0, 4, 4],
                [0, 0, 0, 0, 0],
                [1, 4, 0, 1, 1],
                [1, 4, 0, 1, 1],
            ]
        ),
    )


def test_compute_distances10(adata_cdr3, adata_cdr3_mock_distance_calculator):
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_tcr="any",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ),
    )


def test_compute_distances11(adata_cdr3, adata_cdr3_mock_distance_calculator):
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_tcr="all",
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )


def test_dist_to_connectivities(adata_cdr3):
    # empty anndata, just need the object
    tn = TcrNeighbors(adata_cdr3, metric="alignment", cutoff=10)
    tn._dist_mat = scipy.sparse.csr_matrix(
        [[0, 1, 1, 5], [0, 0, 2, 8], [1, 5, 0, 2], [10, 0, 0, 0]]
    )
    C = tn.connectivities
    assert C.nnz == tn._dist_mat.nnz
    npt.assert_equal(
        C.toarray(),
        np.array([[0, 1, 1, 0.6], [0, 0, 0.9, 0.3], [1, 0.6, 0, 0.9], [0.1, 0, 0, 0]]),
    )

    tn2 = TcrNeighbors(adata_cdr3, metric="identity", cutoff=0)
    tn2._dist_mat = scipy.sparse.csr_matrix(
        [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    )
    C = tn2.connectivities
    assert C.nnz == tn2._dist_mat.nnz
    npt.assert_equal(
        C.toarray(), tn2._dist_mat.toarray(),
    )


def test_tcr_neighbors(adata_cdr3):
    conn, dist = st.pp.tcr_neighbors(adata_cdr3, inplace=False)
    assert conn.shape == dist.shape == (5, 5)

    st.pp.tcr_neighbors(
        adata_cdr3,
        metric="levenshtein",
        cutoff=3,
        receptor_arms="TRA",
        dual_tcr="all",
        key_added="nbs",
    )
    assert adata_cdr3.uns["nbs"]["connectivities"].shape == (5, 5)
    assert adata_cdr3.uns["nbs"]["distances"].shape == (5, 5)
