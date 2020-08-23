import pytest
from scirpy.tcr_dist import (
    AlignmentDistanceCalculator,
    DistanceCalculator,
    IdentityDistanceCalculator,
    LevenshteinDistanceCalculator,
    ParallelDistanceCalculator,
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

        def calc_dist_mat(self, seqs, seqs2=None):
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


def test_block_iter():
    seqs1 = list("ABCDE")
    seqs2 = list("HIJKLM")
    b1 = list(ParallelDistanceCalculator._block_iter(seqs1, block_size=2))
    b2 = list(ParallelDistanceCalculator._block_iter(seqs1, seqs2, block_size=3))
    b3 = list(ParallelDistanceCalculator._block_iter(seqs1, seqs2, block_size=50))
    b4 = list(
        ParallelDistanceCalculator._block_iter(list("ABC"), list("ABC"), block_size=1)
    )
    L = list
    assert b1 == [
        (L("AB"), None, (0, 0)),
        (L("AB"), L("CD"), (0, 2)),
        (L("AB"), L("E"), (0, 4)),
        (L("CD"), None, (2, 2)),
        (L("CD"), L("E"), (2, 4)),
        (L("E"), None, (4, 4)),
    ]
    assert b2 == [
        (L("ABC"), L("HIJ"), (0, 0)),
        (L("ABC"), L("KLM"), (0, 3)),
        (L("DE"), L("HIJ"), (3, 0)),
        (L("DE"), L("KLM"), (3, 3)),
    ]
    assert b3 == [(L("ABCDE"), L("HIJKLM"), (0, 0))]
    assert b4 == [
        (L("A"), L("A"), (0, 0)),
        (L("A"), L("B"), (0, 1)),
        (L("A"), L("C"), (0, 2)),
        (L("B"), L("A"), (1, 0)),
        (L("B"), L("B"), (1, 1)),
        (L("B"), L("C"), (1, 2)),
        (L("C"), L("A"), (2, 0)),
        (L("C"), L("B"), (2, 1)),
        (L("C"), L("C"), (2, 2)),
    ]


def test_identity_dist():
    identity = IdentityDistanceCalculator()
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA"])
    assert isinstance(res, scipy.sparse.coo_matrix)
    npt.assert_almost_equal(
        res.toarray(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


def test_identity_dist_with_two_seq_arrays():
    identity = IdentityDistanceCalculator()
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA", "SAS"], ["KKL", "ARS", "RSA"])
    assert isinstance(res, scipy.sparse.coo_matrix)
    assert res.shape == (4, 3)
    assert res.nnz == 3
    npt.assert_almost_equal(
        res.toarray(), np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    )

    # test with completely empty result matrix
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA", "SAS"], ["foo", "bar", "baz"])
    assert isinstance(res, scipy.sparse.coo_matrix)
    assert res.shape == (4, 3)
    assert res.nnz == 0
    npt.assert_almost_equal(
        res.toarray(), np.zeros((4, 3)),
    )


def test_levenshtein_compute_block():
    levenshtein1 = LevenshteinDistanceCalculator(1)
    seqs = np.array(["A", "AAA", "AA"])
    seqs2 = np.array(["AB", "BAA"])
    b1 = list(levenshtein1._compute_block(seqs, None, (10, 20)))
    b2 = list(levenshtein1._compute_block(seqs, seqs, (10, 20)))
    b3 = list(levenshtein1._compute_block(seqs, seqs2, (10, 20)))
    b4 = list(levenshtein1._compute_block(seqs2, seqs, (10, 20)))

    assert b1 == [(1, 10, 20), (2, 10, 22), (1, 11, 21), (2, 11, 22), (1, 12, 22)]
    assert b2 == [
        (1, 10, 20),
        (2, 10, 22),
        (1, 11, 21),
        (2, 11, 22),
        (2, 12, 20),
        (2, 12, 21),
        (1, 12, 22),
    ]
    assert b3 == [(2, 10, 20), (2, 11, 21), (2, 12, 20), (2, 12, 21)]
    assert b4 == [(2, 10, 20), (2, 10, 22), (2, 11, 21), (2, 11, 22)]


def test_levensthein_dist():
    levenshtein10 = LevenshteinDistanceCalculator(10, block_size=50)
    levenshtein10_2 = LevenshteinDistanceCalculator(10, block_size=2)
    levenshtein1 = LevenshteinDistanceCalculator(1, n_jobs=1, block_size=1)

    res10 = levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))
    res10_2 = levenshtein10_2.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))
    res1 = levenshtein1.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))

    assert isinstance(res10, scipy.sparse.coo_matrix)
    assert isinstance(res10_2, scipy.sparse.coo_matrix)
    assert isinstance(res1, scipy.sparse.coo_matrix)

    npt.assert_equal(res10.toarray(), res10_2.toarray())
    npt.assert_almost_equal(
        res10.toarray(),
        np.array([[1, 2, 3, 3], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )
    npt.assert_almost_equal(
        res1.toarray(),
        np.array([[1, 2, 0, 0], [0, 1, 2, 2], [0, 0, 1, 2], [0, 0, 0, 1]]),
    )


def test_levensthein_dist_with_two_seq_arrays():
    levenshtein10 = LevenshteinDistanceCalculator(2)
    res = levenshtein10.calc_dist_mat(
        np.array(["A", "AA", "AAA", "AAR", "ZZZZZZ"]), np.array(["RRR", "AR"])
    )
    assert isinstance(res, scipy.sparse.coo_matrix)
    assert res.shape == (5, 2)
    npt.assert_almost_equal(
        res.toarray(), np.array([[0, 2], [0, 2], [0, 3], [3, 2], [0, 0]]),
    )


def test_alignment_compute_block():
    aligner = AlignmentDistanceCalculator(cutoff=255)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
    seqs = ["AWAW", "VWVW", "HHHH"]

    b1 = aligner._compute_block(seqs, None, (0, 0))
    b2 = aligner10._compute_block(seqs, None, (10, 20))
    b3 = aligner10._compute_block(seqs, seqs, (10, 20))

    assert b1 == [(1, 0, 0), (9, 0, 1), (39, 0, 2), (1, 1, 1), (41, 1, 2), (1, 2, 2)]
    assert b2 == [(1, 10, 20), (9, 10, 21), (1, 11, 21), (1, 12, 22)]
    assert b3 == [(1, 10, 20), (9, 10, 21), (9, 11, 20), (1, 11, 21), (1, 12, 22)]


def test_alignment_dist():
    with pytest.raises(ValueError):
        AlignmentDistanceCalculator(3000)
    aligner = AlignmentDistanceCalculator(cutoff=255, n_jobs=1)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.coo_matrix)
    npt.assert_almost_equal(
        res.toarray(), np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]])
    )

    res = aligner10.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.coo_matrix)
    npt.assert_almost_equal(res.toarray(), np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]]))


def test_alignment_dist_with_two_seq_arrays():
    aligner = AlignmentDistanceCalculator(cutoff=10, n_jobs=1)
    res = aligner.calc_dist_mat(
        ["AAAA", "AATA", "HHHH", "WWWW"], ["WWWW", "AAAA", "ATAA"]
    )
    assert isinstance(res, scipy.sparse.coo_matrix)
    assert res.shape == (4, 3)

    npt.assert_almost_equal(
        res.toarray(), np.array([[0, 1, 5], [0, 5, 10], [0, 0, 0], [1, 0, 0]])
    )


@pytest.mark.parametrize("metric", ["alignment", "identity", "levenshtein"])
def test_tcr_dist(adata_cdr3, metric):
    unique_seqs = np.array(["AAA", "ARA", "AFFFFFA", "FAFAFA", "FFF"])
    seqs2 = np.array(["RRR", "FAFA", "WWWWWWW"])
    dist_mat = st.tcr_dist.tcr_dist(unique_seqs, metric=metric, cutoff=8, n_jobs=2)
    assert dist_mat.shape == (5, 5)

    dist_mat = st.tcr_dist.tcr_dist(
        unique_seqs, seqs2, metric=metric, cutoff=8, n_jobs=2
    )
    assert dist_mat.shape == (5, 3)


def test_seq_to_cell_idx():
    unique_seqs = np.array(["AAA", "ABA", "CCC", "XXX", "AA"])
    cdr_seqs = np.array(["AAA", "CCC", "ABA", "CCC", np.nan, "AA", "AA"])
    result = TcrNeighbors._seq_to_cell_idx(unique_seqs, cdr_seqs)
    assert result == {0: [0], 1: [2], 2: [1, 3], 3: [], 4: [5, 6]}


def test_build_index_dict(adata_cdr3):
    tn = TcrNeighbors(
        adata_cdr3,
        receptor_arms="TRA",
        dual_tcr="primary_only",
        sequence="nt",
        cutoff=0,
        metric="identity",
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

    tn = TcrNeighbors(
        adata_cdr3,
        receptor_arms="all",
        dual_tcr="all",
        sequence="aa",
        metric="identity",
        cutoff=0,
    )
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

    tn2 = TcrNeighbors(
        adata_cdr3,
        receptor_arms="any",
        dual_tcr="any",
        sequence="aa",
        metric="alignment",
        cutoff=10,
    )
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
        adata_cdr3,
        metric="identity",
        cutoff=0,
        receptor_arms="TRA",
        dual_tcr="any",
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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
        sequence="aa",
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


def test_compute_distances12(adata_cdr3, adata_cdr3_mock_distance_calculator):
    """Test for #174. Gracefully handle the case when there are no distances. """
    adata_cdr3.obs["TRA_1_cdr3"] = np.nan
    adata_cdr3.obs["TRB_1_cdr3"] = np.nan
    # test both receptor arms, primary chain only
    tn = TcrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_tcr="primary_only",
        sequence="aa",
        cutoff=0,
    )
    tn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(tn.dist.toarray(), np.zeros((5, 5)))


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
    conn, dist = st.pp.tcr_neighbors(adata_cdr3, inplace=False, sequence="aa")
    assert conn.shape == dist.shape == (5, 5)

    st.pp.tcr_neighbors(
        adata_cdr3,
        metric="levenshtein",
        cutoff=3,
        receptor_arms="TRA",
        dual_tcr="all",
        key_added="nbs",
        sequence="aa",
    )
    assert adata_cdr3.uns["nbs"]["connectivities"].shape == (5, 5)
    assert adata_cdr3.uns["nbs"]["distances"].shape == (5, 5)
