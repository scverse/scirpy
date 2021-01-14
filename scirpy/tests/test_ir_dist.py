import pytest
from scirpy.ir_dist.metrics import DistanceCalculator
from scirpy.ir_dist._clonotype_neighbors import ClonotypeNeighbors
import numpy as np
import numpy.testing as npt
import scirpy as ir
import scipy.sparse
from .fixtures import adata_cdr3
from .util import _squarify
from scirpy.util import _is_symmetric


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
            dist_mat = scipy.sparse.csr_matrix(
                np.array(
                    [
                        [1, 4, 0, 0, 0, 0],
                        [4, 1, 0, 0, 0, 0],
                        [0, 0, 1, 10, 10, 0],
                        [0, 0, 10, 1, 5, 0],
                        [0, 0, 10, 5, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )[mask, :][:, mask]
            )
            assert _is_symmetric(dist_mat)
            return dist_mat

    return MockDistanceCalculator()


@pytest.mark.parametrize(
    "metric,cutoff,expected_square,expected",
    [
        (
            "identity",
            0,
            _squarify(
                [
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ),
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ),
        (
            "levenshtein",
            10,
            _squarify(
                [
                    [1, 2, 3, 3, 1],
                    [0, 1, 3, 3, 2],
                    [0, 0, 1, 1, 3],
                    [0, 0, 0, 1, 3],
                    [0, 0, 0, 0, 1],
                ]
            ),
            [
                [2, 3, 1, 3],
                [1, 4, 2, 4],
                [3, 4, 3, 4],
                [3, 4, 3, 4],
                [2, 3, 1, 3],
            ],
        ),
    ],
)
def test_sequence_dist(metric, cutoff, expected_square, expected):
    seqs1 = np.array(["AAA", "ARA", "FFA", "FFA", "AAA"])
    seqs2 = np.array(["ARA", "CAC", "AAA", "CAC"])

    res = ir.ir_dist.sequence_dist(seqs1, metric=metric, cutoff=cutoff)
    npt.assert_almost_equal(res.toarray(), np.array(expected_square))

    res = ir.ir_dist.sequence_dist(seqs1, seqs2, metric=metric, cutoff=cutoff)
    res_t = ir.ir_dist.sequence_dist(seqs2, seqs1, metric=metric, cutoff=cutoff)
    npt.assert_almost_equal(
        res.toarray(),
        np.array(expected),
    )
    npt.assert_almost_equal(
        res_t.toarray(),
        np.array(expected).T,
    )


def test_compute_distances1(adata_cdr3):
    # test single chain with identity distance
    ir.pp.ir_dist(adata_cdr3, metric="identity", cutoff=0, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_identity",
        sequence_key="cdr3",
    )
    dist = cn.compute_distances()
    npt.assert_equal(
        dist.toarray(),
        np.array(
            [
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0] * 5,
                [1, 0, 0, 1, 0],
                [0] * 5,
            ]
        ),
    )


def test_compute_distances2(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single receptor arm with multiple chains and identity distance
    cn = IrNeighbors(
        adata_cdr3,
        metric="identity",
        cutoff=0,
        receptor_arms="VJ",
        dual_ir="any",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="VJ",
        dual_ir="primary_only",
        sequence="aa",
    )
    cn.compute_distances()
    assert tn.dist.nnz == 9
    npt.assert_equal(
        tn.dist.toarray(),
        np.array(
            [
                [1, 4, 0, 1, 0],
                [4, 1, 0, 4, 0],
                [0] * 5,
                [1, 4, 0, 1, 0],
                [0] * 5,
            ]
        ),
    )


def test_compute_distances4(adata_cdr3, adata_cdr3_mock_distance_calculator):
    # test single receptor arm with multiple chains and custom distance
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="VJ",
        dual_ir="any",
        sequence="aa",
    )
    cn.compute_distances()

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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="VJ",
        dual_ir="all",
        sequence="aa",
    )
    cn.compute_distances()

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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_ir="primary_only",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_ir="primary_only",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_ir="all",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="any",
        dual_ir="any",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_ir="any",
        sequence="aa",
    )
    cn.compute_distances()
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
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_ir="all",
        sequence="aa",
    )
    cn.compute_distances()
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
    adata_cdr3.obs["IR_VJ_1_cdr3"] = np.nan
    adata_cdr3.obs["IR_VDJ_1_cdr3"] = np.nan
    # test both receptor arms, primary chain only
    cn = IrNeighbors(
        adata_cdr3,
        metric=adata_cdr3_mock_distance_calculator,
        receptor_arms="all",
        dual_ir="primary_only",
        sequence="aa",
        cutoff=0,
    )
    cn.compute_distances()
    print(tn.dist.toarray())
    npt.assert_equal(tn.dist.toarray(), np.zeros((5, 5)))


# TODO move to util tests
# def test_dist_to_connectivities(adata_cdr3):
#     # empty anndata, just need the object
#     cn = IrNeighbors(adata_cdr3, metric="alignment", cutoff=10)
#     tn._dist_mat = scipy.sparse.csr_matrix(
#         [[0, 1, 1, 5], [0, 0, 2, 8], [1, 5, 0, 2], [10, 0, 0, 0]]
#     )
#     C = tn.connectivities
#     assert C.nnz == tn._dist_mat.nnz
#     npt.assert_equal(
#         C.toarray(),
#         np.array([[0, 1, 1, 0.6], [0, 0, 0.9, 0.3], [1, 0.6, 0, 0.9], [0.1, 0, 0, 0]]),
#     )

#     tn2 = IrNeighbors(adata_cdr3, metric="identity", cutoff=0)
#     tn2._dist_mat = scipy.sparse.csr_matrix(
#         [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
#     )
#     C = tn2.connectivities
#     assert C.nnz == tn2._dist_mat.nnz
#     npt.assert_equal(
#         C.toarray(),
#         tn2._dist_mat.toarray(),
#     )
