import pytest
from scirpy.ir_dist.metrics import DistanceCalculator
from scirpy.ir_dist._clonotype_neighbors import ClonotypeNeighbors
import numpy as np
import numpy.testing as npt
import scirpy as ir
import scipy.sparse
from .fixtures import adata_cdr3, adata_cdr3_2  # NOQA
from .util import _squarify
from scirpy.util import _is_symmetric
import pandas.testing as pdt
import pandas as pd


def _assert_frame_equal(left, right):
    """alias to pandas.testing.assert_frame_equal configured for the tests in this file"""
    pdt.assert_frame_equal(left, right, check_dtype=False, check_categorical=False)


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


@pytest.mark.parametrize(
    "metric,expected_key,expected_dist_vj,expected_dist_vdj",
    [
        (
            "identity",
            "ir_dist_aa_identity",
            np.identity(2),
            np.identity(5),
        ),
        (
            "mock",
            "ir_dist_aa_custom",
            np.array([[1, 4], [4, 1]]),
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 10, 10, 0],
                    [0, 10, 1, 5, 0],
                    [0, 10, 5, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_ir_dist(
    adata_cdr3,
    adata_cdr3_mock_distance_calculator,
    metric,
    expected_key,
    expected_dist_vj,
    expected_dist_vdj,
):
    expected_seq_vj = np.array(["AAA", "AHA"])
    expected_seq_vdj = np.array(["AAA", "KK", "KKK", "KKY", "LLL"])
    ir.pp.ir_dist(
        adata_cdr3,
        metric=metric if metric != "mock" else adata_cdr3_mock_distance_calculator,
        sequence="aa",
    )
    res = adata_cdr3.uns[expected_key]
    npt.assert_array_equal(res["VJ"]["seqs"], expected_seq_vj)
    npt.assert_array_equal(res["VDJ"]["seqs"], expected_seq_vdj)
    npt.assert_array_equal(res["VJ"]["distances"].toarray(), expected_dist_vj)
    npt.assert_array_equal(res["VDJ"]["distances"].toarray(), expected_dist_vdj)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances1(adata_cdr3, n_jobs):
    # test single chain with identity distance
    ir.pp.ir_dist(adata_cdr3, metric="identity", cutoff=0, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_identity",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes, pd.DataFrame({"IR_VJ_1_junction_aa": ["AAA", "AHA", "nan"]})
    )
    dist = cn.compute_distances()
    npt.assert_equal(
        dist.toarray(),
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances2(adata_cdr3, n_jobs):
    # test single receptor arm with multiple chains and identity distance
    ir.pp.ir_dist(adata_cdr3, metric="identity", cutoff=0, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="any",
        distance_key="ir_dist_aa_identity",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances3(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    # test single chain with custom distance
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes, pd.DataFrame({"IR_VJ_1_junction_aa": ["AAA", "AHA", "nan"]})
    )
    dist = cn.compute_distances()
    assert dist.nnz == 4
    dist = dist.toarray()
    print(dist)
    npt.assert_equal(
        dist,
        np.array(
            [
                [1, 4, 0],
                [4, 1, 0],
                [0, 0, 0],
            ]
        ),
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances4(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    # test single receptor arm with multiple chains and custom distance
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="any",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances()
    assert dist.nnz == 16
    npt.assert_equal(
        dist.toarray(),
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances5(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    # test single receptor arm with multiple chains and custom distance
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="VJ",
        dual_ir="all",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances6(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    # test both receptor arms, primary chain only
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="all",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
        np.array(
            [
                [1, 10, 0, 0],
                [10, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("dual_ir", ["all", "any", "primary_only"])
@pytest.mark.parametrize(
    "receptor_arms,expected",
    [
        (
            "all",
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            "any",
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            ),
        ),
        ("VJ", np.array([[1, 0], [0, 0]])),
        ("VDJ", np.array([[1, 0], [0, 1]])),
    ],
)
def test_compute_distances6_2(
    adata_cdr3_2,
    adata_cdr3_mock_distance_calculator,
    receptor_arms,
    dual_ir,
    expected,
    n_jobs,
):
    """Test that `dual_ir` does not impact the second reduction step"""
    ir.pp.ir_dist(
        adata_cdr3_2, metric=adata_cdr3_mock_distance_calculator, sequence="aa"
    )
    cn = ClonotypeNeighbors(
        adata_cdr3_2,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(dist, expected)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances7(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="any",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
        np.array(
            [
                [1, 4, 0, 1],
                [4, 1, 0, 4],
                [0, 0, 0, 0],
                [1, 4, 0, 1],
            ]
        ),
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances8(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="any",
        dual_ir="all",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL", "LLL"],
                "IR_VDJ_2_junction_aa": ["KKK", "KKK", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances9(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="any",
        dual_ir="any",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL", "LLL"],
                "IR_VDJ_2_junction_aa": ["KKK", "KKK", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances()
    npt.assert_equal(
        dist.toarray(),
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances10(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="all",
        dual_ir="any",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL", "LLL"],
                "IR_VDJ_2_junction_aa": ["KKK", "KKK", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_distances11(adata_cdr3, adata_cdr3_mock_distance_calculator, n_jobs):
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="all",
        dual_ir="all",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
        n_jobs=n_jobs,
        chunksize=1,
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["AAA", "AHA", "nan", "AAA", "AAA"],
                "IR_VJ_2_junction_aa": ["AHA", "nan", "nan", "AAA", "nan"],
                "IR_VDJ_1_junction_aa": ["KKY", "KK", "nan", "LLL", "LLL"],
                "IR_VDJ_2_junction_aa": ["KKK", "KKK", "nan", "AAA", "nan"],
            }
        ),
    )
    dist = cn.compute_distances().toarray()
    print(dist)
    npt.assert_equal(
        dist,
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
    """Test for #174. Gracefully handle the case when there are no distances."""
    adata_cdr3.obs["IR_VJ_1_junction_aa"] = np.nan
    adata_cdr3.obs["IR_VDJ_1_junction_aa"] = np.nan
    # test both receptor arms, primary chain only
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    cn = ClonotypeNeighbors(
        adata_cdr3,
        receptor_arms="all",
        dual_ir="primary_only",
        distance_key="ir_dist_aa_custom",
        sequence_key="junction_aa",
    )
    _assert_frame_equal(
        cn.clonotypes,
        pd.DataFrame(
            {
                "IR_VJ_1_junction_aa": ["nan"],
                "IR_VDJ_1_junction_aa": ["nan"],
            }
        ),
    )
    dist = cn.compute_distances()
    npt.assert_equal(dist.toarray(), np.zeros((1, 1)))


def test_compute_distances13(adata_cdr3, adata_cdr3_mock_distance_calculator):
    """Test for #174. Gracefully handle the case when there are IR."""
    adata_cdr3.obs["IR_VJ_1_junction_aa"] = np.nan
    adata_cdr3.obs["IR_VDJ_1_junction_aa"] = np.nan
    adata_cdr3.obs["has_ir"] = "False"
    # test both receptor arms, primary chain only
    ir.pp.ir_dist(adata_cdr3, metric=adata_cdr3_mock_distance_calculator, sequence="aa")
    with pytest.raises(ValueError):
        cn = ClonotypeNeighbors(
            adata_cdr3,
            receptor_arms="all",
            dual_ir="primary_only",
            distance_key="ir_dist_aa_custom",
            sequence_key="junction_aa",
        )
