import warnings
from itertools import combinations
from typing import cast

import igraph as ig
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import scipy.sparse
from anndata import AnnData, read_h5ad
from mudata import MuData

import scirpy as ir
from scirpy.util import (
    DataHandler,
    _is_false,
    _is_na,
    _is_symmetric,
    _is_true,
    _normalize_counts,
    _translate_dna_to_protein,
)
from scirpy.util._negative_binomial import fit_nbinom
from scirpy.util.graph import (
    _distance_to_connectivity,
    _get_sparse_from_igraph,
    igraph_from_sparse_matrix,
    layout_components,
)

from . import TESTDATA


@pytest.mark.filterwarnings("ignore::anndata.OldFormatWarning")
def test_data_handler_upgrade_schema_pre_scirpy_v0_7():
    """Test that running a function on very old (pre v0.7) schema
    raises an error
    """
    adata = read_h5ad(TESTDATA / "wu2020_200_v0_6.h5ad")
    with pytest.raises(ValueError):
        DataHandler(adata, "airr", "airr")

    # Trying to run check upgrade schema raises an error
    with pytest.raises(ValueError):
        ir.io.upgrade_schema(adata)


def test_data_handler_upgrade_schema_pre_scirpy_v0_12():
    """Test that running a functon on an old anndata object raises an error
    Also test that the function can successfully be ran after calling `upgrade_schema`.
    """
    adata = read_h5ad(TESTDATA / "wu2020_200_v0_11.h5ad")
    with pytest.raises(ValueError):
        params = DataHandler(adata, "airr", "airr")

    ir.io.upgrade_schema(adata)
    params = DataHandler(adata, "airr", "airr")
    assert params.adata is adata


def test_data_handler_no_airr():
    """Test that a key error is raised if DataHandler is executed
    on an anndata without AirrData
    """
    adata = AnnData(np.ones((10, 10)))
    with pytest.raises(KeyError, match=r"No AIRR data found.*"):
        DataHandler(adata, "airr", "airr")


@pytest.fixture
def mdata_with_smaller_adata():
    adata_gex = AnnData(obs=pd.DataFrame(index=["c1", "c2", "c3"]).assign(both=[11, 12, 13]))
    adata_airr = AnnData(obs=pd.DataFrame(index=["c3", "c4", "c5"]).assign(both=[14, 15, 16]))
    mdata = MuData({"gex": adata_gex, "airr": adata_airr})
    mdata["airr"].obs["airr_only"] = [3, 4, 5]

    mdata.obs["mudata_only"] = [1, 2, 3, 4, 5]
    mdata.obs["both"] = [np.nan, np.nan, 114, 115, 116]
    return mdata


def test_data_handler_get_obs(mdata_with_smaller_adata):
    params = DataHandler(mdata_with_smaller_adata, "airr")
    # can retrieve value from mudata
    npt.assert_equal(params.get_obs("mudata_only").values, np.array([1, 2, 3, 4, 5]))
    # Mudata takes precedence
    npt.assert_equal(params.get_obs("both").values, np.array([np.nan, np.nan, 114, 115, 116]))
    # can retrieve value from anndata
    npt.assert_equal(params.get_obs("airr_only").values, np.array([3, 4, 5]))

    # generates dataframe if sequence is specified
    pdt.assert_frame_equal(
        params.get_obs(["mudata_only"]),
        pd.DataFrame(index=["c1", "c2", "c3", "c4", "c5"]).assign(mudata_only=[1, 2, 3, 4, 5]),
    )

    # multiple columns are concatenated into a dataframe
    pdt.assert_frame_equal(
        params.get_obs(["mudata_only", "both", "airr_only"]),
        pd.DataFrame(index=["c1", "c2", "c3", "c4", "c5"]).assign(
            mudata_only=[1, 2, 3, 4, 5],
            both=[np.nan, np.nan, 114, 115, 116],
            airr_only=[np.nan, np.nan, 3, 4, 5],
        ),
    )

    # only retreiving from the airr modality results in fewer rows
    pdt.assert_frame_equal(
        params.get_obs(["airr_only"]),
        pd.DataFrame(index=["c1", "c2", "c3", "c4", "c5"]).assign(airr_only=[np.nan, np.nan, 3, 4, 5]),
    )


@pytest.mark.parametrize(
    "value,exception",
    [
        [pd.Series([1, 2, 3], index=["c1", "c3", "c4"]), None],
        [pd.Series([1, 2, 3], index=["c1", "c3", "c8"]), None],
        [pd.Series([1, 2, 3, 4, 5], index=["c1", "c2", "c3", "c4", "c5"]), None],
        [[1, 2, 3], None],
        [[1, 2, 3, 4], ValueError],
        [[1, 2, 3, 4, 5], None],
        [[1, 2, 3, 4, 5, 6], ValueError],
    ],
)
def test_data_handler_set_obs(mdata_with_smaller_adata, value, exception):
    params = DataHandler(mdata_with_smaller_adata, "airr")
    if exception is not None:
        with pytest.raises(exception):
            params.set_obs("test", value)
    else:
        params.set_obs("test", value)


def test_data_handler_initalize_from_object(adata_tra):
    dh = DataHandler(adata_tra, "airr", "airr")
    dh2 = DataHandler(dh)
    assert dh._data is dh2._data is adata_tra
    assert dh.adata is dh2.adata


def test_is_symmetric():
    M = np.array([[1, 2, 2], [2, 1, 3], [2, 3, 1]])
    S_csr = scipy.sparse.csr_matrix(M)
    S_csc = scipy.sparse.csc_matrix(M)
    S_lil = scipy.sparse.lil_matrix(M)
    assert _is_symmetric(M)
    assert _is_symmetric(S_csr)
    assert _is_symmetric(S_csc)
    assert _is_symmetric(S_lil)

    M = np.array([[1, 2, 2], [2, 1, np.nan], [2, np.nan, np.nan]])
    S_csr = scipy.sparse.csr_matrix(M)
    S_csc = scipy.sparse.csc_matrix(M)
    S_lil = scipy.sparse.lil_matrix(M)
    assert _is_symmetric(M)
    assert _is_symmetric(S_csr)
    assert _is_symmetric(S_csc)
    assert _is_symmetric(S_lil)

    M = np.array([[1, 2, 2], [2, 1, 3], [3, 2, 1]])
    S_csr = scipy.sparse.csr_matrix(M)
    S_csc = scipy.sparse.csc_matrix(M)
    S_lil = scipy.sparse.lil_matrix(M)
    assert not _is_symmetric(M)
    assert not _is_symmetric(S_csr)
    assert not _is_symmetric(S_csc)
    assert not _is_symmetric(S_lil)


def test_is_na():
    warnings.filterwarnings("error")
    assert _is_na(None)
    assert _is_na(np.nan)
    assert _is_na("nan")
    assert not _is_na(42)
    assert not _is_na("Foobar")
    assert not _is_na({})
    array_test = np.array(["None", "nan", None, np.nan, "foobar"])
    array_expect = np.array([True, True, True, True, False])
    array_test_bool = np.array([True, False, True])
    array_expect_bool = np.array([False, False, False])

    npt.assert_equal(_is_na(array_test), array_expect)
    npt.assert_equal(_is_na(pd.Series(array_test)), array_expect)

    npt.assert_equal(_is_na(array_test_bool), array_expect_bool)
    npt.assert_equal(_is_na(pd.Series(array_test_bool)), array_expect_bool)


def test_is_false():
    warnings.filterwarnings("error")
    assert _is_false(False)
    assert _is_false(0)
    # assert _is_false("") -> I redelacred this as nan, as read_airr results in
    # null fields being "".
    assert _is_false("False")
    assert _is_false("false")
    assert not _is_false(42)
    assert not _is_false(True)
    assert not _is_false("true")
    assert not _is_false("foobar")
    assert not _is_false(np.nan)
    assert not _is_false(None)
    assert not _is_false("nan")
    assert not _is_false("None")
    array_test = np.array(
        ["False", "false", 0, 1, True, False, "true", "Foobar", np.nan, "nan"],
        dtype=object,
    )
    array_test_str = array_test.astype("str")
    array_expect = np.array([True, True, True, False, False, True, False, False, False, False])
    array_test_bool = np.array([True, False, True])
    array_expect_bool = np.array([False, True, False])

    npt.assert_equal(_is_false(array_test), array_expect)
    npt.assert_equal(_is_false(array_test_str), array_expect)
    npt.assert_equal(_is_false(pd.Series(array_test)), array_expect)
    npt.assert_equal(_is_false(pd.Series(array_test_str)), array_expect)
    npt.assert_equal(_is_false(array_test_bool), array_expect_bool)
    npt.assert_equal(_is_false(pd.Series(array_test_bool)), array_expect_bool)


def test_is_true():
    warnings.filterwarnings("error")
    assert not _is_true(False)
    assert not _is_true(0)
    assert not _is_true("")
    assert not _is_true("False")
    assert not _is_true("false")
    assert not _is_true("0")
    assert not _is_true(np.nan)
    assert not _is_true(None)
    assert not _is_true("nan")
    assert not _is_true("None")
    assert _is_true(42)
    assert _is_true(True)
    assert _is_true("true")
    assert _is_true("foobar")
    assert _is_true("True")
    array_test = np.array(
        ["False", "false", 0, 1, True, False, "true", "Foobar", np.nan, "nan"],
        dtype=object,
    )
    array_test_str = array_test.astype("str")
    array_expect = np.array([False, False, False, True, True, False, True, True, False, False])
    array_test_bool = np.array([True, False, True])
    array_expect_bool = np.array([True, False, True])

    npt.assert_equal(_is_true(array_test), array_expect)
    npt.assert_equal(_is_true(array_test_str), array_expect)
    npt.assert_equal(_is_true(pd.Series(array_test)), array_expect)
    npt.assert_equal(_is_true(pd.Series(array_test_str)), array_expect)
    npt.assert_equal(_is_true(array_test_bool), array_expect_bool)
    npt.assert_equal(_is_true(pd.Series(array_test_bool)), array_expect_bool)


@pytest.fixture
def group_df():
    return pd.DataFrame().assign(
        cell=["c1", "c2", "c3", "c4", "c5", "c6"],
        sample=["s2", "s1", "s2", "s2", "s2", "s1"],
    )


def test_normalize_counts(group_df):
    with pytest.raises(ValueError):
        _normalize_counts(group_df, True, None)

    npt.assert_equal(_normalize_counts(group_df, False), [1] * 6)
    npt.assert_equal(_normalize_counts(group_df, "sample"), [0.25, 0.5, 0.25, 0.25, 0.25, 0.5])
    npt.assert_equal(_normalize_counts(group_df, True, "sample"), [0.25, 0.5, 0.25, 0.25, 0.25, 0.5])


@pytest.mark.filterwarnings("ignore:UserWarning")
@pytest.mark.parametrize("arrange_boxes", ("size", "rpack", "squarify"))
@pytest.mark.parametrize("component_layout", ("fr_size_aware", "fr"))
def test_layout_components(arrange_boxes, component_layout):
    g = ig.Graph()

    # add 100 unconnected nodes
    g.add_vertices(100)

    # add 50 2-node components
    g.add_vertices(100)
    g.add_edges([(ii, ii + 1) for ii in range(100, 200, 2)])

    # add 33 3-node components
    g.add_vertices(100)
    for ii in range(200, 299, 3):
        g.add_edges([(ii, ii + 1), (ii, ii + 2), (ii + 1, ii + 2)])

    # add a couple of larger components
    n = 300
    for ii in np.random.randint(4, 30, size=10):
        g.add_vertices(ii)
        g.add_edges(combinations(range(n, n + ii), 2))
        n += ii

    try:
        layout_components(g, arrange_boxes=arrange_boxes, component_layout=component_layout)
    except ImportError:
        warnings.warn(f"The '{component_layout}' layout-test was skipped because rectangle packer is not installed. ")


def test_translate_dna_to_protein(adata_tra):
    for nt, aa in zip(
        ir.get.airr(adata_tra, "junction", "VJ_1"),
        ir.get.airr(adata_tra, "junction_aa", "VJ_1"),
        strict=False,
    ):
        assert _translate_dna_to_protein(cast(str, nt)) == aa


@pytest.mark.parametrize(
    "dist,expected_conn,max_value",
    [
        (
            [[0, 1, 1, 5], [0, 0, 2, 8], [1, 5, 0, 2], [10, 0, 0, 0]],
            [[0, 1, 1, 0.6], [0, 0, 0.9, 0.3], [1, 0.6, 0, 0.9], [0.1, 0, 0, 0]],
            None,
        ),
        (
            [[0, 1, 1, 5], [0, 0, 2, 8], [1, 5, 0, 2], [10, 0, 0, 0]],
            [
                [0, 1, 1, 0.8],
                [0, 0, 0.95, 0.65],
                [1, 0.8, 0, 0.95],
                [0.55, 0, 0, 0],
            ],
            20,
        ),
        (
            [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            None,
        ),
    ],
)
def test_dist_to_connectivities(dist, expected_conn, max_value):
    dist_mat = scipy.sparse.csr_matrix(dist)
    C = _distance_to_connectivity(dist_mat, max_value=max_value)
    assert C.nnz == dist_mat.nnz
    npt.assert_equal(
        C.toarray(),
        np.array(expected_conn),
    )


@pytest.mark.parametrize("simplify", [True, False])
@pytest.mark.parametrize(
    "matrix,is_symmetric",
    [
        (
            [
                [1.0, 0.0, 0.0, 0.6],
                [0.0, 1.0, 0.9, 0.3],
                [1.0, 0.6, 1.0, 0.9],
                [0.1, 0.0, 0.0, 1.0],
            ],
            False,
        ),
        (
            [
                [0.0, 1.0, 1.0, 0.6],
                [0.0, 0.0, 0.9, 0.3],
                [1.0, 0.6, 0.0, 0.9],
                [0.1, 0.0, 0.0, 0.0],
            ],
            False,
        ),
        (
            [
                [1.0, 0.3, 0.0, 0.6],
                [0.3, 1.0, 0.9, 0.3],
                [0.0, 0.9, 1.0, 0.9],
                [0.6, 0.3, 0.9, 1.0],
            ],
            True,
        ),
    ],
)
def test_igraph_from_adjacency(matrix, is_symmetric, simplify):
    matrix = scipy.sparse.csr_matrix(matrix)
    g = igraph_from_sparse_matrix(matrix, matrix_type="connectivity", simplify=simplify)
    assert len(list(g.vs)) == matrix.shape[0]
    matrix_roundtrip = _get_sparse_from_igraph(g, simplified=simplify, weight_attr="weight")
    if simplify and not is_symmetric:
        with pytest.raises(AssertionError):
            npt.assert_equal(matrix.toarray(), matrix_roundtrip.toarray())
    else:
        npt.assert_equal(matrix.toarray(), matrix_roundtrip.toarray())


@pytest.mark.parametrize(
    "X,expected",
    [
        (
            np.array(
                "4 2 7 3 1 2 5 0 4 3 6 0 5 4 0 5 2 2 4 4 1 4 3 1 4 2 5 5 7 3 4 3 4 2 3 10 "
                "1 3 4 3 7 4 2 3 1 5 1 5 2 7 3 3 8 6 4 5 3 2 2 3 2 5 1 3 1 4 3 2 5 1 5 3 4 "
                "1 5 2 2 2 4 1 5 8 7 3 1 2 1 3 1 6 6 6 1 3 3 0 2 2 5 4".split()
            ).astype(int),
            ([18.4954044, 0.8462615]),
        )
    ],
)
def test_fit_negative_binomial(X, expected):
    npt.assert_almost_equal(fit_nbinom(X), expected, decimal=5)
