import pandas as pd
import numpy.testing as npt
import sctcrpy as st
from anndata import AnnData
import numpy as np
import pytest
from sctcrpy._util import _is_symmetric
import random


@pytest.fixture
def adata_conn():
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(cell_id=["cell1", "cell2", "cell3", "cell4"])
        .set_index("cell_id")
    )
    adata.uns["tcr_neighbors"] = {
        "connectivities": np.array(
            [[1, 0, 0.5, 0], [0, 1, 1, 0], [0.5, 1, 1, 0], [0, 0, 0, 1]]
        )
    }
    assert _is_symmetric(adata.uns["tcr_neighbors"]["connectivities"])
    return adata


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


def test_define_clonotypes(adata_conn):
    ct_expected = ["0", "0", "0", "1"]
    ct_size_expected = [3, 3, 3, 1]

    clonotype, clonotype_size = st.tl.define_clonotypes(
        adata_conn, inplace=False, partitions="connected"
    )
    npt.assert_equal(clonotype, ct_expected)
    npt.assert_equal(clonotype_size, ct_size_expected)

    st.tl.define_clonotypes(adata_conn, key_added="ct", resolution=0.5, n_iterations=10)
    npt.assert_equal(adata_conn.obs["ct"].values, ct_expected)
    npt.assert_equal(adata_conn.obs["ct_size"].values, ct_size_expected)

    st.tl.define_clonotypes(adata_conn, key_added="ct2", resolution=2, n_iterations=10)
    npt.assert_equal(adata_conn.obs["ct2"].values, ["0", "1", "2", "3"])
    npt.assert_equal(adata_conn.obs["ct2_size"].values, [1] * 4)


def test_clonotype_network(adata_conn):
    st.tl.define_clonotypes(adata_conn, partitions="connected")
    random.seed(42)
    coords = st.tl.clonotype_network(adata_conn, min_size=1, layout="fr", inplace=False)
    npt.assert_almost_equal(
        coords,
        np.array(
            [
                [2.41359095, 0.23412465],
                [1.10836324, 1.63916503],
                [1.61680611, 0.80266963],
                [3.06104282, 2.14395562],
            ]
        ),
    )

    random.seed(42)
    st.tl.clonotype_network(
        adata_conn, min_size=2, layout="components", inplace=True, key_added="X_ctn"
    )
    coords = adata_conn.obsm["X_ctn"]
    npt.assert_almost_equal(
        coords,
        np.array([[2.0, 0.0], [0.0, 2.0], [1.00022264, 0.99981708], [np.nan, np.nan]]),
    )
