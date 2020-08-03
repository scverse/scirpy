# pylama:ignore=W0611,W0404
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
import scirpy as st
from anndata import AnnData
import numpy as np
from scirpy.util import _is_symmetric
from .fixtures import (
    adata_conn,
    adata_define_clonotypes,
    adata_define_clonotype_clusters,
    adata_clonotype_network,
    adata_clonotype,
)
import random
import pytest


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


@pytest.mark.parametrize(
    "same_v_gene,ct_expected,ct_size_expected",
    [
        (False, ["0", "0", "0", "1"], [3, 3, 3, 1]),
        (
            "primary_only",
            ["0_av1_bv1", "0_av1_bv1", "0_av2_bv2", "1_av1_bv1"],
            [2, 2, 1, 1],
        ),
        (
            "all",
            [
                "0_av1_bv1_a2v1_b2v1",
                "0_av1_bv1_a2v2_b2v2",
                "0_av2_bv2_a2v2_b2v2",
                "1_av1_bv1_a2v1_b2v1",
            ],
            [1, 1, 1, 1],
        ),
    ],
)
def test_define_clonotype_clusters(
    adata_conn, same_v_gene, ct_expected, ct_size_expected
):
    clonotype, clonotype_size = st.tl.define_clonotype_clusters(
        adata_conn,
        metric="alignment",
        sequence="aa",
        same_v_gene=same_v_gene,
        inplace=False,
        partitions="connected",
    )
    npt.assert_equal(clonotype, ct_expected)
    npt.assert_equal(clonotype_size, ct_size_expected)

    st.tl.define_clonotype_clusters(
        adata_conn,
        metric="alignment",
        sequence="aa",
        same_v_gene=same_v_gene,
        key_added="ct",
        partitions="leiden",
        resolution=0.5,
        n_iterations=10,
    )
    npt.assert_equal(adata_conn.obs["ct"].values, ct_expected)
    npt.assert_equal(adata_conn.obs["ct_size"].values, ct_size_expected)

    # Test with higher leiden resolution
    st.tl.define_clonotype_clusters(
        adata_conn,
        neighbors_key="tcr_neighbors_aa_alignment",
        same_v_gene=False,
        key_added="ct2",
        partitions="leiden",
        resolution=2,
        n_iterations=10,
    )
    npt.assert_equal(adata_conn.obs["ct2"].values, ["0", "1", "2", "3"])
    npt.assert_equal(adata_conn.obs["ct2_size"].values, [1] * 4)


def test_clonotypes_end_to_end1(adata_define_clonotypes):
    # default parameters of tcr-neighbors should yield nt-identity
    st.pp.tcr_neighbors(adata_define_clonotypes, receptor_arms="all", dual_tcr="all")
    clonotypes, _ = st.tl.define_clonotypes(adata_define_clonotypes, inplace=False)
    print(clonotypes)
    expected = [0, 0, 1, 2, 3]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end1(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="all",
        dual_tcr="all",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end2(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="any",
        dual_tcr="any",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end3(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="all",
        dual_tcr="any",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 0, 0, 0, 0, 1, 0, 2, 3]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end4(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="any",
        dual_tcr="all",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end5(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="all",
        dual_tcr="primary_only",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 1, 2, 0, 0, 3, 4, 5, 6]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_clusters_end_to_end6(adata_define_clonotype_clusters):
    st.pp.tcr_neighbors(
        adata_define_clonotype_clusters,
        cutoff=0,
        receptor_arms="TRB",
        dual_tcr="primary_only",
        sequence="aa",
    )
    clonotypes, _ = st.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters, inplace=False
    )
    print(clonotypes)
    expected = [0, 0, 0, 1, 0, 0, 2, 3, 4, 5]
    npt.assert_equal(clonotypes, [str(x) for x in expected])


def test_clonotype_network(adata_conn):
    st.tl.define_clonotype_clusters(
        adata_conn, sequence="aa", metric="alignment", partitions="connected"
    )
    random.seed(42)
    coords = st.tl.clonotype_network(
        adata_conn,
        sequence="aa",
        metric="alignment",
        min_size=1,
        layout="fr",
        inplace=False,
    )
    npt.assert_almost_equal(
        coords,
        np.array(
            [
                [5.147361, 3.1383265],
                [3.4346971, 4.2259229],
                [4.0405687, 3.4865629],
                [5.2082453, 5.1293543],
            ]
        ),
    )

    random.seed(42)
    st.tl.clonotype_network(
        adata_conn,
        sequence="aa",
        metric="alignment",
        min_size=2,
        layout="components",
        inplace=True,
        key_added="ctn",
    )
    coords = adata_conn.obsm["X_ctn"]
    npt.assert_almost_equal(
        coords,
        np.array(
            [[98.0, 1.0], [1.0, 98.0], [49.5107979, 49.4911286], [np.nan, np.nan]]
        ),
    )

    with pytest.raises(ValueError):
        st.tl.clonotype_network(adata_conn[[1, 3], :])


def test_clonotype_network_igraph(adata_clonotype_network):
    g, lo = st.tl.clonotype_network_igraph(adata_clonotype_network)
    assert g.vcount() == 3
    npt.assert_almost_equal(
        np.array(lo.coords),
        np.array(
            [
                [2.41359095, 0.23412465],
                [1.61680611, 0.80266963],
                [3.06104282, 2.14395562],
            ]
        ),
    )


def test_clonotype_convergence(adata_clonotype):
    res = st.tl.clonotype_convergence(
        adata_clonotype,
        key_coarse="clonotype_cluster",
        key_fine="clonotype",
        inplace=False,
    )
    st.tl.clonotype_convergence(
        adata_clonotype,
        key_coarse="clonotype_cluster",
        key_fine="clonotype",
        inplace=True,
        key_added="is_convergent_",
    )
    pdt.assert_extension_array_equal(res, adata_clonotype.obs["is_convergent_"].values)
    pdt.assert_extension_array_equal(
        res,
        pd.Categorical(
            ["not convergent"] * 5 + ["convergent"] * 2 + ["not convergent"] * 2,
            categories=["convergent", "not convergent"],
        ),
    )

    res = st.tl.clonotype_convergence(
        adata_clonotype,
        key_fine="clonotype_cluster",
        key_coarse="clonotype",
        inplace=False,
    )
    pdt.assert_extension_array_equal(
        res,
        pd.Categorical(
            ["not convergent"] * 9, categories=["convergent", "not convergent"],
        ),
    )
