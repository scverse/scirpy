# pylama:ignore=W0611,W0404
import pandas as pd
import numpy.testing as npt
import pandas.testing as pdt
import scirpy as ir
from anndata import AnnData
import numpy as np
from scirpy.util import _is_symmetric
from .fixtures import (
    adata_conn,
    adata_conn_diagonal,
    adata_define_clonotypes,
    adata_define_clonotype_clusters,
    adata_clonotype_network,
    adata_clonotype,
)
import random
import pytest

# TODO test clonotype definition with v_genes and within_group

# TODO add regression test for #236.


def test_define_clonotypes_diagonal_connectivities(adata_conn_diagonal):
    clonotype, clonotype_size = st.tl.define_clonotype_clusters(
        adata_conn_diagonal,
        metric="alignment",
        within_group=None,
        sequence="aa",
        same_v_gene=False,
        inplace=False,
        partitions="connected",
    )
    print(clonotype)


@pytest.mark.parametrize(
    "same_v_gene,within_group,ct_expected,ct_size_expected",
    [
        (False, None, ["0", "0", "0", "1"], [3, 3, 3, 1]),
        (
            True,
            None,
            ["0_av1_bv1", "0_av1_bv1", "0_av2_bv2", "1_av1_bv1"],
            [2, 2, 1, 1],
        ),
        (
            True,
            None,
            [
                "0_av1_bv1_a2v1_b2v1",
                "0_av1_bv1_a2v2_b2v2",
                "0_av2_bv2_a2v2_b2v2",
                "1_av1_bv1_a2v1_b2v1",
            ],
            [1, 1, 1, 1],
        ),
        (False, "receptor_type", ["0_TCR", "0_BCR", "0_BCR", "1_BCR"], [1, 2, 2, 1]),
        (
            True,
            "receptor_type",
            ["0_av1_bv1_TCR", "0_av1_bv1_BCR", "0_av2_bv2_BCR", "1_av1_bv1_BCR"],
            [1, 1, 1, 1],
        ),
    ],
)
def test_define_clonotype_clusters(
    adata_conn, same_v_gene, within_group, ct_expected, ct_size_expected
):
    ir.pp.ir_dist(adata_conn)
    clonotype, clonotype_size = ir.tl.define_clonotype_clusters(
        adata_conn,
        metric="alignment",
        within_group=within_group,
        sequence="aa",
        same_v_gene=same_v_gene,
        inplace=False,
        partitions="connected",
    )
    npt.assert_equal(clonotype, ct_expected)
    npt.assert_equal(clonotype_size, ct_size_expected)

    ir.tl.define_clonotype_clusters(
        adata_conn,
        metric="alignment",
        within_group=within_group,
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
    ir.tl.define_clonotype_clusters(
        adata_conn,
        neighbors_key="ir_dist_aa_alignment",
        within_group=None,
        same_v_gene=False,
        key_added="ct2",
        partitions="leiden",
        resolution=2,
        n_iterations=10,
    )
    npt.assert_equal(adata_conn.obs["ct2"].values, ["0", "1", "2", "3"])
    npt.assert_equal(adata_conn.obs["ct2_size"].values, [1] * 4)


def test_clonotypes_end_to_end1(adata_define_clonotypes):
    # default parameters of ir-neighbors should yield nt-identity
    ir.pp.ir_dist(adata_define_clonotypes)
    clonotypes, clonotype_size, _ = ir.tl.define_clonotypes(
        adata_define_clonotypes,
        inplace=False,
        within_group=None,
        receptor_arms="all",
        dual_ir="all",
    )  # type: ignore
    print(clonotypes)
    expected = [0, 0, 1, 2, 3]
    expected_size = [2, 2, 1, 1, 1]
    npt.assert_equal(clonotypes.values, [str(x) for x in expected])
    npt.assert_equal(clonotype_size.values, expected_size)


@pytest.mark.parametrize(
    "receptor_arms,dual_ir,expected,expected_size",
    [
        (
            "all",
            "all",
            [0, 0, 1, 2, 3, np.nan, 4, 5, 6, 7, 8],
            [2, 2, 1, 1, 1, np.nan, 1, 1, 1, 1, 1],
        ),
        (
            "any",
            "any",
            [0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, 1],
            [9, 9, 9, 9, 9, np.nan, 9, 9, 9, 9, 1],
        ),
        (
            "all",
            "any",
            [0, 0, 0, 0, 0, np.nan, 0, 1, 0, 2, 3],
            [7, 7, 7, 7, 7, np.nan, 7, 1, 7, 1, 1],
        ),
        (
            "any",
            "all",
            [0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, 1],
            [9, 9, 9, 9, 9, np.nan, 9, 9, 9, 9, 1],
        ),
        (
            "all",
            "primary_only",
            [0, 0, 1, 2, 0, np.nan, 0, 3, 4, 5, 6],
            [4, 4, 1, 1, 4, np.nan, 4, 1, 1, 1, 1],
        ),
        (
            "VDJ",
            "primary_only",
            [0, 0, 0, 1, 0, np.nan, 0, 2, 3, 4, 5],
            [5, 5, 5, 1, 5, np.nan, 5, 1, 1, 1, 1],
        ),
    ],
)
def test_clonotype_clusters_end_to_end(
    adata_define_clonotype_clusters, receptor_arms, dual_ir, expected, expected_size
):
    ir.pp.ir_dist(
        adata_define_clonotype_clusters,
        cutoff=0,
        sequence="aa",
    )
    clonotypes, clonotype_size, _ = ir.tl.define_clonotype_clusters(
        adata_define_clonotype_clusters,
        inplace=False,
        within_group=None,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
    )  # type: ignore
    print(clonotypes)
    npt.assert_equal(
        list(clonotypes.values), [str(x) if not np.isnan(x) else x for x in expected]
    )
    npt.assert_almost_equal(clonotype_size.values, expected_size)


def test_clonotype_network(adata_conn):
    ir.tl.define_clonotype_clusters(
        adata_conn,
        sequence="aa",
        metric="alignment",
        partitions="connected",
        within_group=None,
    )
    random.seed(42)
    coords = ir.tl.clonotype_network(
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
    ir.tl.clonotype_network(
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
        ir.tl.clonotype_network(adata_conn[[1, 3], :])


def test_clonotype_network_igraph(adata_clonotype_network):
    g, lo = ir.tl.clonotype_network_igraph(adata_clonotype_network)
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
    res = ir.tl.clonotype_convergence(
        adata_clonotype,
        key_coarse="clonotype_cluster",
        key_fine="clonotype",
        inplace=False,
    )
    ir.tl.clonotype_convergence(
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

    res = ir.tl.clonotype_convergence(
        adata_clonotype,
        key_fine="clonotype_cluster",
        key_coarse="clonotype",
        inplace=False,
    )
    pdt.assert_extension_array_equal(
        res,
        pd.Categorical(
            ["not convergent"] * 9,
            categories=["convergent", "not convergent"],
        ),
    )
