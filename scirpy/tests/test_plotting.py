# pylama:ignore=W0611,W0404
from scipy import sparse
from scirpy import pl
from .fixtures import (
    adata_tra,
    adata_clonotype,
    adata_diversity,
    adata_vdj,
    adata_conn,
    adata_define_clonotype_clusters,
    adata_clonotype_network,
    adata_define_clonotypes,
    adata_clonotype_modularity,
)
import matplotlib.pyplot as plt
import pytest


def test_clonal_expansion(adata_clonotype):
    p = pl.clonal_expansion(adata_clonotype, groupby="group")
    assert isinstance(p, plt.Axes)

    p = pl.clonal_expansion(
        adata_clonotype, groupby="group", show_nonexpanded=False, viztype="barh"
    )
    assert isinstance(p, plt.Axes)


def test_alpha_diversity(adata_diversity):
    p = pl.alpha_diversity(adata_diversity, groupby="group", target_col="clonotype_")
    assert isinstance(p, plt.Axes)


def test_group_abundance(adata_clonotype):
    p = pl.group_abundance(adata_clonotype, groupby="clone_id", target_col="group")
    assert isinstance(p, plt.Axes)


def test_spectratype(adata_tra):
    p = pl.spectratype(adata_tra, color="sample")
    assert isinstance(p, plt.Axes)


def test_repertoire_overlap(adata_tra):
    p = pl.repertoire_overlap(adata_tra, groupby="sample", dendro_only=True)
    assert isinstance(p, plt.Axes)


def test_clonotype_imbalance(adata_tra):
    p = pl.clonotype_imbalance(
        adata_tra,
        replicate_col="sample",
        groupby="chain_pairing",
        case_label="Single pair",
        plot_type="volcano",
    )
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("full_combination", [True, False])
def test_vdj_usage(adata_vdj, full_combination):
    p = pl.vdj_usage(
        adata_vdj, normalize_to="sample", full_combination=full_combination
    )
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("matrix_type", ["array", "csr", "csc"])
@pytest.mark.parametrize("use_raw", [False, None])
@pytest.mark.parametrize("cmap", [None, "cividis"])
def test_clonotype_network_gene(adata_clonotype_network, matrix_type, use_raw, cmap):
    adata = adata_clonotype_network
    if matrix_type == "csr":
        adata.X = sparse.csr_matrix(adata.X)
    elif matrix_type == "csc":
        adata.X = sparse.csc_matrix(adata.X)
    p = pl.clonotype_network(adata, color="CD8A", use_raw=use_raw, cmap=cmap)
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("jitter", [None, 0.02])
@pytest.mark.parametrize("show_size_legend", [True, False])
@pytest.mark.parametrize("show_labels", [True, False])
@pytest.mark.parametrize("labels", [None, ["2"]])
def test_clonotype_modularity(
    adata_clonotype_modularity, jitter, show_size_legend, show_labels, labels
):
    pl.clonotype_modularity(
        adata_clonotype_modularity,
        target_col="clonotype_modularity_x",
        jitter=jitter,
        show_size_legend=show_size_legend,
        show_labels=show_labels,
        labels=labels,
    )


@pytest.mark.parametrize("color_by_n_cells", [True, False])
@pytest.mark.parametrize("scale_by_n_cells", [True, False])
@pytest.mark.parametrize("show_size_legend", [True, False])
@pytest.mark.parametrize("show_legend", [True, False])
@pytest.mark.parametrize("show_labels", [True, False])
def test_clonotype_network(
    adata_clonotype_network,
    color_by_n_cells,
    scale_by_n_cells,
    show_size_legend,
    show_legend,
    show_labels,
):
    adata = adata_clonotype_network
    p = pl.clonotype_network(
        adata,
        color_by_n_cells=color_by_n_cells,
        scale_by_n_cells=scale_by_n_cells,
        show_size_legend=show_size_legend,
        show_legend=show_legend,
        show_labels=show_labels,
    )
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("show_size_legend", [True, False])
@pytest.mark.parametrize("show_legend", [True, False])
def test_clonotype_network_pie(
    adata_clonotype_network,
    show_size_legend,
    show_legend,
):
    adata = adata_clonotype_network
    p = pl.clonotype_network(
        adata,
        color="receptor_type",
        show_size_legend=show_size_legend,
        show_legend=show_legend,
    )
    assert isinstance(p, plt.Axes)
