# pylama:ignore=W0611,W0404
import logomaker
import matplotlib.pyplot as plt
import pytest
import seaborn as sns
from mudata import MuData
from scipy import sparse

from scirpy import pl


def test_clonal_expansion(adata_clonotype):
    p = pl.clonal_expansion(adata_clonotype, groupby="group")
    assert isinstance(p, plt.Axes)

    p = pl.clonal_expansion(adata_clonotype, groupby="group", show_nonexpanded=False, viztype="barh")
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("adata_clonotype", [True], indirect=["adata_clonotype"], ids=["MuData"])
def test_clonal_expansion_mudata_prefix(adata_clonotype):
    """Regression test for #445"""
    p = pl.clonal_expansion(adata_clonotype, groupby="group", target_col="airr:clone_id")
    assert isinstance(p, plt.Axes)


def test_alpha_diversity(adata_diversity):
    p = pl.alpha_diversity(adata_diversity, groupby="group", target_col="clonotype_")
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("adata_clonotype", [True], indirect=["adata_clonotype"], ids=["MuData"])
def test_group_abundance_default(adata_clonotype):
    """Regression test for #435"""
    # Change to 'tcr' as airr_mod instead of the default 'airr'.
    adata_clonotype = MuData({"tcr": adata_clonotype["airr"]})
    p = pl.group_abundance(
        adata_clonotype,
        groupby="tcr:clone_id",
        target_col="tcr:group",
    )
    assert isinstance(p, plt.Axes)


def test_group_abundance(adata_clonotype):
    mdata_modifier = "airr:" if isinstance(adata_clonotype, MuData) else ""

    p = pl.group_abundance(
        adata_clonotype,
        groupby=f"{mdata_modifier}clone_id",
        target_col=f"{mdata_modifier}group",
    )
    assert isinstance(p, plt.Axes)


def test_spectratype(adata_tra):
    p = pl.spectratype(adata_tra, color="sample")
    assert isinstance(p, plt.Axes)

    # test if error message highlighting the API change is raised
    with pytest.raises(ValueError):
        pl.spectratype(adata_tra, ["IR_VJ_1_junction_aa"], color="sample")

    with pytest.raises(ValueError):
        pl.spectratype(adata_tra, cdr3_col="IR_VJ_1_junction_aa", color="sample")


def test_repertoire_overlap(adata_tra):
    p = pl.repertoire_overlap(adata_tra, groupby="sample")
    assert isinstance(p, sns.matrix.ClusterGrid)


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
    p = pl.vdj_usage(adata_vdj, normalize_to="sample", full_combination=full_combination)
    assert isinstance(p, plt.Axes)


@pytest.mark.extra
@pytest.mark.parametrize("matrix_type", ["array", "csr", "csc"])
@pytest.mark.parametrize("use_raw", [False, None])
@pytest.mark.parametrize("cmap", [None, "cividis"])
def test_clonotype_network_gene(adata_clonotype_network, matrix_type, use_raw, cmap):
    adata = adata_clonotype_network
    tmp_ad = adata.mod["gex"] if isinstance(adata, MuData) else adata
    if matrix_type == "csr":
        tmp_ad.X = sparse.csr_matrix(tmp_ad.X)
    elif matrix_type == "csc":
        tmp_ad.X = sparse.csc_matrix(tmp_ad.X)
    p = pl.clonotype_network(adata, color="CD8A", use_raw=use_raw, cmap=cmap)
    assert isinstance(p, plt.Axes)


@pytest.mark.parametrize("jitter", [None, 0.02])
@pytest.mark.parametrize("show_size_legend", [True, False])
@pytest.mark.parametrize("show_labels", [True, False])
@pytest.mark.parametrize("labels", [None, ["2"]])
def test_clonotype_modularity(adata_clonotype_modularity, jitter, show_size_legend, show_labels, labels):
    pl.clonotype_modularity(
        adata_clonotype_modularity,
        target_col="clonotype_modularity_x",
        jitter=jitter,
        show_size_legend=show_size_legend,
        show_labels=show_labels,
        labels=labels,
    )


@pytest.mark.extra
@pytest.mark.parametrize(
    "adata_clonotype_network,kwargs",
    [[{}, {}], [{"key_added": "foo"}, {"basis": "foo"}]],
    indirect=["adata_clonotype_network"],
)
@pytest.mark.parametrize("color_by_n_cells", [True, False])
@pytest.mark.parametrize("scale_by_n_cells", [True, False])
@pytest.mark.parametrize("show_size_legend", [True, False])
@pytest.mark.parametrize("show_legend", [True, False])
@pytest.mark.parametrize("show_labels", [True, False])
def test_clonotype_network(
    adata_clonotype_network, color_by_n_cells, scale_by_n_cells, show_size_legend, show_legend, show_labels, kwargs
):
    adata = adata_clonotype_network
    p = pl.clonotype_network(
        adata,
        color_by_n_cells=color_by_n_cells,
        scale_by_n_cells=scale_by_n_cells,
        show_size_legend=show_size_legend,
        show_legend=show_legend,
        show_labels=show_labels,
        **kwargs,
    )
    assert isinstance(p, plt.Axes)


@pytest.mark.extra
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


@pytest.mark.extra
def test_logoplot(adata_cdr3):
    p = pl.logoplot_cdr3_motif(adata_cdr3, chains="VJ_1")
    assert isinstance(p, logomaker.Logo)
