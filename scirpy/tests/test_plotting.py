# pylama:ignore=W0611,W0404
import pandas as pd
from anndata import AnnData
import pytest
from scirpy import pl
from .fixtures import (
    adata_tra,
    adata_clonotype,
    adata_diversity,
    adata_vdj,
    adata_clonotype_network,
)
import scirpy._plotting as irplt
import matplotlib.pyplot as plt
import numpy.testing as npt
import pandas.testing as pdt
import numpy as np


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
    p = pl.group_abundance(adata_clonotype, groupby="clonotype", target_col="group")
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


def test_vdj_usage(adata_vdj):
    p = pl.vdj_usage(adata_vdj, normalize_to="sample")
    assert isinstance(p, plt.Axes)


def test_clonotype_network(adata_clonotype_network):
    p = pl.clonotype_network(adata_clonotype_network)
    assert isinstance(p[0], plt.Axes)
