# pylama:ignore=W0611,W0404
import pandas as pd
from anndata import AnnData
import pytest
from sctcrpy import pl
from .fixtures import (
    adata_tra,
    adata_clonotype,
    adata_diversity,
    adata_clonotype_network,
)
import matplotlib.pyplot as plt
import numpy.testing as npt
import numpy as np


def test_clip_and_count(adata_tra):
    p = pl.clip_and_count(
        adata_tra, target_col="TRA_1_cdr3", groupby="sample", fraction=False
    )
    assert isinstance(p, plt.Axes)


def test_clonal_expansion(adata_clonotype):
    p = pl.clonal_expansion(adata_clonotype, groupby="group")
    assert isinstance(p, plt.Axes)


def test_alpha_diversity(adata_diversity):
    p = pl.alpha_diversity(adata_diversity, groupby="group", target_col="clonotype_")
    assert isinstance(p, plt.Axes)


def test_group_abundance(adata_clonotype):
    p = pl.group_abundance(adata_clonotype, groupby="clonotype", target_col="group")
    assert isinstance(p, plt.Axes)


def test_spectratype(adata_tra):
    p = pl.spectratype(adata_tra, target_col="sample")
    assert isinstance(p, plt.Axes)


def test_clonotype_network(adata_clonotype_network):
    p = pl.clonotype_network(adata_clonotype_network)
    assert isinstance(p, plt.Axes)


def test_clonotype_network_igraph(adata_clonotype_network):
    g, lo = pl.clonotype_network_igraph(adata_clonotype_network)
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
