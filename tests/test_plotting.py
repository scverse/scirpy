# pylama:ignore=W0611,W0404
import pandas as pd
from anndata import AnnData
import pytest
from sctcrpy import pl
from .fixtures import adata_tra, adata_clonotype, adata_diversity
import matplotlib.pyplot as plt


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
    p = pl.group_abundance(adata_clonotype, groupby="group")
    assert isinstance(p, plt.Axes)


def test_spectratype(adata_clonotype):
    p = pl.group_abundance(adata_clonotype, groupby="group")
    assert isinstance(p, plt.Axes)
