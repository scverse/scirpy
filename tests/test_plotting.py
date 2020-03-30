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


def test_clip_and_count(adata_tra, adata_clonotype):
    res_fraction = irplt._clip_and_count._prepare_df(
        adata_clonotype, "group", "clonotype", 2, True
    )
    res_counts = irplt._clip_and_count._prepare_df(
        adata_clonotype, "group", "clonotype", 2, False
    )
    pdt.assert_frame_equal(
        res_fraction,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 3 / 5], ">= 2": [1.0, 2 / 5]}
        ).set_index("group"),
        check_names=False,
    )
    pdt.assert_frame_equal(
        res_counts,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 3], ">= 2": [3, 2]}
        ).set_index("group"),
        check_names=False,
        check_dtype=False,
    )

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


def test_vdj_usage(adata_vdj):
    p = pl.vdj_usage(adata_vdj, fraction="sample")
    assert isinstance(p, plt.Axes)


def test_clonotype_network(adata_clonotype_network):
    p = pl.clonotype_network(adata_clonotype_network)
    assert isinstance(p[0], plt.Axes)


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
