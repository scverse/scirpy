import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scirpy import pl
from scirpy.util import DataHandler


@pytest.fixture
def test_df():
    return pd.DataFrame.from_dict(
        {
            "ct1": {"A": 3.0, "B": 1.0, "C": 4},
            "ct2": {"A": 0.0, "B": 1.0, "C": 2.5},
        },
        orient="index",
    )


@pytest.fixture
def test_dict():
    return {
        "ct1": np.array([2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7]),
        "ct2": np.array([4, 5, 6, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 2, 3, 4, 5, 3]),
        "ct3": np.array([2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7]),
    }


@pytest.fixture
def adata_obs():
    adata = AnnData(obs=pd.DataFrame().assign(col1=["foo", "foo", "bar", "bar"], col2=["foo", "nan", "xxx", "nan"]))
    return adata


def test_get_color(adata_obs):
    # alphabeteical order
    col1_color = {"bar": "#8dd3c7", "foo": "#ffed6f"}
    col2_color = {"foo": "#1f77b4", "nan": "#ff7f0e", "xxx": "#2ca02c"}
    assert pl.styling._get_colors(DataHandler(adata_obs), "col1", palette="Set3") == col1_color
    assert pl.styling._get_colors(DataHandler(adata_obs), "col2") == col2_color
    # check that it is the same color on repeated calling, even without specifying
    # the palette
    assert pl.styling._get_colors(DataHandler(adata_obs), "col1") == col1_color
    assert pl.styling._get_colors(DataHandler(adata_obs), "col2") == col2_color


def test_bar(test_df):
    pl.base.bar(test_df)
    p = pl.base.bar(test_df, style=None)
    assert isinstance(p, plt.Axes)


def test_line(test_df):
    p = pl.base.line(test_df)
    assert isinstance(p, plt.Axes)


def test_barh(test_df):
    p = pl.base.barh(test_df)
    assert isinstance(p, plt.Axes)


def test_ol_scatter(test_df):
    test_df.columns = ["x", "y", "z"]
    p = pl.base.ol_scatter(test_df)
    assert isinstance(p, plt.Axes)


def test_curve(test_dict):
    # with default options
    p = pl.base.curve(test_dict)
    assert isinstance(p, plt.Axes)

    # test curve layouts
    for cl in ["overlay", "stacked", "shifted"]:
        pl.base.curve(test_dict, curve_layout=cl)

    # with shade
    pl.base.curve(test_dict, shade=True)
