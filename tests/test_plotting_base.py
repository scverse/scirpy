from scirpy import pl
import pytest
import pandas as pd
import matplotlib.pyplot as plt


@pytest.fixture
def test_df():
    return pd.DataFrame.from_dict(
        {"ct1": {"A": 3.0, "B": 1.0, "C": 4}, "ct2": {"A": 0.0, "B": 1.0, "C": 2.5},},
        orient="index",
    )


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
    test_df.columns = ['x', 'y', 'z']
    p = pl.base.ol_scatter(test_df)
    assert isinstance(p, plt.Axes)


def test_curve(test_df):
    # with default options
    p = pl.base.curve(test_df)
    assert isinstance(p, plt.Axes)

    # test curve layouts
    for cl in ["overlay", "stacked", "shifted"]:
        pl.base.curve(test_df, curve_layout=cl)

    # with shade
    pl.base.curve(test_df, shade=True)
