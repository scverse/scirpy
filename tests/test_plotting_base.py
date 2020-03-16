from sctcrpy import pl
import pytest
import pandas as pd


@pytest.fixture
def test_df():
    return pd.DataFrame.from_dict(
        {"ct1": {"A": 3.0, "B": 1.0, "C": 4}, "ct2": {"A": 0.0, "B": 1.0, "C": 2.5},},
        orient="index",
    )


def test_bar(test_df):
    pl.base.bar(test_df)
    pl.base.bar(test_df, style=None)


def test_line(test_df):
    pl.base.line(test_df)


def test_barh(test_df):
    pl.base.barh(test_df)


def test_curve(test_df):
    # with default options
    pl.base.curve(test_df)

    # test curve layouts
    for cl in ["overlay", "stacked", "shifted"]:
        pl.base.curve(test_df, curve_layout=cl)

    # with shade
    pl.base.curve(test_df, shade=True)
