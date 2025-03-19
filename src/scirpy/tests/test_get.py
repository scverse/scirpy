import numpy as np
import pandas.testing as pdt
import pytest

from scirpy import get


def test_obs_context(adata_cdr3):
    adata_cdr3.obs["foo"] = "xxx"
    obs_pre = adata_cdr3.obs.copy()
    with get.obs_context(
        adata_cdr3,
        {
            "foo": "bar",
            "a": "b",
            "c": [1, 2, 3, 4, 5],
            "VJ_1_cdr3": get.airr(adata_cdr3, "junction_aa", "VJ_1"),
        },
    ) as a:
        assert a.obs is adata_cdr3.obs
        assert np.all(a.obs["foo"] == "bar")
        assert a.obs["VJ_1_cdr3"].tolist() == ["AAA", "AHA", np.nan, "AAA", "AAA"]
    pdt.assert_frame_equal(obs_pre, adata_cdr3.obs)


@pytest.mark.parametrize("kwargs", [{}, {"chain": "VJ_1"}])
def test_airr_context(adata_cdr3, kwargs):
    with get.airr_context(adata_cdr3, "junction_aa", **kwargs):
        assert adata_cdr3.obs["VJ_1_junction_aa"].tolist() == [
            "AAA",
            "AHA",
            np.nan,
            "AAA",
            "AAA",
        ]
