import pandas as pd
import sctcrpy as st
from scanpy import AnnData
import pytest
import numpy.testing as npt
import numpy as np
from sctcrpy._util import _get_from_uns


def test_define_clonotypes():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "AAAA", "nan", "nan", "nan"],
            ["cell2", "nan", "nan", "nan", "nan"],
            ["cell3", "AAAA", "nan", "nan", "nan"],
            ["cell4", "AAAA", "BBBB", "nan", "nan"],
            ["cell5", "nan", "nan", "CCCC", "DDDD"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3", "TRB_1_cdr3", "TRB_2_cdr3"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    res = st.tl.define_clonotypes(adata, inplace=False)
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res,
        ["clonotype_1", np.nan, "clonotype_1", "clonotype_0", "clonotype_2"],
    )

    res_primary_only = st.tl.define_clonotypes(
        adata, flavor="primary_only", inplace=False
    )
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res_primary_only,
        ["clonotype_0", np.nan, "clonotype_0", "clonotype_0", "clonotype_1"],
    )

    # test inplace
    st.tl.define_clonotypes(adata, key_added="clonotype_")
    npt.assert_equal(res, adata.obs["clonotype_"].values)


def test_alpha_diversity():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype_"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    res = st.tl.alpha_diversity(
        adata, groupby="group", target_col="clonotype_", inplace=False
    )
    assert res == {"A": 0, "B": 2}

    # test that inplace works
    st.tl.alpha_diversity(adata, groupby="group", target_col="clonotype_")
    assert (
        _get_from_uns(
            adata,
            "alpha_diversity",
            parameters={"groupby": "group", "target_col": "clonotype_"},
        )
        == res
    )


def test_clonal_expansion():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
            ["cell8", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    res = st.tl.clonal_expansion(
        adata, groupby="group", clip_at=2, inplace=False, fraction=False
    )
    assert res == {"A": {"1": 0, ">= 2": 1}, "B": {"1": 3, ">= 2": 1}}

    res_frac = st.tl.clonal_expansion(adata, groupby="group", clip_at=2, inplace=False)
    assert res_frac == {"A": {"1": 0, ">= 2": 1.0}, "B": {"1": 0.75, ">= 2": 0.25}}

    # check if inplace works
    st.tl.clonal_expansion(adata, groupby="group", clip_at=2, inplace=True)
    assert (
        _get_from_uns(
            adata,
            "clonal_expansion",
            parameters={
                "groupby": "group",
                "clip_at": 2,
                "fraction": True,
                "target_col": "clonotype",
            },
        )
        == res_frac
    )

    # check if target_col works
    adata.obs["new_col"] = adata.obs["clonotype"]
    adata.obs.drop("clonotype", axis="columns", inplace=True)

    # check if it raises value error if target_col does not exist
    with pytest.raises(ValueError):
        res2 = st.tl.clonal_expansion(
            adata, groupby="group", clip_at=2, inplace=False, fraction=False
        )

    # check if it works with correct target_col
    res2 = st.tl.clonal_expansion(
        adata,
        groupby="group",
        clip_at=2,
        inplace=False,
        fraction=False,
        target_col="new_col",
    )
    assert res2 == res
