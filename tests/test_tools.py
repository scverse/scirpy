import pandas as pd
import sctcrpy as st
from scanpy import AnnData
import pytest


def test_define_clonotypes():
    pass


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
    assert res["diversity"] == {"A": 0, "B": 2}

    # test that inplace works
    st.tl.alpha_diversity(adata, groupby="group", target_col="clonotype_")
    assert adata.uns["sctcrpy"]["alpha_diversity"] == res


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
    assert adata.uns["sctcrpy"]["clonal_expansion"] == res_frac

    # check if target_col works
    adata.obs["new_col"] = adata.obs["clonotype"]
    adata.obs.drop("clonotype", axis="columns", inplace=True)
    with pytest.raises(ValueError):
        res = st.tl.clonal_expansion(
            adata, groupby="group", clip_at=2, inplace=False, fraction=False
        )
    res = st.tl.clonal_expansion(
        adata,
        groupby="group",
        clip_at=2,
        inplace=False,
        fraction=False,
        target_col="new_col",
    )
    assert res == {"A": {"1": 0, ">= 2": 1}, "B": {"1": 3, ">= 2": 1}}
