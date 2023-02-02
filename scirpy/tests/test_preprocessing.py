import awkward as ak
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from anndata import AnnData

from ..io import AirrCell, read_10x_vdj
from ..pp import index_chains
from ..pp._merge_adata import merge_airr
from . import TESTDATA
from .util import _make_airr_chains_valid


@pytest.mark.parametrize(
    "airr_chains,expected_index",
    [
        (
            [
                # fmt: off
                [
                    {"locus": "TRA", "junction_aa": "AAA", "duplicate_count": 3, "productive": True},
                    {"locus": "TRA", "junction_aa": "KKK", "duplicate_count": 6, "productive": True},
                    {"locus": "TRB", "junction_aa": "LLL", "duplicate_count": 3, "productive": True},
                ],
                [
                    {"locus": "TRB", "junction_aa": "KKK", "duplicate_count": 6, "productive": True},
                    {"locus": "TRA", "junction_aa": "AAA", "duplicate_count": 3, "productive": True},
                    {"locus": "TRB", "junction_aa": "LLL", "duplicate_count": 3, "productive": True},
                ],
                # fmt: on
            ],
            [
                # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
                [1, 2, 0, np.nan, False],
                [1, 0, np.nan, 2, False],
            ],
        )
    ],
    ids=["standard case, multiple rows"],
)
def test_index_chains(airr_chains, expected_index):
    """Test that chain indexing works as expected (default parameters)"""
    adata = AnnData(
        X=None, obs=pd.DataFrame(index=[f"cell_{i}" for i in range(len(airr_chains))])  # type: ignore
    )
    adata.obsm["airr2"] = ak.Array(airr_chains)
    index_chains(adata, airr_key="airr2", key_added="chain_indices2")
    pdt.assert_frame_equal(
        adata.obsm["chain_indices2"],
        pd.DataFrame(
            expected_index,
            index=adata.obs_names,
            columns=["VJ_1", "VDJ_1", "VJ_2", "VDJ_2", "multichain"],
        ),
    )


@pytest.mark.parametrize(
    "productive,require_junction_aa,sort_chains_by,expected_index",
    [
        (
            True,
            True,
            {
                "duplicate_count": 0,
                "consensus_count": 0,
                "junction": "",
                "junction_aa": "",
            },
            # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
            [3, np.nan, 0, np.nan, False],
        ),
        (
            False,
            True,
            {"junction_aa": ""},
            # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
            [3, np.nan, 1, np.nan, True],
        ),
        (
            True,
            False,
            {"junction_aa": ""},
            # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
            [3, np.nan, 0, np.nan, True],
        ),
        (
            False,
            False,
            {"junction_aa": ""},
            # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
            [3, np.nan, 1, np.nan, True],
        ),
        (
            True,
            False,
            {"sort": 10000},
            # VJ_1, VDJ_1, VJ_2, VDJ_2, multichain
            [2, np.nan, 3, np.nan, True],
        ),
    ],
    ids=[
        "default parameters",
        "productive = False",
        "require_junction_aa = False",
        "productive = False & require_junction_aa = False",
        "custom sort function",
    ],
)
def test_index_chains_custom_parameters(
    productive, require_junction_aa, sort_chains_by, expected_index
):
    """Test that parameters for chain indexing work as intended (Single data, different params)"""
    airr_chains = [
        [
            {"locus": "TRA", "junction_aa": "AAA", "sort": 2, "productive": True},
            {"locus": "TRA", "junction_aa": "AAB", "sort": 5, "productive": False},
            {"locus": "TRA", "productive": True},
            {"locus": "TRA", "junction_aa": "AAD", "sort": 3, "productive": True},
        ]
    ]
    adata = AnnData(
        X=None, obs=pd.DataFrame(index=[f"cell_{i}" for i in range(len(airr_chains))])
    )
    adata.obsm["airr"] = ak.Array(airr_chains)
    index_chains(
        adata,
        productive=productive,
        require_junction_aa=require_junction_aa,
        sort_chains_by=sort_chains_by,
    )
    pdt.assert_frame_equal(
        adata.obsm["chain_indices"],
        pd.DataFrame(
            [expected_index],
            index=adata.obs_names,
            columns=["VJ_1", "VDJ_1", "VJ_2", "VDJ_2", "multichain"],
        ),
        check_dtype=True,
    )


def test_merge_airr_identity():
    """Test that mergeing adata with itself results in the identity"""
    adata1 = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata2 = adata1.copy()

    obs_expected = adata1.obs.copy()
    airr_expected = ak.to_list(adata1.obsm["airr"])

    adata_merged = merge_airr(adata1, adata2)
    obs_merged = adata_merged.obs
    airr_merged = ak.to_list(adata_merged.obsm["airr"])

    pdt.assert_frame_equal(obs_merged, obs_expected)
    airr_expected == airr_merged


@pytest.mark.parametrize(
    "airr1,airr2,airr_expected",
    [
        (
            [[{"locus": "TRA", "junction_aa": "AAA"}], []],
            [[{"locus": "TRB", "junction_aa": "BBB"}], []],
            [
                [
                    {"locus": "TRA", "junction_aa": "AAA"},
                    {"locus": "TRB", "junction_aa": "BBB"},
                ],
                [],
            ],
        ),
        (
            [[{"locus": "TRA", "junction_aa": "AAA"}], []],
            [[], [{"locus": "TRB", "junction_aa": "BBB"}]],
            [
                [{"locus": "TRA", "junction_aa": "AAA"}],
                [{"locus": "TRB", "junction_aa": "BBB"}],
            ],
        ),
    ],
)
def test_merge_airr(airr1, airr2, airr_expected):
    airr1 = _make_airr_chains_valid(airr1)
    airr2 = _make_airr_chains_valid(airr2)
    airr_expected = _make_airr_chains_valid(airr_expected)

    adata = AnnData(
        obs=pd.DataFrame(index=[str(x) for x in range(len(airr1))]),
        obsm={"xxx": ak.Array(airr1)},
    )
    adata2 = AnnData(
        obs=pd.DataFrame(index=[str(x) for x in range(len(airr2))]),
        obsm={"yyy": ak.Array(airr2)},
    )
    adata_merged = merge_airr(adata, adata2, airr_key="xxx", airr_key2="yyy")
    assert ak.to_list(adata_merged.obsm["airr"]) == airr_expected


@pytest.mark.parametrize(
    "obs1,obs2,obs_expected",
    [
        pytest.param(
            {"cell1": {"a": 1, "b": "test"}, "cell2": {"a": 1, "b": "test"}},
            {"cell1": {"a": 1, "b": "test"}, "cell2": {"a": 1, "b": "test"}},
            {"cell1": {"a": 1, "b": "test"}, "cell2": {"a": 1, "b": "test"}},
        ),
        pytest.param(
            {"cell1": {"a": 1, "b": "test"}},
            {"cell2": {"a": 1, "b": "test"}},
            {"cell1": {"a": 1, "b": "test"}, "cell2": {"a": 1, "b": "test"}},
        ),
        pytest.param(
            {"cell1": {"a": 1, "b": "test"}, "cell2": {"a": 1, "b": "test"}},
            {"cell1": {"a": 2, "b": "test"}, "cell2": {"a": 1, "b": "xxx"}},
            ValueError,
        ),
    ],
)
def test_merge_airr_obs(obs1, obs2, obs_expected):
    adata1 = AnnData(
        obs=pd.DataFrame.from_dict(obs1, orient="index"),
        obsm={"airr": ak.Array([[AirrCell.empty_chain_dict()]] * len(obs1))},
    )
    adata2 = AnnData(
        obs=pd.DataFrame.from_dict(obs2, orient="index"),
        obsm={"airr": ak.Array([[AirrCell.empty_chain_dict()]] * len(obs2))},
    )
    if isinstance(obs_expected, type) and issubclass(obs_expected, Exception):
        with pytest.raises(obs_expected):
            adata_merged = merge_airr(adata1, adata2)
    else:
        adata_merged = merge_airr(adata1, adata2)
        assert adata_merged.obs.to_dict(orient="index") == obs_expected
