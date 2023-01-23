from ast import type_ignore
from os import read
from .util import _normalize_df_types
from anndata import AnnData
import numpy as np
from ..io import read_10x_vdj, read_bracer
from . import TESTDATA
from ..pp import merge_with_ir, index_chains
from ..pp._merge_adata import merge_airr_chains
import pandas.testing as pdt
import numpy.testing as npt
from ..util import _is_na
import pandas as pd
import pytest
import awkward as ak


@pytest.mark.parametrize(
    "airr_chains,expected_index",
    [
        # standard case, multiple rows
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
        # single VJ chain
        # (
        #     [[{"locus": "TRA", "junction_aa": "AAA", "duplicate_count": 3, "productive": True}]],
        #     [[]]
        # )
        # single VDJ chain
        # multichain (3 VJ chains)
        # no multichain (3 VJ chains, but one is not productive)
        # no multichain (3 VJ chains, but one does not have a junction_aa sequence)
        # ties in counts
        # deal with missing sort keys
    ],
)
def test_index_chains(airr_chains, expected_index):
    """Test that chain indexing works as expected (Multiple data, default parameters)"""
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


def test_merge_airr_chains_identity():
    """Test that mergeing adata with itself results in the identity"""
    adata1 = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata2 = adata1.copy()
    obs_expected = adata1.obs.copy()
    merge_airr_chains(adata1, adata2)
    obs_merged = adata1.obs
    _normalize_df_types(obs_merged)
    _normalize_df_types(obs_expected)
    pdt.assert_frame_equal(obs_merged, obs_expected)


def test_merge_airr_chains_concat():
    """Test that merging two unrelated ir objects (reindexing the first one) is
    the same as concatenating the objects"""
    adata1 = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata_bracer = read_bracer(TESTDATA / "bracer/changeodb.tab")
    obs_expected = pd.concat([adata1.obs.copy(), adata_bracer.obs])

    adata1 = AnnData(
        obs=adata1.obs.reindex(list(adata1.obs_names) + list(adata_bracer.obs_names))
    )
    merge_airr_chains(adata1, adata_bracer)
    obs_merged = adata1.obs
    _normalize_df_types(obs_merged)
    _normalize_df_types(obs_expected)
    pdt.assert_frame_equal(
        # extra chains will be different (since the two DFs have different keys)
        obs_expected.drop("extra_chains", axis=1),
        obs_merged.drop("extra_chains", axis=1),
        check_dtype=False,
        check_column_type=False,
        check_categorical=False,
    )


@pytest.mark.parametrize("is_cell,result", [("None", None), ("foo", ValueError)])
def test_merge_airr_chains_no_ir(is_cell, result: BaseException):
    """Test that merging an IR anndata with a non-IR anndata
    also works with `merge_airr_chains`.

    When a cell-level attribute (`is_cell`) is present, and an inconsistent value
    gets merged, the function should fail with a ValueError. However, if `is_cell` is
    "None", merging should still be possible.
    """
    cell_ids_anndata = np.array(
        [
            "AAACCTGAGATAGCAT-1",
            "AAACCTGAGTACGCCC-1",
            "AAACCTGCATAGAAAC-1",
            "AAACCTGGTCCGTTAA-1",
            "AAACTTGGTCCGTTAA-1",
            "cell_without_tcr",
        ]
    )
    # X with 6 cells and 2 genes
    adata = AnnData(X=np.ones((6, 2)))
    adata.obs_names = cell_ids_anndata
    adata.obs["foo"] = "bar"
    adata.obs["is_cell"] = is_cell
    adata_ir = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata_ir.obs["foo_ir"] = "bar_ir"

    if result is not None:
        with pytest.raises(result):
            merge_airr_chains(adata, adata_ir)
    else:
        merge_airr_chains(adata, adata_ir)

        npt.assert_array_equal(adata.obs.index, cell_ids_anndata)
        assert np.all(np.isin(adata_ir.obs.columns, adata.obs.columns))
        assert "foo" in adata.obs.columns
        assert "foo_ir" in adata.obs.columns
        assert list(adata.obs_names) == list(cell_ids_anndata)


def test_merge_adata():
    """Test that merging of an IR anndata with a non-IR anndata works
    with `merge_with_ir`"""
    cell_ids_anndata = np.array(
        [
            "AAACCTGAGATAGCAT-1",
            "AAACCTGAGTACGCCC-1",
            "AAACCTGCATAGAAAC-1",
            "AAACCTGGTCCGTTAA-1",
            "AAACTTGGTCCGTTAA-1",
            "cell_without_tcr",
        ]
    )
    # X with 6 cells and 2 genes
    adata = AnnData(X=np.ones((6, 2)))
    adata.obs_names = cell_ids_anndata
    adata.obs["foo"] = "bar"
    adata_ir = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata_ir.obs["foo_ir"] = "bar_ir"

    merge_with_ir(adata, adata_ir)

    npt.assert_array_equal(adata.obs.index, cell_ids_anndata)
    assert np.all(np.isin(adata_ir.obs.columns, adata.obs.columns))
    assert "foo" in adata.obs.columns
    assert "foo_ir" in adata.obs.columns
    assert list(adata.obs_names) == list(cell_ids_anndata)

    # Check that an additional merge raises a ValueError (use merge by chain!)
    with pytest.raises(ValueError):
        merge_with_ir(adata, adata_ir, on=["foo_ir"])
