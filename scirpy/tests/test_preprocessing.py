from os import read
from .util import _normalize_df_types
from anndata import AnnData
import numpy as np
from ..io import read_10x_vdj, read_bracer
from . import TESTDATA
from .._preprocessing import merge_with_ir
from .._preprocessing._merge_adata import merge_airr_chains
import pandas.testing as pdt
import numpy.testing as npt
from ..util import _is_na
import pandas as pd
import pytest


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
        obs_expected,
        obs_merged,
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
