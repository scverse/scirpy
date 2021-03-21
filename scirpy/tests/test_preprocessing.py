from os import read
from .util import _normalize_df_types
from anndata import AnnData
import numpy as np
from ..io import read_10x_vdj, read_bracer
from . import TESTDATA
from .._preprocessing import merge_with_ir
from .._preprocessing._merge_adata import _merge_ir_obs
import pandas.testing as pdt
import numpy.testing as npt
from ..util import _is_na
import pandas as pd


def test_merge_ir_obs():
    adata1 = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    adata2 = adata1.copy()
    obs_merged = _merge_ir_obs(adata1, adata2)
    pdt.assert_frame_equal(adata1.obs, obs_merged)

    adata_bracer = read_bracer(TESTDATA / "bracer/changeodb.tab")
    obs_merged = _merge_ir_obs(adata1, adata_bracer)
    obs_expected = pd.concat([adata1.obs, adata_bracer.obs])
    _normalize_df_types(obs_merged)
    _normalize_df_types(obs_expected)
    pdt.assert_frame_equal(
        obs_expected,
        obs_merged,
        check_dtype=False,
        check_column_type=False,
        check_categorical=False,
    )


def test_merge_adata():
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
    assert np.all(np.isin(IR_OBS_COLS, adata.obs.columns))
    assert "foo" in adata.obs.columns
    assert "foo_ir" in adata.obs.columns

    # Check that an additional merge (will trigger merge by chains)
    # results in the same object.
    obs_after_first_merge = adata.obs.copy()
    merge_with_ir(adata, adata_ir, on=["foo_ir"])

    # we don't care about the order of the columns
    obs_after_first_merge.sort_index(axis="columns", inplace=True)
    adata.obs.sort_index(axis="columns", inplace=True)

    # there's also still the problem with different representations of 'nan' values
    # which we ignore for now.
    # turn nans into consistent value (nan)
    _normalize_df_types(adata.obs)
    _normalize_df_types(obs_after_first_merge)

    pdt.assert_frame_equal(
        obs_after_first_merge,
        adata.obs,
        check_dtype=False,
        check_column_type=False,
        check_categorical=False,
    )
