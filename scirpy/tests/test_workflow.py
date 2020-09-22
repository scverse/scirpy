"""Test entire workflows using small example datasets.

The workflow is additionally tested using a larger dataset
by running the tutorial on the CI.

Currently, these tests are mainly designed to be ran on the
Bioconda-CI when building the container as a quick consistency check.
The tests need to be quick in order not to overload the bioconda CI,
but AIRR-compliance mandates to have these tests.
"""
from . import TESTDATA
import scirpy as ir
import pytest
import pandas.testing as pdt
import pandas as pd
from scirpy.util import _is_na
import numpy as np


@pytest.mark.conda
def test_workflow():
    adata = ir.io.read_10x_vdj(
        TESTDATA / "10x/vdj_nextgem_hs_pbmc3_t_filtered_contig_annotations.csv.gz"
    )
    adata_obs_expected = pd.read_pickle(
        TESTDATA / "test_workflow/adata.obs.expected.pkl.gz"
    )
    ir.tl.chain_pairing(adata)
    ir.pp.tcr_neighbors(adata)
    ir.tl.define_clonotypes(adata)
    ir.tl.clonotype_network(adata)
    ir.tl.clonal_expansion(adata)

    ir.pl.clonotype_network(adata)

    # turn nans into consistent value (nan)
    for col in adata.obs.columns:
        adata.obs.loc[_is_na(adata.obs[col]), col] = np.nan

    # # Use this code to re-generate the "expected file", if necessary.
    # adata.obs.to_pickle(
    #     "tests/data/test_workflow/adata.obs.expected.pkl.gz", protocol=4
    # )

    pdt.assert_frame_equal(
        adata.obs, adata_obs_expected, check_dtype=False, check_categorical=False
    )
