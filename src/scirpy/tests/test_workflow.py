"""Test entire workflows using small example datasets.

The workflow is additionally tested using a larger dataset
by running the tutorial on the CI.

Currently, these tests are mainly designed to be ran on the
Bioconda-CI when building the container as a quick consistency check.
The tests need to be quick in order not to overload the bioconda CI,
but AIRR-compliance mandates to have these tests.
"""

import pandas as pd
import pandas.testing as pdt
import pytest
import scanpy as sc

import scirpy as ir

from . import TESTDATA
from .util import _normalize_df_types


def _pd_versioned_pkl(path):
    """Resolve a .pkl.gz path to a pandas-major-version-specific variant.

    E.g. ``foo.pkl.gz`` -> ``foo.pd3.pkl.gz`` when pandas 3.x is installed.
    """
    major = pd.__version__.split(".")[0]
    path = str(path)
    assert path.endswith(".pkl.gz"), path
    return path[: -len(".pkl.gz")] + f".pd{major}.pkl.gz"


@pytest.mark.conda
@pytest.mark.parametrize("save_intermediates", [False, True])
@pytest.mark.parametrize(
    "adata_path,upgrade_schema,obs_expected",
    [
        (
            TESTDATA / "10x/vdj_nextgem_hs_pbmc3_t_filtered_contig_annotations.csv.gz",
            False,
            TESTDATA / "test_workflow/adata_10x_pbmc3_t.obs.expected.pkl.gz",
        ),
        (
            TESTDATA / "wu2020_200_v0_11.h5ad",
            True,
            TESTDATA / "test_workflow/adata_wu_200_old_schema.obs.expected.pkl.gz",
        ),
    ],
)
def test_workflow(adata_path, save_intermediates, upgrade_schema, obs_expected, tmp_path):
    def _save_and_load(adata):
        """If save_intermediates is True, save the anndata to a temporary location
        and re-load it from disk.
        """
        if save_intermediates:
            adata.write_h5ad(tmp_path / "tmp_adata.h5ad")
            return sc.read_h5ad(tmp_path / "tmp_adata.h5ad")
        else:
            return adata

    if upgrade_schema:
        adata = sc.read_h5ad(adata_path)
        ir.io.upgrade_schema(adata)
    else:
        adata = ir.io.read_10x_vdj(adata_path, include_fields=None)

    adata_obs_expected = pd.read_pickle(_pd_versioned_pkl(obs_expected))

    ir.tl.chain_qc(adata)
    adata = _save_and_load(adata)
    ir.pp.ir_dist(adata)
    adata = _save_and_load(adata)
    ir.tl.define_clonotypes(adata)
    adata = _save_and_load(adata)
    ir.tl.clonotype_network(adata)
    adata = _save_and_load(adata)
    ir.tl.clonal_expansion(adata)
    adata = _save_and_load(adata)
    ir.pl.clonotype_network(adata)
    adata = _save_and_load(adata)

    # turn nans into consistent value (nan)
    _normalize_df_types(adata.obs)

    # # Use this code to re-generate the "expected file", if necessary.
    # adata.obs.to_pickle(_pd_versioned_pkl(obs_expected), protocol=4)

    pdt.assert_frame_equal(adata.obs, adata_obs_expected, check_dtype=False, check_categorical=False)


def test_workflow_mudata_non_overlapping_gex_airr():
    """Test if the workflow still works when gene expression and AIRR data do not have exactly the same dimension.

    Testdataset was created as

    >>> mdata = ir.datasets.wu2020_3k()
    >>> mdata = MuData({"gex": mdata["gex"][50:200], "airr": mdata["airr"][:150]})
    """
    mdata = ir.io.read_h5mu(TESTDATA / "wu2020_200.h5mu")

    ir.tl.chain_qc(mdata)
    ir.pp.ir_dist(mdata)
    ir.tl.define_clonotypes(mdata)
    ir.tl.clonotype_network(mdata)
    ir.tl.clonal_expansion(mdata)
    ir.pl.clonotype_network(mdata)
