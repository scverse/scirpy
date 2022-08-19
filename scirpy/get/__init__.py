from typing import Sequence, Literal
import pandas as pd
import numpy as np
import awkward._v2 as ak


def airr(
    adata,
    airr_variable,
    chain: Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"],
) -> pd.Series:
    """Retrieve AIRR variables for each cell, given a specific chain.

    Parameters
    ----------
    adata
        scirpy AnnData object
    airr_variable
        a column from the AIRR Rearrangment schema (see adata.var)
    chain
        choose the recptor arm (VJ/VDJ) and if you want to retrieve the primary or secondary chain.

    Returns
    -------
    a pandas series aligned to adata.obs. Contains missing values in places where a cell
    does not have the requested chain.
    """
    chain = chain.upper()
    valid_chains = ["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]
    if chain not in valid_chains:
        raise ValueError(
            f"Invalid value for chain. Valid values are {', '.join(valid_chains)}"
        )
    if airr_variable not in adata.var_names:
        raise ValueError("airr_variable is not in adata.var_names")

    idx = adata.obsm["chain_indices"][chain]
    mask = ~pd.isnull(idx)

    # TODO ensure that this doesn't get converted to something not supporting missing values
    # when saving anndata
    result = np.full(idx.shape, fill_value=None, dtype=object)

    # to_numpy fails with some dtypes and missing values are only supported through masked arrays
    # TODO to_numpy would surely be faster than to_list. Maybe this can be updated in the future.
    result[mask] = ak.to_list(
        adata.X[
            np.where(mask)[0],
            np.where(adata.var_names == airr_variable)[0],
            idx[mask].astype(int),
        ],
    )
    return pd.Series(result, index=adata.obs_names)


def most_frequent(array: Sequence, n=10):
    """Get the most frequnet categories of an Array"""
    return pd.Series(array).value_counts().index[:n].tolist()


def obs_context(adata, **kwargs):
    """A context manager that temporarily adds columns to adata.obs"""
    pass


def add_to_obs(adata, **kwargs):
    """Add columnt to obs"""
    pass
