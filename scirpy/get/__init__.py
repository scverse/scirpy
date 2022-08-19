import itertools
from typing import Sequence, Literal, Union
import pandas as pd
import numpy as np
import awkward._v2 as ak
from anndata import AnnData


def airr(
    adata: AnnData,
    airr_variable: Union[str, Sequence[str]],
    chain: Union[
        Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"],
        Sequence[Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]],
    ],
) -> Union[pd.Series, pd.DataFrame]:
    """Retrieve AIRR variables for each cell, given a specific chain.

    Parameters
    ----------
    adata
        scirpy AnnData object
    airr_variable
        One or multiple columns from the AIRR Rearrangment schema (see adata.var).
        If multiple values are specified, a dataframe will be returned.
    chain
        choose the recptor arm (VJ/VDJ) and if you want to retrieve the primary or secondary chain.
        If multiple chains are specified, a adataframe will be returned

    Returns
    -------
    a pandas series or dataframe aligned to adata.obs. Contains missing values in places where a cell
    does not have the requested chain.
    """
    multiple_vars = not isinstance(airr_variable, str)
    multiple_chains = not isinstance(chain, str)

    if multiple_vars or multiple_chains:
        if not multiple_vars:
            airr_variable = [airr_variable]
        if not multiple_chains:
            chain = [chain]
        return pd.DataFrame(
            {
                f"{tmp_chain}_{tmp_var}": _airr_col(adata, tmp_var, tmp_chain)
                for tmp_chain, tmp_var in itertools.product(chain, airr_variable)
            }
        )
    else:
        return _airr_col(adata, airr_variable, chain)


def _airr_col(
    adata: AnnData,
    airr_variable: str,
    chain: Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"],
) -> pd.Series:
    """called by `airr()` to retrieve a single column"""
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
    raise NotImplementedError


def add_to_obs(adata, **kwargs):
    """Add columnt to obs"""
    raise NotImplementedError
