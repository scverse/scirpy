import itertools
from typing import Sequence, Literal, Union, cast
import pandas as pd
import numpy as np
import awkward as ak
from anndata import AnnData


def airr(
    adata: AnnData,
    airr_variable: Union[str, Sequence[str]],
    chain: Union[
        Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"],
        Sequence[Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]],
    ],
    *,
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
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

    airr_data = cast(ak.Array, adata.obsm[airr_key])
    chain_indices = cast(pd.DataFrame, adata.obsm[chain_idx_key])

    if multiple_vars or multiple_chains:
        if not multiple_vars:
            airr_variable = [airr_variable]
        if not multiple_chains:
            chain = [chain]  # type: ignore
        return pd.DataFrame(
            {
                f"{tmp_chain}_{tmp_var}": _airr_col(
                    airr_data, chain_indices, tmp_var, tmp_chain
                )
                for tmp_chain, tmp_var in itertools.product(chain, airr_variable)
            },
            index=adata.obs_names,
        )
    else:
        return pd.Series(
            _airr_col(airr_data, chain_indices, airr_variable, chain),
            index=adata.obs_names,
        )


def _airr_col(
    airr_data: ak.Array,
    chain_indices: pd.DataFrame,
    airr_variable: str,
    chain: str,
) -> np.ndarray:
    """called by `airr()` to retrieve a single column"""
    chain = chain.upper()
    if chain not in chain_indices.columns:
        raise ValueError(
            f"Invalid value for chain. Valid values are {', '.join(chain_indices.columns)}"
        )

    idx = chain_indices[chain]
    mask = ~pd.isnull(idx)

    # TODO ensure that this doesn't get converted to something not supporting missing values
    # when saving anndata
    result = np.full(idx.shape, fill_value=None, dtype=object)

    # to_numpy would be faster, but it doesn't work with strings (as this would create an object dtype
    # which is not allowed as per the awkward documentation)
    # Currently the performance hit doesn't seem to be a deal breaker, can maybe revisit this in the future.
    # It is anyway not very efficient to create a result array with an object dtype.
    _ak_slice = airr_data[np.where(mask)[0], airr_variable, idx[mask].astype(int)]
    result[mask] = ak.to_list(_ak_slice)
    return result


def most_frequent(array: Sequence, n=10):
    """Get the most frequent categories of an Array"""
    return pd.Series(array).value_counts().index[:n].tolist()


def obs_context(adata, **kwargs):
    """A context manager that temporarily adds columns to adata.obs"""
    # TODO
    raise NotImplementedError
