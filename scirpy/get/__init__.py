import itertools
from contextlib import contextmanager
from enum import Enum, auto
from typing import Literal, Sequence, Union, cast

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData

_VALID_CHAINS = ["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]
ChainType = Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]


def airr(
    adata: AnnData,
    airr_variable: Union[str, Sequence[str]],
    chain: Union[ChainType, Sequence[ChainType]],
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
    airr_key
        Key in `adata.obsm` under which the AwkwardArray with AIRR information is stored
    chain_idx_key
        Key in `adata.obsm` under which the chain indices are stored

    Returns
    -------
    a pandas series or dataframe aligned to adata.obs. Contains missing values in places where a cell
    does not have the requested chain.
    """
    multiple_vars = not isinstance(airr_variable, str)
    multiple_chains = not isinstance(chain, str)

    airr_data = cast(ak.Array, adata.obsm[airr_key])
    chain_indices = cast(ak.Array, adata.obsm[chain_idx_key])

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
    chain_indices: ak.Array,
    airr_variable: str,
    chain: ChainType,
) -> np.ndarray:
    """called by `airr()` to retrieve a single column"""
    chain = chain.upper()
    if chain not in _VALID_CHAINS:
        raise ValueError(
            f"Invalid value for chain. Valid values are {', '.join(_VALID_CHAINS)}"
        )

    # split VJ_1 into ("VJ", 0)
    receptor_arm, chain_i = chain.split("_")
    chain_i = int(chain_i) - 1

    # require -1 instead of None for indexing
    idx = ak.fill_none(chain_indices[:, receptor_arm, chain_i], -1)

    # first, select the variable and pad it with None, such that we can index
    # we need to pad with the maximum value from idx, but at least 1
    padded = ak.pad_none(airr_data[:, airr_variable], max(np.max(idx), 1))

    # to_numpy would be faster, but it doesn't work with strings (as this would create an object dtype
    # which is not allowed as per the awkward documentation)
    # Currently the performance hit doesn't seem to be a deal breaker, can maybe revisit this in the future.
    # It is anyway not very efficient to create a result array with an object dtype.
    result = np.array(ak.to_list(padded[np.arange(len(idx)), idx]), dtype=object)

    return result


# TODO #356: do we want this?
# def obs_context(adata, **kwargs):
#     """A context manager that temporarily adds columns to adata.obs"""
#     raise NotImplementedError
#
# def most_frequent(array: Sequence, n=10):
#     """Get the most frequent categories of an Array"""
#     return pd.Series(array).value_counts().index[:n].tolist()


@contextmanager
def _obs_context(adata, **kwargs):
    """Temporarily add columns to adata.obs"""
    orig_obs = adata.obs.copy()
    adata.obs = adata.obs.assign(**kwargs)
    try:
        yield adata
    finally:
        adata.obs = orig_obs


def _has_ir(adata, chain_idx_key="chain_indices"):
    """Return a mask of all cells that have a valid IR configuration"""
    chain_idx = adata.obsm[chain_idx_key]
    return ak.to_numpy(
        (ak.sum(chain_idx["VJ"], axis=1) + ak.sum(chain_idx["VDJ"], axis=1)) > 0
    )
