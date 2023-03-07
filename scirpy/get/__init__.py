import itertools
from contextlib import contextmanager
from enum import Enum, auto
from typing import Literal, Sequence, Union, cast

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from ..util import _ParamsCheck

_VALID_CHAINS = ["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]
ChainType = Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]


@_ParamsCheck.inject_param_docs()
def airr(
    adata: _ParamsCheck.TYPE,
    airr_variable: Union[str, Sequence[str]],
    chain: Union[ChainType, Sequence[ChainType]],
    *,
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
) -> Union[pd.Series, pd.DataFrame]:
    """\
    Retrieve AIRR variables for each cell, given a specific chain.

    Parameters
    ----------
    {adata}
    airr_variable
        One or multiple columns from the AIRR Rearrangment schema (see adata.var).
        If multiple values are specified, a dataframe will be returned.
    chain
        choose the recptor arm (VJ/VDJ) and if you want to retrieve the primary or secondary chain.
        If multiple chains are specified, a adataframe will be returned
    {airr_mod}
    {airr_key}
    {chain_idx_key}

    Returns
    -------
    a pandas series or dataframe aligned to adata.obs. Contains missing values in places where a cell
    does not have the requested chain.
    """
    params = _ParamsCheck(adata, airr_mod, airr_key, chain_idx_key)
    multiple_vars = not isinstance(airr_variable, str)
    multiple_chains = not isinstance(chain, str)

    if multiple_vars or multiple_chains:
        if not multiple_vars:
            airr_variable = [airr_variable]
        if not multiple_chains:
            chain = [chain]  # type: ignore
        return pd.DataFrame(
            {
                f"{tmp_chain}_{tmp_var}": _airr_col(
                    params.airr, params.chain_indices, tmp_var, tmp_chain
                )
                for tmp_chain, tmp_var in itertools.product(chain, airr_variable)
            },
            index=params.adata.obs_names,
        )
    else:
        return pd.Series(
            _airr_col(params.airr, params.chain_indices, airr_variable, chain),
            index=params.adata.obs_names,
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

    idx = chain_indices[:, receptor_arm, chain_i]
    mask = ~ak.to_numpy(ak.is_none(idx))

    result = np.full((len(idx),), fill_value=None, dtype=object)

    # to_numpy would be faster, but it doesn't work with strings (as this would create an object dtype
    # which is not allowed as per the awkward documentation)
    # Currently the performance hit doesn't seem to be a deal breaker, can maybe revisit this in the future.
    # It is anyway not very efficient to create a result array with an object dtype.
    _ak_slice = airr_data[
        np.where(mask)[0],
        airr_variable,
        # astype(int) is required if idx[mask] is an empty array of unknown type.
        ak.to_numpy(idx[mask], allow_missing=False).astype(int),
    ]
    result[mask] = ak.to_list(_ak_slice)
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
        (ak.count(chain_idx["VJ"], axis=1) + ak.count(chain_idx["VDJ"], axis=1)) > 0
    )
