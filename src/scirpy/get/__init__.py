import itertools
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, Literal, Union, cast, overload

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from scirpy.util import DataHandler

_VALID_CHAINS = ["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]
ChainType = Literal["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]


@DataHandler.inject_param_docs()
def airr(
    adata: DataHandler.TYPE,
    airr_variable: str | Sequence[str],
    chain: ChainType | Sequence[ChainType] = ("VJ_1", "VDJ_1", "VJ_2", "VDJ_2"),
    *,
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
) -> pd.DataFrame | pd.Series:
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
    A :class:`~pandas.Series` or :class:`~pandas.DataFrame` aligned to `adata.obs`.
    Contains missing values in places where a cell does not have the requested chain.
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
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
                    params.airr,
                    params.chain_indices,
                    tmp_var,
                    cast(ChainType, tmp_chain),
                )
                for tmp_chain, tmp_var in itertools.product(chain, airr_variable)
            },
            index=params.adata.obs_names,
        )
    else:
        return pd.Series(
            _airr_col(params.airr, params.chain_indices, airr_variable, cast(ChainType, chain)),
            index=params.adata.obs_names,
        )


def _airr_col(
    airr_data: ak.Array,
    chain_indices: ak.Array,
    airr_variable: str,
    chain: ChainType,
) -> np.ndarray:
    """Called by `airr()` to retrieve a single column"""
    chain = chain.upper()  # type: ignore
    if chain not in _VALID_CHAINS:
        raise ValueError(f"Invalid value for chain. Valid values are {', '.join(_VALID_CHAINS)}")

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


@contextmanager
def obs_context(data: AnnData | MuData, temp_cols: pd.DataFrame | Mapping[str, Any]):
    """
    Contextmanager that temporarily adds columns to obs.

    Example
    -------

    .. code-block:: python

        with ir.get.obs_context(
            mdata,
            {
                "new_col_with_constant_value": "foo",
                "new_col_with_sequence": range(len(mdata)),
                "v_gene_primary_vj_chain": ir.get.airr(mdata, "v_call", "VJ_1"),
            },
        ) as m:
            ir.pl.group_abundance(m, groupby="v_gene_primary_vj_chain", target_col="new_col_with_constant_value")


    Parameters
    ----------
    data
        AnnData or MuData object
    temp_cols
        Dictionary where keys are column names and values are columns. Columns will be added
        to obs using :py:meth:`pandas.DataFrame.assign`. It is also possible to pass a :class:`~pandas.DataFrame` in
        which case the columns of the DataFrame will be added to `obs` and matched based on the index.
    """
    orig_obs = data.obs.copy()
    data.obs = data.obs.assign(**cast(Mapping, temp_cols))
    data.strings_to_categoricals()
    try:
        yield data
    finally:
        data.obs = orig_obs


@DataHandler.inject_param_docs()
def airr_context(
    data: DataHandler.TYPE,
    airr_variable: str | Sequence[str],
    chain: ChainType | Sequence[ChainType] = ("VJ_1", "VDJ_1", "VJ_2", "VDJ_2"),
    *,
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
):
    """\
    Contextmanager that temporarily adds AIRR information to obs.

    This is essentially a wrapper around :func:`~scirpy.get.obs_context` and equivalent to

    .. code-block:: python

        ir.get.obs_context(data, ir.get.airr(airr_variable, chain))

    Example
    -------

    To list all combinations of patient and V genes in the :term:`VJ <V(D)J>` and :term:`VDJ <V(D)J>` chains:

    .. code-block:: python

        with ir.get.airr_context(mdata, "v_call", chain=["VJ_1", "VDJ_1"]) as m:
            combinations = (
                m.obs.groupby(["patient", "VJ_1_v_call", "VDJ_1_v_call"], observed=True)
                .size()
                .reset_index(name="n")
            )

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
    """
    dh = DataHandler(data, airr_mod, airr_key, chain_idx_key)
    # ensure this is a list and `get.airr` always returns a data frame
    if isinstance(chain, str):
        chain = cast(Sequence[ChainType], [chain])
    return obs_context(dh.data, cast(pd.DataFrame, airr(dh, airr_variable, chain)))


def _has_ir(params: DataHandler):
    """Return a mask of all cells that have a valid IR configuration"""
    return (
        ak.to_numpy(ak.count(params.chain_indices["VJ"], axis=1))
        + ak.to_numpy(ak.count(params.chain_indices["VDJ"], axis=1))
    ) > 0
