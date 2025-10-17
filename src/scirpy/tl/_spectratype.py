from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from scirpy.get import airr as get_airr
from scirpy.util import DataHandler

from ._group_abundance import _group_abundance


@DataHandler.inject_param_docs()
def spectratype(
    adata: DataHandler.TYPE,
    chain: Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"] | Sequence[Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"]] = "VJ_1",
    *,
    target_col: str,
    cdr3_col: str = "junction_aa",
    combine_fun: Callable = np.sum,
    fraction: None | str | bool = None,
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    **kwargs,
) -> pd.DataFrame:
    """\
    Summarizes the distribution of :term:`CDR3` region lengths.

    Ignores NaN values.

    Parameters
    ----------
    {adata}
    chain
        One or multiple chains from which to use CDR3 sequences
    target_col
        Color by this column from `obs`. E.g. sample or diagnosis
    cdr3_col
        AIRR rearrangement column from which sequences are obtained
    combine_fun
        A function definining how the groupby columns should be merged
        (e.g. sum, mean, median, etc).
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized.
    {airr_mod}
    {airr_key}
    {chain_idx_key}


    Returns
    -------
    A DataFrame with spectratype information.
    """
    if "groupby" in kwargs or "IR_" in str(cdr3_col) or "IR_" in str(chain):
        raise ValueError(
            """\
            The function signature has been updated when the scirpy 0.13 datastructure was introduced.
            Please use the `chain` attribute to choose `VJ_1`, `VDJ_1`, `VJ_2`, or `VDJ_2` chain(s).
            """
        )
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)

    # Get airr and remove NAs
    airr_df = get_airr(params, [cdr3_col], chain).dropna(how="any")
    obs_cols = [target_col] if not isinstance(fraction, str) else [target_col, fraction]
    obs = params.get_obs(obs_cols)

    # Combine (potentially) multiple length columns into one
    obs["lengths"] = airr_df.applymap(len).apply(combine_fun, axis=1)

    cdr3_lengths = _group_abundance(obs, groupby="lengths", target_col=target_col, fraction=fraction)

    # Should include all lengths, not just the abundant ones
    cdr3_lengths = cdr3_lengths.reindex(range(int(obs["lengths"].max()) + 1)).fillna(value=0.0)

    cdr3_lengths.sort_index(axis=1, inplace=True)

    return cdr3_lengths
