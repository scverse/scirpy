from anndata import AnnData
from typing import Callable, Union, Collection
import numpy as np
import pandas as pd
from ._group_abundance import _group_abundance
from ..util import _is_na
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def spectratype(
    adata: AnnData,
    groupby: Union[str, Collection[str]] = "IR_VJ_1_junction_aa",
    *,
    target_col: str,
    combine_fun: Callable = np.sum,
    fraction: Union[None, str, bool] = None,
) -> pd.DataFrame:
    """Summarizes the distribution of :term:`CDR3` region lengths.

    Ignores NaN values.

    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Column(s) containing CDR3 sequences.
    target_col
        Color by this column from `obs`. E.g. sample or diagnosis
    combine_fun
        A function definining how the groupby columns should be merged
        (e.g. sum, mean, median, etc).
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized.


    Returns
    -------
    A DataFrame with spectratype information.
    """
    if len(np.intersect1d(adata.obs.columns, groupby)) < 1:
        raise ValueError(
            "`groupby` not found in obs. Where do you store CDR3 length information?"
        )

    if isinstance(groupby, str):
        groupby = [groupby]
    else:
        groupby = list(set(groupby))

    # Remove NAs
    ir_obs = adata.obs.loc[~np.any(_is_na(adata.obs[groupby].values), axis=1), :].copy()

    # Combine (potentially) multiple length columns into one
    ir_obs["lengths"] = ir_obs.loc[:, groupby].applymap(len).apply(combine_fun, axis=1)

    cdr3_lengths = _group_abundance(
        ir_obs, groupby="lengths", target_col=target_col, fraction=fraction
    )

    # Should include all lengths, not just the abundant ones
    cdr3_lengths = cdr3_lengths.reindex(range(int(ir_obs["lengths"].max()) + 1)).fillna(
        value=0.0
    )

    cdr3_lengths.sort_index(axis=1, inplace=True)

    return cdr3_lengths
