from anndata import AnnData
from typing import Callable, Union, Collection
import numpy as np
import pandas as pd
from .._util import _is_na, _normalize_counts


def spectratype(
    adata: AnnData,
    groupby: str,
    *,
    target_col: Union[str, Collection] = "TRA_1_cdr3_len",
    combine_fun: Callable = np.sum,
    fraction: Union[None, str, bool] = None,
) -> pd.DataFrame:
    """Show the distribution of CDR3 region lengths. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Columns containing CDR3 lengths.        
    combine_fun
        A function definining how the target columns should be merged 
        (e.g. sum, mean, median, etc).  
    fraction
        If True, compute fractions of expanded clonotypes rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name 
        of a grouping (e.g. samples). 


    Returns
    -------
    A DataFrame with spectratype information. 
    """
    if len(np.intersect1d(adata.obs.columns, target_col)) < 1:
        raise ValueError(
            "`target_col` not found in obs. Where do you store CDR3 length information?"
        )

    if isinstance(target_col, str):
        target_col = [target_col]
    else:
        target_col = list(set(target_col))
    tcr_obs = adata.obs.copy()
    tcr_obs["lengths"] = tcr_obs.loc[:, target_col].apply(combine_fun, axis=1)

    na_mask = _is_na(tcr_obs[groupby]) | _is_na(tcr_obs["lengths"])
    tcr_obs = tcr_obs.loc[~na_mask, :]

    # Calculate distribution of lengths in each group
    cdr3_lengths = (
        tcr_obs.groupby([groupby, "lengths"]).size().reset_index(name="count")
    )

    # normalize to fractions
    scale_vector = _normalize_counts(tcr_obs, normalize=fraction, default_col=groupby)
    cdr3_lengths["count"] /= scale_vector

    cdr3_lengths = cdr3_lengths.pivot(index="lengths", columns=groupby, values="count")
    cdr3_lengths = cdr3_lengths.reindex(
        range(int(tcr_obs["lengths"].max()) + 1)
    ).fillna(value=0.0)

    # By default, the most abundant group should be the first on the plot,
    # therefore we need their order
    cdr3_lengths[cdr3_lengths.apply(np.sum, axis=0).index.values]

    return cdr3_lengths
