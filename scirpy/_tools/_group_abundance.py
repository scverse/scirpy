from anndata import AnnData
from typing import Union
import numpy as np
import pandas as pd
from .._util import _is_na, _normalize_counts
from typing import Sequence
from .._compat import Literal


def _group_abundance(
    tcr_obs: pd.DataFrame,
    groupby: str,
    target_col: str,
    *,
    fraction: Union[None, str, bool] = None,
    sort: Union[Literal["count", "alphabetical"], Sequence[str]] = "count",
) -> pd.DataFrame:
    # remove NA rows
    na_mask = _is_na(tcr_obs[groupby]) | _is_na(tcr_obs[target_col])
    tcr_obs = tcr_obs.loc[~na_mask, :]

    # normalize to fractions
    scale_vector = _normalize_counts(tcr_obs, normalize=fraction, default_col=groupby)
    tcr_obs = tcr_obs.assign(count=1, weight=scale_vector)

    # Calculate distribution of lengths in each group. Use sum instead of count
    # to reflect weights
    group_counts = (
        tcr_obs.groupby([groupby, target_col])["count", "weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "weighted_count"})
    )

    result_df = group_counts.pivot(
        index=groupby, columns=target_col, values="weighted_count"
    ).fillna(value=0.0)

    # required that we can still sort by abundance even if normalized
    result_df_count = group_counts.pivot(
        index=groupby, columns=target_col, values="count"
    ).fillna(value=0.0)

    # By default, the most abundant group should be the first on the plot,
    # therefore we need their order
    if isinstance(sort, str) and sort == "alphabetical":
        ranked_target = sorted(result_df.index)
    elif isinstance(sort, str) and sort == "count":
        ranked_target = (
            result_df_count.apply(np.sum, axis=1)
            .sort_values(ascending=False)
            .index.values
        )
    else:
        ranked_target = sort

    ranked_groups = (
        result_df_count.apply(np.sum, axis=0).sort_values(ascending=False).index.values
    )
    result_df = result_df.loc[ranked_target, ranked_groups]

    return result_df


def group_abundance(
    adata: AnnData,
    groupby: str,
    target_col: str = "has_tcr",
    *,
    fraction: Union[None, str, bool] = None,
    sort: Union[Literal["count", "alphabetical"], Sequence[str]] = "count",
) -> pd.DataFrame:
    """Creates summary statsitics on how many
    cells belong to a certain category within a certain group. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. E.g, sample, or group. 
    target_col
        Caregorical variable from `obs` according to which the abundance/fractions
        will be computed.        
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized. 
    sort
        How to arrange the dataframe columns. 
        Default is by the category count ("count"). 
        Other options are "alphabetical" or to provide a list of column names.
        By providing an explicit list, the DataFrame can also be subsetted to
        specific categories. 

    Returns
    -------
    Returns a data frame with the number of cells per group 
    """
    if target_col not in adata.obs.columns:
        raise ValueError("`target_col` not found in obs`")

    tcr_obs = adata.obs

    return _group_abundance(
        tcr_obs, groupby, target_col=target_col, fraction=fraction, sort=sort
    )
