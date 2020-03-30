from .. import tl
from anndata import AnnData
from . import base
from .._util import _normalize_counts, _is_na
from typing import Union


def _prepare_df(adata, groupby, target_col, clip_at, fraction, groupby_count):
    """Turn the result of the `clip_and_count` tool into a plottable
    dataframe
    
    groupby_count allows to set a different groupby parameter
    for the clip_and_count function. Useful for clonal expansion. 
    """
    tmp_col = target_col + "clipped_count"
    tmp_col_weight = target_col + "weight"

    obs = adata.obs.loc[:, [groupby, target_col]]
    obs[tmp_col] = tl.clip_and_count(
        adata, target_col, groupby=groupby_count, clip_at=clip_at, inplace=False
    )
    # filter NA values
    obs = obs.loc[~_is_na(obs[target_col]), :]

    # add normalization vector
    size_vector = _normalize_counts(obs, fraction, groupby)
    obs[tmp_col_weight] = size_vector

    obs = (
        obs.groupby([groupby, tmp_col], observed=True)[tmp_col_weight]
        .sum()
        .reset_index()
        .pivot(index=groupby, columns=tmp_col, values=tmp_col_weight)
        .fillna(0)
    )

    return obs


def clip_and_count(
    adata: AnnData,
    groupby: str,
    target_col: str,
    *,
    clip_at: int = 3,
    fraction: bool = True,
    **kwargs,
):
    """Plots the the *number of cells* in `target_col` that fall into 
    a certain count bin for each group in `group_by`. 

    Removes all entries with `NaN` in `target_col` prior to plotting. 

    Parameters
    ----------
    adata
        AnnData object to work on
    groupby
        Group by this column from `obs`
    target_col
        Column to count on.
    clip_at
        All entries in `target_col` with more copies than `clip_at`
        will be summarized into a single group.         
    fraction
        If True, compute fractions rather than reporting
        abosolute numbers.
    **kwargs
        Additional arguments passed to :meth:`base.bar`
    """
    plot_df = _prepare_df(adata, groupby, target_col, clip_at, fraction, groupby)

    return base.bar(plot_df, **kwargs)


def clonal_expansion(
    adata: AnnData,
    groupby: str,
    target_col: str = "clonotype",
    *,
    expanded_in: Union[str, None] = None,
    clip_at: int = 3,
    fraction: bool = True,
    **kwargs,
):
    """Plots the the *number of cells* in `target_col` that fall into 
    a certain count bin for each group in `group_by`. 

    Removes all entries with `NaN` in `target_col` prior to plotting. 

    Parameters
    ----------
    adata
        AnnData object to work on
    groupby
        Group by this column from `obs`
    target_col
        Column to count on. Default to clonotype. 
    expanded_in
        Calculate clonal expansion within groups. Usually makes sense to set
        this to the column containing sample annotation. If set to None, 
        a clonotype counts as expanded if there's any cell of the same clonotype
        across the entire dataset. 
    clip_at
        All entries in `target_col` with more copies than `clip_at`
        will be summarized into a single group.         
    fraction
        If True, compute fractions rather than reporting
        abosolute numbers.
    **kwargs
        Additional arguments passed to :meth:`base.bar`
    """
    plot_df = _prepare_df(
        adata, groupby, target_col, clip_at, fraction, groupby_count=expanded_in
    )

    return base.bar(plot_df, **kwargs)
