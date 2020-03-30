from .. import tl
from anndata import AnnData
from . import base
from .._util import _normalize_counts, _is_na


def _prepare_df(adata, groupby, target_col, clip_at, fraction):
    """Turn the result of the `clip_and_count` tool into a plottable
    dataframe"""
    tmp_col = target_col + "clipped_count"
    tmp_col_weight = target_col + "weight"

    obs = adata.obs.loc[:, [groupby, target_col]]
    obs[tmp_col] = tl.clip_and_count(
        adata, target_col, groupby=groupby, clip_at=clip_at, inplace=False
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
    plot_df = _prepare_df(adata, groupby, target_col, clip_at, fraction)

    return base.bar(plot_df, **kwargs)


def clonal_expansion(
    adata: AnnData, groupby: str, *, clip_at: int = 3, fraction: bool = True, **kwargs
):
    """Plot the fraction of cells in each group belonging to
    singleton, doublet or triplet clonotype. 

    This is a wrapper for :meth:`pl.clip_and_count`. 
    """
    default_style_kws = {"title": "Clonal expansion"}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    return clip_and_count(
        adata,
        groupby,
        target_col="clonotype",
        clip_at=clip_at,
        fraction=fraction,
        style_kws=default_style_kws,
        **kwargs,
    )
