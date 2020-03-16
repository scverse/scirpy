from .. import tl
from anndata import AnnData
from . import base


def clip_and_count(
    adata: AnnData,
    groupby: str,
    target_col: str,
    *,
    clip_at: int = 3,
    fraction: bool = True,
    **kwargs,
):
    """Plots the the number of identical entries in `target_col` 
    for each group in `group_by`. 

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
    res = tl.clip_and_count(
        adata, groupby, target_col, clip_at=clip_at, fraction=fraction
    )

    return base.bar(res, **kwargs)


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
