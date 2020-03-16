from .. import tl
from anndata import AnnData


def clip_and_count(
    adata: AnnData,
    groupby: str,
    target_col: str,
    *,
    clip_at: int = 3,
    fraction: bool = True
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
    """
    res = tl.clip_and_count(
        adata, groupby, target_col, clip_at=clip_at, fraction=fraction
    )

    res.plot.bar(stacked=True)


def clonal_expansion(
    adata: AnnData, groupby: str, *, clip_at: int = 3, fraction: bool = True
):
    """Plot the fraction of cells in each group belonging to
    singleton, doublet or triplet clonotype. 

    This is a wrapper for :meth:`pl.clip_and_count`. 
    """
    return clip_and_count(
        adata, groupby, target_col="clonotype", clip_at=clip_at, fraction=fraction
    )
