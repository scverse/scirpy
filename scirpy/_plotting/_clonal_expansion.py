from .. import tl
from anndata import AnnData
from . import base
from typing import Union
from .._compat import Literal
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def clonal_expansion(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clone_id",
    clip_at: int = 3,
    expanded_in: Union[str, None] = None,
    summarize_by: Literal["cell", "clone_id"] = "cell",
    normalize: bool = True,
    show_nonexpanded: bool = True,
    viztype: Literal["bar", "barh"] = "bar",
    **kwargs,
):
    """Visualize clonal expansion.

    Plots the fraction of cells that belong to an expanded :term:`Clonotype` by
    a categorical variable.

    If `summarize_by` is set to "clone_id" it plots the fraction
    of clonotypes instead of the fraction of cells.

    Removes all entries with `NaN` in `target_col` prior to plotting.

    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this categorical variable in `adata.obs`.
    target_col
        Column in `adata.obs` containing the clonotype information.
    clip_at
        All entries in `target_col` with more copies than `clip_at`
        will be summarized into a single group.
    expanded_in
        Calculate clonal expansion within groups. To calculate expansion
        within patients, set this to the column containing patient annotation.
        If set to None, a clonotype counts as expanded if there's any cell of the
        same clonotype across the entire dataset. See also :term:`Public clonotype`.
    summarize_by
        Can be either `cell` to count cells belonging to a clonotype (the default),
        or `clone_id` to count clonotypes. The former leads to a over-representation
        of expanded clonotypes but better represents the fraction of expanded cells.
    normalize
        If True, compute fractions rather than reporting
        abosolute numbers.
    show_nonexpanded
        Whether or not to show the fraction of non-expanded cells/clonotypes
    viztype
        `bar` for bars, `barh` for horizontal bars.
    **kwargs
        Additional arguments passed to :func:`scirpy.pl.base.bar`
    """
    plot_df = tl.summarize_clonal_expansion(
        adata,
        groupby,
        target_col=target_col,
        summarize_by=summarize_by,
        normalize=normalize,
        expanded_in=expanded_in,
        clip_at=clip_at,
    )
    if not show_nonexpanded:
        plot_df.drop("1", axis="columns", inplace=True)

    return {"bar": base.bar, "barh": base.barh}[viztype](plot_df, **kwargs)
