from collections.abc import Sequence
from typing import Literal

from scirpy import tl
from scirpy.util import DataHandler

from . import base


@DataHandler.inject_param_docs()
def clonal_expansion(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    expanded_in: str | None = None,
    breakpoints: Sequence[int] = (1, 2),
    clip_at: int | None = None,
    summarize_by: Literal["cell", "clone_id"] = "cell",
    normalize: bool = True,
    show_nonexpanded: bool = True,
    viztype: Literal["bar", "barh"] = "bar",
    airr_mod: str = "airr",
    **kwargs,
):
    """
    Visualize clonal expansion.

    Plots the fraction of cells that belong to an expanded :term:`Clonotype` by
    a categorical variable.

    If `summarize_by` is set to "clone_id" it plots the fraction
    of clonotypes instead of the fraction of cells.

    Removes all entries with `NaN` in `target_col` prior to plotting.

    Parameters
    ----------
    {adata}
    groupby
        Group by this categorical variable in `adata.obs`.
    target_col
        Column in `adata.obs` containing the clonotype information.
    expanded_in
        Calculate clonal expansion within groups. To calculate expansion
        within patients, set this to the column containing patient annotation.
        If set to None, a clonotype counts as expanded if there's any cell of the
        same clonotype across the entire dataset. See also :term:`Public clonotype`.
    breakpoints
        summarize clonotypes with a size smaller or equal than the specified numbers
        into groups. For instance, if this is (1, 2, 5), there will be four categories:

        * all clonotypes with a size of 1 (singletons)
        * all clonotypes with a size of 2
        * all clonotypes with a size between 3 and 5 (inclusive)
        * all clonotypes with a size > 5
    clip_at
        This argument is superseded by `breakpoints` and is only kept for backwards-compatibility.
        Specifying a value of `clip_at = N` equals to specifying `breakpoints = (1, 2, 3, ..., N)`
        Specifying both `clip_at` overrides `breakpoints`.
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
    {airr_mod}
    **kwargs
        Additional arguments passed to :func:`scirpy.pl.base.bar`
    """
    params = DataHandler(adata, airr_mod)
    plot_df = tl.summarize_clonal_expansion(
        params,
        groupby,
        target_col=target_col,
        summarize_by=summarize_by,
        normalize=normalize,
        expanded_in=expanded_in,
        breakpoints=breakpoints,
        clip_at=clip_at,
    )
    if not show_nonexpanded:
        plot_df.drop("<= 1", axis="columns", inplace=True)

    return {"bar": base.bar, "barh": base.barh}[viztype](plot_df, **kwargs)
