import matplotlib.pyplot as plt
from .._compat import Literal
from anndata import AnnData
from .. import tl
from . import base
from typing import Union, Sequence
from .styling import _get_colors
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def group_abundance(
    adata: Union[dict, AnnData],
    groupby: str,
    target_col: str = "has_ir",
    *,
    normalize: Union[None, str, bool] = None,
    max_cols: Union[None, int] = None,
    sort: Union[Literal["count", "alphabetical"], Sequence[str]] = "count",
    **kwargs,
) -> plt.Axes:
    """Plots the number of cells per group, split up by a categorical variable.

    Generates a stacked bar chart with one bar per group. Stacks
    are colored according to the categorical variable specified in `target_col`.

    Ignores NaN values.

    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. For instance, "sample" or "diagnosis".
    target_col
        Column on which to compute the abundance.
        Defaults to `has_ir` which computes the number of all cells
        that have a T-cell receptor.
    normalize
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, the name
        of a column containing a categorical variable can be provided,
        according to which the values will be normalized.
    max_cols:
        Only plot the first `max_cols` columns. If set to `None` (the default)
        the function will raise a `ValueError` if attempting to plot more
        than 100 columns. Set to `0` to disable.
    sort
        How to arrange the dataframe columns.
        Default is by the category count ("count").
        Other options are "alphabetical" or to provide a list of column names.
        By providing an explicit list, the DataFrame can also be subsetted to
        specific categories. Sorting (and subsetting) occurs before `max_cols`
        is applied.
    **kwargs
        Additional arguments passed to :func:`scirpy.pl.base.bar`.

    Returns
    -------
    Axes object
    """
    abundance = tl.group_abundance(
        adata, groupby, target_col=target_col, fraction=normalize, sort=sort
    )
    if abundance.shape[0] > 100 and max_cols is None:
        raise ValueError(
            "Attempting to plot more than 100 columns. "
            "Set `max_cols` to a sensible value or to `0` to disable this message"
        )
    if max_cols is not None and max_cols > 0:
        abundance = abundance.iloc[:max_cols, :]

    if "color" not in kwargs:
        colors = _get_colors(adata, target_col)
        kwargs["color"] = [colors[cat] for cat in abundance.columns]

    # Create text for default labels
    if normalize:
        fraction_base = target_col if normalize is True else normalize
        title = "Fraction of " + target_col + " in each " + groupby
        xlab = groupby
        ylab = "Fraction of cells in " + fraction_base
    else:
        title = "Number of cells in " + groupby + " by " + target_col
        xlab = groupby
        ylab = "Number of cells"

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    kwargs["style_kws"] = default_style_kws

    return base.bar(abundance, **kwargs)
