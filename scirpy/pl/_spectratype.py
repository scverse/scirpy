import matplotlib.pyplot as plt
import numpy as np
from .._compat import Literal
from anndata import AnnData
from .. import tl
from . import base
from typing import Union, List, Collection, Callable
from .styling import _get_colors
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def spectratype(
    adata: Union[dict, AnnData],
    cdr3_col: Union[str, Collection[str]] = ["IR_VJ_1_junction_aa"],
    *,
    color: str,
    combine_fun: Callable = np.sum,
    normalize: Union[None, str, bool] = None,
    viztype: Literal["bar", "line", "curve"] = "bar",
    kde_kws: Union[dict, None] = None,
    **kwargs,
) -> Union[List[plt.Axes], AnnData]:
    """Show the distribution of CDR3 region lengths.

    Ignores NaN values.

    Parameters
    ----------
    adata
        AnnData object to work on.
    cdr3_col
        Column(s) containing CDR3 lengths.
    color
        Color by this column from `obs`. E.g. sample or diagnosis
    combine_fun
        A function definining how the `cdr3_col` columns should be merged,
        in case multiple ones were specified.
        (e.g. sum, mean, median, etc).
    normalize
        If True, compute fractions of abundances relative to the `cdr3_col` column
        rather than reporting abosolute numbers. Alternatively, the name of a column
        containing a categorical variable can be provided according to which
        the values will be normalized.
    viztype
        Type of plot to produce.
    **kwargs
        Additional parameters passed to the base plotting function


    Returns
    -------
    Axes object
    """

    data = tl.spectratype(
        adata,
        cdr3_col,
        target_col=color,
        combine_fun=combine_fun,
        fraction=normalize,
    )

    groupby_text = cdr3_col if isinstance(cdr3_col, str) else "|".join(cdr3_col)
    title = "Spectratype of " + groupby_text + " by " + color
    xlab = groupby_text + " length"
    if normalize:
        fraction_base = color if normalize is True else normalize
        ylab = "Fraction of cells in " + fraction_base
    else:
        ylab = "Number of cells"

    if "color" not in kwargs:
        colors = _get_colors(adata, color)
        kwargs["color"] = [colors[cat] for cat in data.columns]

    # For KDE curves, we need to convert the contingency tables back
    if viztype == "curve":
        if normalize:
            data = (
                10 * data
            ) / data.min()  # Scales up data so that even fraction become an integer count
        countable = {}
        for cn in data.columns:
            counts = data[cn].round()
            if counts.sum() > 0:
                countable[cn] = np.repeat(data.index.values, counts)
            else:
                countable[cn] = np.zeros(10)
        data = countable

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])

    plot_router = {
        "bar": base.bar,
        "line": base.line,
        "curve": base.curve,
    }
    return plot_router[viztype](data, style_kws=default_style_kws, **kwargs)
