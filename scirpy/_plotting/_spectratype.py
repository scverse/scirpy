import matplotlib.pyplot as plt
import numpy as np
from .._compat import Literal
from anndata import AnnData
from .. import tl
from . import _base as base
from typing import Union, List, Collection, Callable


def spectratype(
    adata: Union[dict, AnnData],
    groupby: Union[str, Collection[str]] = ["TRA_1_cdr3"],
    *,
    target_col: str,
    combine_fun: Callable = np.sum,
    fraction: Union[None, str, bool] = None,
    viztype: Literal["bar", "line", "curve"] = "bar",
    **kwargs,
) -> Union[List[plt.Axes], AnnData]:
    """Show the distribution of CDR3 region lengths. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Column(s) containing CDR3 lengths.        
    target_col
        Color by this column from `obs`. E.g. sample or diagnosis 
    combine_fun
        A function definining how the groupby columns should be merged 
        (e.g. sum, mean, median, etc).  
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized.  
    viztype
        Type of plot to produce
    **kwargs
        Additional parameters passed to the base plotting function

    
    Returns
    -------
    Axes object
    """

    data = tl.spectratype(
        adata,
        groupby,
        target_col=target_col,
        combine_fun=combine_fun,
        fraction=fraction,
    )

    # # We need to convert the contingency tables back for the KDE in seaborn,
    # # using pseudo-counts in case of fractions
    # if fraction:
    #     ftr = 1000 / np.max(data.values)
    # countable, counted = [], []
    # for cn in data.columns:
    #     counts = np.round(data[cn] * ftr)
    #     if counts.sum() > 0:
    #         countable.append(np.repeat(plottable.index.values, counts))
    #         counted.append(cn)
    # # countable, counted = countable[:top_n], counted[:top_n]

    groupby_text = groupby if isinstance(groupby, str) else "|".join(groupby)
    title = "Spectratype of " + groupby_text + " by " + target_col
    xlab = groupby_text + " length"
    if fraction:
        fraction_base = target_col if fraction is True else fraction
        ylab = "Fraction of cells in " + fraction_base
    else:
        ylab = "Number of cells"

    color_key = f"{target_col}_colors"
    if color_key in adata.uns and "color" not in kwargs:
        cat_index = {
            cat: i for i, cat in enumerate(adata.obs[target_col].cat.categories)
        }
        kwargs["color"] = [adata.uns[color_key][cat_index[cat]] for cat in data.columns]

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])

    plot_router = {
        "bar": base.bar,
        "line": base.line,
        "curve": base.curve,
    }
    return plot_router[viztype](data, style_kws=default_style_kws, **kwargs)
