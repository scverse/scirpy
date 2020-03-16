import matplotlib.pyplot as plt
import numpy as np
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from .. import tl
from . import _base as base
from typing import Union, List


def spectratype(
    adata: Union[dict, AnnData],
    groupby: str,
    *,
    target_cols: list = ["TRB_1_cdr3_len"],
    viztype: Literal["bar", "curve"],
    fraction: Union[None, str, bool] = None,
    **kwargs
) -> Union[List[plt.Axes], AnnData]:
    """Plots how many cells belong to each clonotype. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_cols
        Column on which to compute the abundance. 
    viztype
         
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton,
        doublet or triplet clonotype.
    **kwargs
        Additional parameters passed to the base plotting function

    
    Returns
    -------
    Axes object
    """

    data = tl.spectratype(adata, groupby, target_cols=target_cols, fraction=fraction)

    # TODO: is the pseudocounting really necessary? See also #21
    # # We need to convert the contingency tables back for the KDE in seaborn,
    # # using pseudo-counts in case of fractions
    # if fraction:
    #     ftr = 1000 / plottable.max().max()
    # countable, counted = [], []
    # for cn in plottable.columns:
    #     counts = np.round(plottable[cn] * ftr)
    #     if counts.sum() > 0:
    #         countable.append(np.repeat(plottable.index.values, counts))
    #         counted.append(cn)
    # countable, counted = countable[:top_n], counted[:top_n]

    # Create text for default labels
    title = "Spectratype of " + groupby + " (" + "|".join(target_cols) + ")"
    fraction_base = groupby if fraction is True else fraction
    if fraction:
        xlab = "Fraction of cells in " + fraction_base
        ylab = "Fraction of cells in " + fraction_base
    else:
        xlab = "Number of cells"
        ylab = "Number of cells"

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])

    plot_router = {
        "bar": base.bar,
        "line": base.line,
        "curve": base.curve,
    }
    return plot_router[viztype](data, style_kws=default_style_kws, **kwargs)
