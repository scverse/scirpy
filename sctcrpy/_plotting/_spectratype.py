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
    target_col: list = ["TRB_1_cdr3_len"],
    group_order: Union[list, None] = None,
    top_n: int = 10,
    viztype: Literal["bar", "line", "curve", "table"] = "bar",
    vizarg: Union[dict, None] = None,
    ax: Union[plt.axes, None] = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    sizeprofile: Union[Literal["small"], None] = None,
    fraction: Union[None, str, bool] = None,
    **kwds
) -> Union[List[plt.axes], AnnData]:
    """Plots how many cells belong to each clonotype. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the abundance. 
    grouporder
        Specifies the order of group (samples).
    top_n
        Number of groups to plot. 
    viztype
        The user can later choose the layout of the plot. Currently supports `bar` and `stacked`.  
    vizarg
        Custom values to be passed to the plot in a dictionary of arguments.   
    ax
        Custom axis if needed.  
    curve_layout
        if the KDE-based curves should be stacked or shifted vetrically.  
    sizeprofile
        Figure size and font sizes to make everything legible. Currenty only `small` is supported.      
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype.
    
    Returns
    -------
    List of axes or the dataFrame to plot.
    """

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, None, groupby)
    target_col_l = "|".join(target_col)

    plottable = tl.spectratype(
        adata,
        groupby,
        target_col=target_col,
        fraction=fraction,
        fraction_base=fraction_base,
    )

    if type(plottable) == dict:
        plottable = pd.DataFrame.from_dict(adata, orient="index")

    if vizarg is None:
        vizarg = dict()

    if group_order is None:
        group_order = plottable.columns.values
    plottable = plottable.loc[:, group_order]

    # We need to convert the contingency tables back for the KDE in seaborn, using pseudo-counts in case of fractions
    if fraction:
        ftr = 1000 / plottable.max().max()
    countable, counted = [], []
    for cn in plottable.columns:
        counts = np.round(plottable[cn] * ftr)
        if counts.sum() > 0:
            countable.append(np.repeat(plottable.index.values, counts))
            counted.append(cn)
    countable, counted = countable[:top_n], counted[:top_n]

    # Create text for default labels
    title = "Spectratype of " + groupby + " (" + target_col_l + ")"
    if fraction:
        xlab = "Fraction of cells in " + fraction_base
        ylab = "Fraction of cells in " + fraction_base
    else:
        xlab = "Number of cells"
        ylab = "Number of cells"

    # Create a dictionary of plot layouts
    plot_router = {
        "bar": {
            "f": base.bar,
            "arg": {
                "data": plottable,
                "title": title,
                "legend_title": groupby,
                "ylab": ylab,
                "xlab": target_col_l,
                "ax": ax,
                "fraction": fraction,
                "stacked": True,
            },
        },
        "line": {
            "f": base.line,
            "arg": {
                "data": plottable,
                "title": title,
                "legend_title": groupby,
                "ylab": ylab,
                "xlab": target_col_l,
                "ax": ax,
                "fraction": fraction,
            },
        },
        "curve": {
            "f": base.curve,
            "arg": {
                "data": countable,
                "labels": counted,
                "title": title,
                "legend_title": groupby,
                "ylab": ylab,
                "xlab": target_col_l,
                "ax": ax,
                "fraction": fraction,
                "curve_layout": curve_layout,
            },
        },
    }

    # Check for settings in the profile and call the basic plotting function with merged arguments
    if viztype == "table":
        return plottable
    else:
        if sizeprofile is None:
            profile_args = _check_for_plotting_profile(adata)
        else:
            profile_args = _check_for_plotting_profile(sizeprofile)
        main_args = dict(
            dict(dict(profile_args, **kwds), **plot_router[viztype]["arg"]), **vizarg
        )
        axl = plot_router[viztype]["f"](**main_args)
        return axl
