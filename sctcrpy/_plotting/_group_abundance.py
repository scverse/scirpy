import matplotlib.pyplot as plt
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from .._util import _is_na
from .. import tl
from . import _base as base
from typing import Union, List
from ._styling import _check_for_plotting_profile


def group_abundance(
    adata: Union[dict, AnnData],
    groupby: str,
    *,
    target_col: str = "clonotype",
    label_col: Union[str, None] = None,
    group_order: Union[list, None] = None,
    top_n: int = 10,
    viztype: Literal["bar", "stacked", "table"] = "bar",
    vizarg: Union[dict, None] = None,
    ax: Union[plt.axes, None] = None,
    sizeprofile: Union[Literal["small"], None] = None,
    fraction: Union[None, str, bool] = None
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
    label_col
        Column containing names for clonotypes. 
    grouporder
        Specifies the order of group (samples).
    top_n
        Top clonotypes to plot. 
    viztype
        The user can later choose the layout of the plot. Currently supports `bar` and `stacked`.  
    vizarg
        Custom values to be passed to the plot in a dictionary of arguments.   
    ax
        Custom axis if needed.   
    sizeprofile
        Figure size and font sizes to make everything legible. Currenty only `small` is supported. 
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype.
    
    Returns
    -------
    List of axes or the dataFrame to plot.
    """

    fraction, fraction_base = _which_fractions(fraction, None, groupby)
    abundance = tl.group_abundance(
        adata,
        groupby,
        target_col=target_col,
        fraction=fraction,
        fraction_base=fraction_base,
    )

    if type(abundance) == dict:
        abundance = pd.DataFrame.from_dict(adata, orient="index")

    if vizarg is None:
        vizarg = dict()

    # Filter the pivot table to leave only data that we want to plot
    target_ranks = abundance.index.values
    target_ranks = target_ranks[:top_n]
    if viztype in ["bar"]:
        target_ranks = target_ranks[::-1]
    target_ranks = target_ranks[~_is_na(target_ranks)]
    if group_order is None:
        group_order = abundance.columns.values
    abundance = abundance.loc[target_ranks, group_order]
    if label_col is not None:
        relabels = dict()
        for d in adata.obs.loc[:, [label_col, target_col]].to_dict(orient="records"):
            relabels[d[target_col]] = d[label_col]
        abundance.index = abundance.index.map(relabels)

    # Create text for default labels
    if fraction:
        title = "Fraction of top " + target_col + "s in each " + groupby
        xlab = "Fraction of cells in " + fraction_base
        ylab = "Fraction of cells in " + fraction_base
    else:
        title = "Number of cells in top " + target_col + "s by " + groupby
        xlab = "Number of cells"
        ylab = "Number of cells"

    # Create a dictionary of plot layouts
    plot_router = {
        "bar": {
            "f": base.stripe,
            "arg": {
                "data": abundance,
                "title": title,
                "legend_title": groupby,
                "xlab": xlab,
                "ax": ax,
                "fraction": fraction,
            },
        },
        "stacked": {
            "f": base.bar,
            "arg": {
                "data": abundance,
                "title": title,
                "legend_title": groupby,
                "ylab": ylab,
                "ax": ax,
                "fraction": fraction,
                "stacked": True,
            },
        },
    }

    # Check for settings in the profile and call the basic plotting function with merged arguments
    if viztype == "table":
        return abundance
    else:
        if sizeprofile is None:
            profile_args = _check_for_plotting_profile(adata)
        else:
            profile_args = _check_for_plotting_profile(sizeprofile)
        main_args = dict(dict(profile_args, **plot_router[viztype]["arg"]), **vizarg)
        axl = plot_router[viztype]["f"](**main_args)
        return axl
