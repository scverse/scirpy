import matplotlib.pyplot as plt
import numpy as np
from .._compat import Literal
from anndata import AnnData
from scanpy import logging
import pandas as pd
from .._util import _get_from_uns, _is_na
from .. import tl
from . import _base as base
from typing import Union, List
from ._styling import _check_for_plotting_profile


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar"
) -> None:
    """Plot the alpha diversity per group. 

    Calls :meth:`tl.alpha_diversity` on-the-fly. 

    Parameters
    ----------
    adata
        Annotated data matrix
    groupby
        Column of `obs` by which the grouping will be performed
    target_col
        Column on which to compute the alpha diversity
    vistype
        Visualization type. Currently only 'bar' is supported. 
    """
    diversity = tl.alpha_diversity(adata, groupby, target_col=target_col)

    groups, diversity = zip(*diversity.items())

    # TODO: this should probably make use of a `barplot` "base" plotting function.
    fig, ax = plt.subplots()
    x = np.arange(len(groups))
    ax.bar(x, diversity)

    ax.set_ylabel("Shannon entropy")
    ax.set_title("Alpha diversity of {} by {}".format(target_col, groupby))
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def clonal_expansion(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    clip_at: int = 3,
    fraction: bool = True
):
    """Plot the fraction of cells in each group belonging to
    singleton, doublet or triplet clonotype. 
    """
    expansion = tl.clonal_expansion(
        adata, groupby, target_col=target_col, clip_at=clip_at, fraction=fraction
    )

    pd.DataFrame.from_dict(expansion, orient="index").plot.bar(stacked=True)


def cdr_convergence(
    adata: Union[dict, AnnData],
    groupby: str,
    *,
    target_col: str = "TRB_1_cdr3",
    clip_at: int = 3,
    group_order: Union[list, None] = None,
    top_n: int = 10,
    viztype: Literal["bar", "table"] = "bar",
    vizarg: Union[dict, None] = None,
    ax: Union[plt.axes, None] = None,
    sizeprofile: Union[Literal["small"], None] = None,
    no_singles: bool = False,
    fraction: Union[None, str, bool] = None,
    **kwds
) -> Union[List[plt.axes], AnnData]:
    """how many nucleotide versions a single CDR3 amino acid sequence typically has in a given group
    cells belong to each clonotype within a certain sample.

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the expansion. Useful if we want to specify the chain.        
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified, cells with NaN values in the group definition columns will be ignored. When the tool is executed by the plotting function, the whitelist is not updated.          
    clip_at
        All clonotypes with more copies than `clip_at` will be summarized into 
        a single group. 
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
        If True, compute fractions of cells rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype.
    no_singles
        If non-convergent clonotypes should be shown explicitely.
    
    Returns
    -------
    List of axes or the dataFrame to plot.
    """

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, None, groupby)
    plottable = tl.cdr_convergence(
        adata,
        groupby,
        target_col=target_col,
        fraction=fraction,
        fraction_base=fraction_base,
    )

    if type(plottable) == dict:
        plottable = pd.DataFrame.from_dict(adata, orient="index")
    if no_singles:
        plottable = plottable.drop("1", axis=1)

    if vizarg is None:
        vizarg = dict()

    if group_order is None:
        group_order = plottable.index.values
    plottable = plottable.loc[group_order, :]

    # Create text for default labels
    title = "Convergence of CDR3 regions in " + groupby + "s"
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
                "legend_title": "Versions",
                "ylab": ylab,
                "xlab": target_col,
                "ax": ax,
                "fraction": fraction,
                "stacked": True,
            },
        }
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
