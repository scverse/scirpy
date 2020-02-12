import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from ._compat import Literal
from anndata import AnnData
from scanpy import logging
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
from ._util import _get_from_uns, _add_to_uns, _which_fractions
from . import tl
from typing import Union, List, Tuple


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar"
) -> None:
    """Plot the alpha diversity per group. 

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
    try:
        diversity = _get_from_uns(
            adata,
            "alpha_diversity",
            parameters={"groupby": groupby, "target_col": target_col},
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing alpha diversity now. "
        )
        tl.alpha_diversity(adata, groupby, target_col=target_col)
        diversity = _get_from_uns(
            adata,
            "alpha_diversity",
            parameters={"groupby": groupby, "target_col": target_col},
        )

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
    # Get pre-comuted results. If not available, compute them.
    try:
        expansion = _get_from_uns(
            adata,
            "clonal_expansion",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
            },
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing clonal expansion now. "
        )
        tl.clonal_expansion(
            adata, groupby, target_col=target_col, clip_at=clip_at, fraction=fraction
        )
        expansion = _get_from_uns(
            adata,
            "clonal_expansion",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
            },
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

    # If we get an adata object, get pre-computed results. If not available, compute them. Otherwise use the dictionary as is.
    if type(adata) == dict:
        plottable = pd.DataFrame.from_dict(adata, orient="index")
    else:
        try:
            plottable = _get_from_uns(
                adata,
                "cdr_convergence",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col,
                    "clip_at": clip_at,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
            )
        except KeyError:
            logging.warning(
                "No precomputed data found for current parameters. "
                "Computing group CDR3 convergence now. "
            )
            tl.cdr_convergence(
                adata,
                groupby,
                target_col=target_col,
                fraction=fraction,
                fraction_base=fraction_base,
            )
            plottable = _get_from_uns(
                adata,
                "cdr_convergence",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col,
                    "clip_at": clip_at,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
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
            "f": nice_bar_plain,
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
        },
    }

    # Check for settings in the profile and call the basic plotting function with merged arguments
    if viztype == "table":
        return plottable
    else:
        if sizeprofile is None:
            profile_args = check_for_plotting_profile(adata)
        else:
            profile_args = check_for_plotting_profile(sizeprofile)
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

    # If we get an adata object, get pre-computed results. If not available, compute them. Otherwise use the dictionary as is.
    if type(adata) == dict:
        plottable = pd.DataFrame.from_dict(adata, orient="index")
    else:
        try:
            plottable = _get_from_uns(
                adata,
                "spectratype",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col_l,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
            )
        except KeyError:
            logging.warning(
                "No precomputed data found for current parameters. "
                "Computing group abundance now. "
            )
            tl.spectratype(
                adata,
                groupby,
                target_col=target_col,
                fraction=fraction,
                fraction_base=fraction_base,
            )
            plottable = _get_from_uns(
                adata,
                "spectratype",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col_l,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
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
            "f": nice_bar_plain,
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
            "f": nice_line_plain,
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
            "f": nice_curve_plain,
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
            profile_args = check_for_plotting_profile(adata)
        else:
            profile_args = check_for_plotting_profile(sizeprofile)
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

    # If we get an adata object, get pre-computed results. If not available, compute them. Otherwise use the dictionary as is.
    if type(adata) == dict:
        abundance = pd.DataFrame.from_dict(adata, orient="index")
    else:
        try:
            abundance = _get_from_uns(
                adata,
                "group_abundance",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
            )
        except KeyError:
            logging.warning(
                "No precomputed data found for current parameters. "
                "Computing group abundance now. "
            )
            tl.group_abundance(
                adata,
                groupby,
                target_col=target_col,
                fraction=fraction,
                fraction_base=fraction_base,
            )
            abundance = _get_from_uns(
                adata,
                "group_abundance",
                parameters={
                    "groupby": groupby,
                    "target_col": target_col,
                    "fraction": fraction,
                    "fraction_base": fraction_base,
                },
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
    target_ranks = target_ranks[~np.isin(target_ranks, ["nan"])]
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
            "f": nice_stripe_plain,
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
            "f": nice_bar_plain,
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
            profile_args = check_for_plotting_profile(adata)
        else:
            profile_args = check_for_plotting_profile(sizeprofile)
        main_args = dict(dict(profile_args, **plot_router[viztype]["arg"]), **vizarg)
        axl = plot_router[viztype]["f"](**main_args)
        return axl


############################################################
# Basic plotting functions
############################################################


def nice_bar_plain(
    data: Union[dict, np.ndarray, pd.DataFrame, AnnData],
    *,
    ax: Union[plt.axes, list, None] = None,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 1.5,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 8,
    stacked: bool = True,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Convert data to a Pandas dataframe if not already a dataframe.
    if not isinstance(data, pd.DataFrame):
        if type(data) == dict:
            data = pd.DataFrame.from_dict(data, orient="index")
        else:
            if type(data) is np.ndarray:
                data = pd.DataFrame(
                    data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]
                )
            else:
                raise ValueError("`data` does not seem to be a valid input type")

    # Create figure if not supplied already. If multiple axes are supplied, it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Draw the plot with Pandas
    ax = data.plot.bar(ax=ax, stacked=stacked)

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)

    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        ax.set_xticklabels(
            ax.get_xticklabels(), fontsize=tick_fontsize, rotation=30, ha="right"
        )
        xax = ax.get_xaxis()
        xax.set_tick_params(length=0)
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        if fraction:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(x))))
            ax.set_yticklabels(
                [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            title_fontsize=label_fontsize,
            fontsize=tick_fontsize,
            frameon=False,
        )
        ax.set_position([0.1, 0.3, 0.6, 0.55])

    return [ax]


def nice_line_plain(
    data: Union[dict, np.ndarray, pd.DataFrame, AnnData],
    *,
    ax: Union[plt.axes, list, None] = None,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 10,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 6,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Convert data to a Pandas dataframe if not already a dataframe.
    if not isinstance(data, pd.DataFrame):
        if type(data) == dict:
            data = pd.DataFrame.from_dict(data, orient="index")
        else:
            if type(data) is np.ndarray:
                data = pd.DataFrame(
                    data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]
                )
            else:
                raise ValueError("`data` does not seem to be a valid input type")

    # Create figure if not supplied already. If multiple axes are supplied, it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Draw the plot with Pandas
    ax = data.plot.line(ax=ax)

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.set_xticklabels(
            [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
        )
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        if fraction:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            ax.set_yticklabels(
                [str(int(x)) for x in ax.get_yticks()], fontsize=tick_fontsize
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            title_fontsize=label_fontsize,
            fontsize=tick_fontsize,
            frameon=False,
        )
        ax.set_position([0.3, 0.2, 0.5, 0.75])
    return [ax]


def nice_curve_plain(
    data: List[Union[np.ndarray, pd.Series]],
    labels: Union[list, np.ndarray, pd.Series],
    *,
    ax: Union[plt.axes, list, None] = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 10,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 6,
    shade: bool = True,
    outline: bool = True,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Counts or pseudo-counts for KDE.
    labels
        The label to display for each curve
    ax
        Custom axis if needed.
    curve_layout
        if the KDE-based curves should be stacked or shifted vetrically.  
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    shade
        If shade area should be plotted. 
    outline
        If the outline should be drawn. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Create figure if not supplied already. If multiple axes are supplied, it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Check what would be the plotting range
    xmax = 0
    for d in data:
        try:
            m = max(d)
            if m > xmax:
                xmax = m
        except:
            pass
    xmax += 1
    x = np.arange(0, xmax, 0.1)

    # Draw a curve for every series
    for i in range(len(data)):
        X = np.array([data[i]]).reshape(-1, 1)
        # kde = KernelDensity(kernel="epanechnikov", bandwidth=3).fit(X)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.6).fit(X)
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        if curve_layout == "shifted":
            y = y + i
            fy = i
        else:
            if curve_layout == "stacked":
                outline = False
                if i < 1:
                    _y = np.zeros(len(y))
                fy = _y[:]
                _y = _y + y
                y = fy + y
        if shade:
            if outline:
                ax.plot(x, y, label=labels[i])
                ax.fill_between(x, y, fy, alpha=0.6)
            else:
                ax.fill_between(x, y, fy, alpha=0.6, label=labels[i])
        else:
            ax.plot(x, y, label=labels[i])

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.set_xticklabels(
            [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
        )
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        if fraction:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            ax.set_yticklabels(
                [str(int(x)) for x in ax.get_yticks()], fontsize=tick_fontsize
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if curve_layout == "shifted":
            for e in range(xmax):
                ax.axvline(e, color="whitesmoke", lw=0.1)
            ax.set_position([0.3, 0.2, 0.7, 0.75])
            ax.set_yticks(range(i + 1))
            ax.set_yticklabels(
                [labels[e] for e in range(i + 1)], fontsize=tick_fontsize
            )
            ax.legend().remove()
        else:
            ax.legend(
                title=legend_title,
                loc="upper left",
                bbox_to_anchor=(1.2, 1),
                title_fontsize=label_fontsize,
                fontsize=tick_fontsize,
                frameon=False,
            )
            ax.set_position([0.3, 0.2, 0.5, 0.75])
    return [ax]


def nice_stripe_plain(
    data: Union[dict, np.ndarray, pd.DataFrame, AnnData],
    *,
    ax: Union[plt.axes, list, None] = None,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 10,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 6,
    stacked: bool = True,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Convert data to a Pandas dataframe if not already a dataframe.
    if not isinstance(data, pd.DataFrame):
        if type(data) == dict:
            data = pd.DataFrame.from_dict(data, orient="index")
        else:
            if type(data) is np.ndarray:
                data = pd.DataFrame(
                    data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:]
                )
            else:
                raise ValueError("`data` does not seem to be a valid input type")

    # Create figure if not supplied already. If multiple axes are supplied, it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Draw the plot with Pandas
    ax = data.plot.barh(ax=ax)

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        if fraction:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_xticklabels(ax.get_xticks(), fontsize=tick_fontsize)
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(x))))
            ax.set_xticklabels(
                [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
            )
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(
            ax.get_yticklabels(), fontsize=tick_fontsize, horizontalalignment="left"
        )
        yax = ax.get_yaxis()
        yax.set_tick_params(length=0, pad=60)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            title_fontsize=label_fontsize,
            fontsize=tick_fontsize,
            frameon=False,
        )
        ax.set_position([0.3, 0.2, 0.4, 0.65])
    return [ax]


def reset_plotting_profile(adata: AnnData) -> None:
    """
    Reverts plotting profile to matplotlib defaults (rcParams).  
    """
    try:
        p = _get_from_uns(adata, "plotting_profile")
    except KeyError:
        p = dict()
    p["title_loc"] = plt.rcParams["axes.titleloc"]
    p["title_pad"] = plt.rcParams["axes.titlepad"]
    p["title_fontsize"] = plt.rcParams["axes.titlesize"]
    p["label_fontsize"] = plt.rcParams["axes.labelsize"]
    p["tick_fontsize"] = plt.rcParams["xtick.labelsize"]
    _add_to_uns(adata, "plotting_profile", p)
    return


def check_for_plotting_profile(profile: Union[AnnData, str, None] = None) -> dict:
    """
    Passes a predefined set of plotting atributes to basic plotting fnctions.
    """
    profiles = {
        "vanilla": {},
        "small": {
            "figsize": (3.44, 2.58),
            "figresolution": 300,
            "title_loc": "center",
            "title_pad": 10,
            "title_fontsize": 10,
            "label_fontsize": 8,
            "tick_fontsize": 6,
        },
    }
    p = profiles["small"]
    if isinstance(profile, AnnData):
        try:
            p = _get_from_uns(profile, "plotting_profile")
        except KeyError:
            pass
    else:
        if isinstance(profile, str):
            if profile in profiles:
                p = profiles[profile]
    return p
