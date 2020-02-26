import matplotlib.pyplot as plt
import numpy as np
from .._compat import Literal
from anndata import AnnData
from scanpy import logging
import pandas as pd
from .._util import _get_from_uns, _which_fractions, _is_na
from .. import tl
from . import _base as base
from typing import Union, List
from ._styling import _check_for_plotting_profile


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar",
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
    fraction: bool = True,
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
    **kwds,
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
    **kwds,
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
    fraction: Union[None, str, bool] = None,
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


def vdj_usage(
    adata: AnnData,
    *,
    target_cols: list = [
        "TRA_1_j_gene",
        "TRA_1_v_gene",
        "TRB_1_v_gene",
        "TRB_1_d_gene",
        "TRB_1_j_gene",
    ],
    for_cells: Union[None, list, np.ndarray, pd.Series] = None,
    cell_weights: Union[None, str, list, np.ndarray, pd.Series] = None,
    fraction_base: Union[None, str] = None,
    ax: Union[plt.axes, None] = None,
    bar_clip: int = 5,
    top_n: Union[None, int] = 10,
    barwidth: float = 0.4,
    draw_bars: bool = True,
) -> Union[AnnData, dict]:
    """Creates a ribbon plot of the most abundant VDJ combinations in a given subset of cells. 

    Currently works with primary alpha and beta chains only.
    Does not search for precomputed results in `adata`.
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    target_cols
        Columns containing gene segment information. Overwrite default only if you know what you are doing!         
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified,
        all cells in  `adata` will be used that have at least a primary alpha or beta chain.
    cell_weights
        A size factor for each cell. By default, each cell count as 1, but due to normalization
        to different sample sizes for example, it is possible that one cell in a small sample
        is weighted more than a cell in a large sample.
    fraction_base
        As an alternative to supplying ready-made cell weights, this feature can also be calculated
        on the fly if a grouping column name is supplied. The parameter `cell_weights` takes piority
        over `fraction_base`. If both is `None`, each cell will have a weight of 1.
    ax
        Custom axis if needed.
    bar_clip
        The maximum number of stocks for bars (number of different V, D or J segments
        that should be shown separately).
    top_n
        The maximum number of ribbons (individual VDJ combinations). If set to `None`,
        all ribbons are drawn.
    barwidth
        Width of bars.
    draw_bars
        If `False`, only ribbons are drawn and no bars.

    Returns
    -------
    List of axes. 
    """

    # Execute the tool
    df = tl.vdj_usage(
        adata,
        target_cols=target_cols,
        for_cells=for_cells,
        cell_weights=cell_weights,
        fraction_base=fraction_base,
    )
    size_column = "cell_weights"
    if top_n is None:
        top_n = df.shape[0]
    if ax is None:
        fig, ax = plt.subplots()

    # Draw a stacked bar for every gene loci and save positions on the bar
    gene_tops = dict()
    for i in range(len(target_cols)):
        td = (
            df.groupby(target_cols[i])[size_column]
            .agg("sum")
            .sort_values(ascending=False)
        )
        genes = td.index.values
        sector = target_cols[i][2:7]
        unct = td[bar_clip + 1 :,].sum()
        if td.size > bar_clip:
            if draw_bars:
                ax.bar(i + 1, unct, width=barwidth, color="grey", edgecolor="black")
            gene_tops["other_" + sector] = unct
            bottom = unct
        else:
            gene_tops["other_" + sector] = 0
            bottom = 0
        for j in range(bar_clip + 1):
            try:
                y = td[bar_clip - j]
                gene = genes[bar_clip - j]
                if gene == "None":
                    gene = "No_" + sector
                gene_tops[gene] = bottom + y
                if draw_bars:
                    ax.bar(
                        i + 1,
                        y,
                        width=barwidth,
                        bottom=bottom,
                        color="lightgrey",
                        edgecolor="black",
                    )
                    ax.text(
                        1 + i - barwidth / 2 + 0.05,
                        bottom + 0.05,
                        gene.replace("TRA", "").replace("TRB", ""),
                    )
                bottom += y
            except:
                pass

    # Collect data for ribbons
    def generow_formatter(r, loci):
        l = [r[size_column]]
        for g in loci:
            l.append(r[g])
        return l

    td = (
        df.groupby(target_cols)[size_column]
        .agg("sum")
        .sort_values(ascending=False)
        .reset_index()
    )
    td["genecombination"] = td.apply(generow_formatter, axis=1, loci=target_cols)

    # Draw ribbons
    for r in td["genecombination"][1 : top_n + 1]:
        d = []
        ht = r[0]
        for i in range(len(r) - 1):
            g = r[i + 1]
            sector = target_cols[i][2:7]
            if g == "None":
                g = "No_" + sector
            if g not in gene_tops:
                g = "other_" + sector
            t = gene_tops[g]
            d.append([t - ht, t])
            t = t - ht
            gene_tops[g] = t
        if draw_bars:
            base.gapped_ribbons(d, ax, gapwidth=barwidth)
        else:
            base.gapped_ribbons(d, ax, gapwidth=0.1)

    # Make tick labels nicer
    ax.set_xticks(range(1, len(target_cols) + 1))
    if target_cols == [
        "TRA_1_j_gene",
        "TRA_1_v_gene",
        "TRB_1_v_gene",
        "TRB_1_d_gene",
        "TRB_1_j_gene",
    ]:
        ax.set_xticklabels(["TRAJ", "TRAV", "TRBV", "TRBD", "TRBJ"])
    else:
        ax.set_xticklabels(target_cols)

    return [ax]
