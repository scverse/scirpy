from .. import tl
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Tuple
import pandas as pd
import numpy as np


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
    fraction: Union[None, str, list, np.ndarray, pd.Series] = None,
    size_column: str = "cell_weights",
    ax: Union[plt.axes, None] = None,
    bar_clip: int = 5,
    top_n: Union[None, int] = 10,
    barwidth: float = 0.4,
    draw_bars: bool = True,
) -> plt.Axes:
    """Creates a ribbon plot of the most abundant VDJ combinations in a given subset of cells. 

    Currently works with primary alpha and beta chains only.
    Does not search for precomputed results in `adata`.
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    target_cols
        Columns containing gene segment information. Overwrite default only if you know what you are doing!         
    fraction
        Either the name of a categorical column that should be used as the base for computing fractions,
        or an iterable specifying a size factor for each cell. By default, each cell count as 1,
        but due to normalization to different sample sizes for example, it is possible that one cell
        in a small sample is weighted more than a cell in a large sample.
    size_column
        The name of the column that will be used for storing cell weights. This value is used internally
        and should be matched with the column name used by the tool function. Best left untouched.
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
    Axes object. 
    """

    # Execute the tool
    df = tl.vdj_usage(
        adata, target_cols=target_cols, fraction=fraction, cell_weights=cell_weights
    )

    if top_n is None:
        top_n = df.shape[0]
    if ax is None:
        fig, ax = plt.subplots()

    # Draw a stacked bar for every gene loci and save positions on the bar
    gene_tops = dict()
    for i in range(len(target_cols)):
        td = (
            df.groupby(target_cols[i])
            .agg({size_column: "sum"})
            .sort_values(by=size_column, ascending=False)
            .reset_index()
        )
        genes = td[target_cols[i]].tolist()
        td = td[size_column]
        sector = target_cols[i][2:7]
        # sector = sector.replace('_', '')
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

    # Count occurance of individual VDJ combinations
    td = df.loc[:, target_cols + [size_column]]
    td["genecombination"] = td.apply(
        lambda x, y: "|".join([x[e] for e in y]), y=target_cols, axis=1
    )
    td = (
        td.groupby("genecombination")
        .agg({size_column: "sum"})
        .sort_values(by=size_column, ascending=False)
        .reset_index()
    )
    td["genecombination"] = td.apply(
        lambda x: [x[size_column]] + x["genecombination"].split("|"), axis=1
    )

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
            gapped_ribbons(d, ax=ax, gapwidth=barwidth)
        else:
            gapped_ribbons(d, ax=ax, gapwidth=0.1)

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

    return ax


def gapped_ribbons(
    data: list,
    *,
    ax: Union[plt.axes, list, None] = None,
    xstart: float = 1.2,
    gapfreq: float = 1.0,
    gapwidth: float = 0.4,
    fun: Callable = lambda x: x[3]
    + (x[4] / (1 + np.exp(-((x[5] / x[2]) * (x[0] - x[1]))))),
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
) -> plt.Axes:
    """Draws ribbons using `fill_between`
    Called by VDJ usage plot to connect bars. 

    Parameters
    ----------
    data
        Breakpoints defining the ribbon as a 2D matrix. Each row is an x position, columns are the lower and upper extent of the ribbon at that position.
    ax
        Custom axis, almost always called with an axis supplied.  
    xstart
        The midpoint of the first bar.
    gapfreq
        Frequency of bars. Normally a bar would be drawn at every integer x position, hence default is 1.
    gapwidth
        At every bar position, there will be a gap. The width of this gap is identical to bar widths, but could also be set to 0 if we need continous ribbons.
    fun
        A function defining the curve of each ribbon segment from breakpoint to breakpoint. By default, it is a sigmoid with 6 parameters:
            range between x position of bars,
            curve start on the x axis,
            slope of curve,
            curve start y position,
            curve end y position,
            compression factor of the sigmoid curve
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    
    Returns
    -------
    Axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
    else:
        if isinstance(ax, list):
            ax = ax[0]

    spread = 10
    xw = gapfreq - gapwidth
    slope = xw * 0.8
    x, y1, y2 = [], [], []
    for i in range(1, len(data)):
        xmin = xstart + (i - 1) * gapfreq
        tx = np.linspace(xmin, xmin + xw, 100)
        xshift = xmin + xw / 2
        p1, p2 = data[i - 1]
        p3, p4 = data[i]
        ty1 = fun((tx, xshift, slope, p1, p3 - p1, spread))
        ty2 = fun((tx, xshift, slope, p2, p4 - p2, spread))
        x += tx.tolist()
        y1 += ty1.tolist()
        y2 += ty2.tolist()
        x += np.linspace(xmin + xw, xstart + i * gapfreq, 10).tolist()
        y1 += np.zeros(10).tolist()
        y2 += np.zeros(10).tolist()
    ax.fill_between(x, y1, y2, alpha=0.6)

    return ax
