from anndata import AnnData
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, Sequence
import numpy as np
from ..util import _normalize_counts
from .styling import _init_ax


def vdj_usage(
    adata: AnnData,
    *,
    vdj_cols: list = [
        "TRA_1_j_gene",
        "TRA_1_v_gene",
        "TRB_1_v_gene",
        "TRB_1_d_gene",
        "TRB_1_j_gene",
    ],
    normalize_to: Union[None, str, Sequence[float]] = None,
    ax: Union[plt.Axes, None] = None,
    bar_clip: int = 5,
    top_n: Union[None, int] = 10,
    barwidth: float = 0.4,
    draw_bars: bool = True,
    full_combination: bool = True,
    fig_kws: Union[dict, None] = None,
) -> plt.Axes:
    """Creates a ribbon plot of the most abundant VDJ combinations.

    Currently works with primary alpha and beta chains only.
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    vdj_cols
        Columns containing gene segment information.
        Overwrite default only if you know what you are doing!         
    normalize_to
        Either the name of a categorical column that should be used as the base
        for computing fractions, or an iterable specifying a size factor for each cell.
        By default, each cell count as 1, but due to normalization to different, for
        instance, sample sizes, it is possible that one cell
        in a small sample is weighted more than a cell in a large sample.
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
    full_combination
        If set to `False`, the bands represent the frequency of a binary gene segment 
        combination of the two connectec loci (e.g. combination of TRBD and TRBJ genes). 
        By default each band represents an individual combination of all five loci.
    fig_kws
        Dictionary of keyword args that will be passed to the matplotlib 
        figure (e.g. `figsize`)

    Returns
    -------
    Axes object. 
    """

    df = adata.obs.assign(
        cell_weights=_normalize_counts(adata.obs, normalize_to)
        if isinstance(normalize_to, (bool, str)) or normalize_to is None
        else normalize_to
    )

    # Init figure
    default_figargs = {"figsize": (7, 4)}
    if fig_kws is not None:
        default_figargs.update(fig_kws)
    if ax is None:
        ax = _init_ax(default_figargs)

    if top_n is None:
        top_n = df.shape[0]
    bar_text_clip = bar_clip + 1
    gene_tops, gene_colors = dict(), dict()

    # Draw a stacked bar for every gene loci and save positions on the bar
    for i, col in enumerate(vdj_cols):
        td = (
            df.groupby(col)
            .agg({"cell_weights": "sum"})
            .sort_values(by="cell_weights", ascending=False)
            .reset_index()
        )
        genes = td[col].tolist()
        td = td["cell_weights"]
        sector = col[2:7].replace("_", "")
        if full_combination:
            unct = td[bar_clip + 1 :,].sum()
            if td.size > bar_clip:
                if draw_bars:
                    ax.bar(i + 1, unct, width=barwidth, color="grey", edgecolor="black")
                gene_tops["other_" + sector] = unct
                bottom = unct
            else:
                gene_tops["other_" + sector] = 0
                bottom = 0
        else:
            bottom = 0
            bar_clip = td.shape[0]
        for j in range(bar_clip + 1):
            try:
                y = td[bar_clip - j]
                gene = genes[bar_clip - j]
                if gene == "None":
                    gene = "No_" + sector
                gene_tops[gene] = bottom + y
                if full_combination:
                    bcolor = "lightgrey"
                else:
                    bcolor = None
                if draw_bars:
                    barcoll = ax.bar(
                        i + 1,
                        y,
                        width=barwidth,
                        bottom=bottom,
                        color=bcolor,
                        edgecolor="black",
                    )
                    gene_colors[gene] = barcoll.patches[-1].get_facecolor()
                    if (bar_clip - bar_text_clip) < j:
                        ax.text(
                            1 + i - barwidth / 2 + 0.05,
                            bottom + 0.05,
                            gene.replace("TRA", "").replace("TRB", ""),
                        )
                bottom += y
            except:
                pass

    # Create a case for full combinations or just the neighbours
    if full_combination:
        draw_mat = [vdj_cols]
    else:
        draw_mat = []
        for lc in range(1, len(vdj_cols)):
            draw_mat.append(vdj_cols[lc - 1 : lc + 1])

    init_n = 0
    for target_pair in draw_mat:
        gene_tops_c = gene_tops.copy()
        init_n += 1
        # Count occurance of individual VDJ combinations
        td = df.loc[:, target_pair + ["cell_weights"]]
        td["genecombination"] = td.apply(
            lambda x, y: "|".join([x[e] for e in y]), y=target_pair, axis=1
        )
        td = (
            td.groupby("genecombination")
            .agg({"cell_weights": "sum"})
            .sort_values(by="cell_weights", ascending=False)
            .reset_index()
        )
        td["genecombination"] = td.apply(
            lambda x: [x["cell_weights"]] + x["genecombination"].split("|"), axis=1
        )

        # Draw ribbons
        for r in td["genecombination"][0 : top_n + 1]:
            d = []
            ht = r[0]
            ribcol = None
            for i in range(len(r) - 1):
                g = r[i + 1]
                sector = target_pair[i][2:7].replace("_", "")
                if g == "None":
                    g = "No_" + sector
                if g not in gene_tops_c:
                    g = "other_" + sector
                if full_combination:
                    pass
                else:
                    if ribcol is None:
                        ribcol = gene_colors[g]
                t = gene_tops_c[g]
                d.append([t - ht, t])
                t = t - ht
                gene_tops_c[g] = t
            if draw_bars:
                gapped_ribbons(
                    d,
                    ax=ax,
                    gapwidth=barwidth,
                    xstart=init_n + (barwidth / 2),
                    ribcol=ribcol,
                )
            else:
                gapped_ribbons(
                    d, ax=ax, gapwidth=0.1, xstart=init_n + 0.05, ribcol=ribcol
                )

    # Make tick labels nicer
    ax.set_xticks(range(1, len(vdj_cols) + 1))
    if vdj_cols == [
        "TRA_1_j_gene",
        "TRA_1_v_gene",
        "TRB_1_v_gene",
        "TRB_1_d_gene",
        "TRB_1_j_gene",
    ]:
        ax.set_xticklabels(["TRAJ", "TRAV", "TRBV", "TRBD", "TRBJ"])
    else:
        ax.set_xticklabels(vdj_cols)

    return ax


def gapped_ribbons(
    data: list,
    *,
    ax: Union[plt.axes, list, None] = None,
    xstart: float = 1.2,
    gapfreq: float = 1.0,
    gapwidth: float = 0.4,
    ribcol: Union[str, Tuple, None] = None,
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
    ribcol
        Face color of the ribbon.
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
    if ribcol is None:
        ax.fill_between(x, y1, y2, alpha=0.6)
    else:
        ax.fill_between(x, y1, y2, color=ribcol, alpha=0.6)

    return ax
