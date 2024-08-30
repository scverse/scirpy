from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from scirpy.get import airr as get_airr
from scirpy.io import AirrCell
from scirpy.util import DataHandler, _is_na, _normalize_counts

from .styling import _init_ax


def _sanitize_gene_name(gene_text):
    """Reformat a VDJ gene name to be displayed in the plot"""
    for tmp_chain in AirrCell.VALID_LOCI:
        gene_text = gene_text.replace(tmp_chain, "")
    return gene_text


@DataHandler.inject_param_docs()
def vdj_usage(
    adata: DataHandler.TYPE,
    *,
    vdj_cols: Sequence = (
        "VJ_1_v_call",
        "VJ_1_j_call",
        "VDJ_1_v_call",
        "VDJ_1_d_call",
        "VDJ_1_j_call",
    ),
    normalize_to: bool | str | Sequence[float] = False,
    ax: plt.Axes | None = None,
    max_segments: int | None = None,
    max_labelled_segments: int | None = 5,
    max_ribbons: None | int = 10,
    barwidth: float = 0.4,
    draw_bars: bool = True,
    full_combination: bool = True,
    fig_kws: dict | None = None,
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
) -> plt.Axes:
    """\
    Creates a ribbon plot of the most abundant VDJ combinations.

    Currently works with primary alpha and beta chains only.

    Parameters
    ----------
    {adata}
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
    max_segments
        The maximum number of segments in a bar (number of different V, D or J segments
        that should be shown separately).
    max_labelled_segments
        The maximum number of segments that receive a gene label
    max_ribbons
        The maximum number of ribbons (individual VDJ combinations). If set to `None`,
        all ribbons are drawn. If `full_combination=False`, the max number of ribbons
        is additionally constrained by the number of segments.
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
    {airr_mod}
    {airr_key}
    {chain_idx_key}

    Returns
    -------
    Axes object.
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    vdj_cols = [x.replace("IR_", "") for x in vdj_cols]
    chains, airr_variables = zip(
        *[
            (f"{arm}_{chain}", airr_variable)
            for arm, chain, airr_variable in (x.split("_", maxsplit=2) for x in vdj_cols)
        ],
        strict=False,
    )

    tmp_obs = (
        params.get_obs([normalize_to]).reindex(params.adata.obs_names)
        if isinstance(normalize_to, str)
        else params.adata.obs
    )
    df = get_airr(params, airr_variables, chains).assign(
        # make sure this also works with mudata columns:
        cell_weights=_normalize_counts(tmp_obs, normalize_to) if isinstance(normalize_to, bool | str) else normalize_to
    )
    for col in df.columns:
        if col.startswith("VJ") or col.startswith("VDJ"):
            df[col] = df[col].astype(str)

    # Init figure
    default_figargs = {"figsize": (7, 4)}
    if fig_kws is not None:
        default_figargs.update(fig_kws)
    if ax is None:
        ax = _init_ax(default_figargs)

    # the number of segments or ribbons can't exceed the number of cells
    if max_ribbons is None:
        max_ribbons = df.shape[0]
    if max_segments is None:
        max_segments = df.shape[0]
    if max_labelled_segments is None:
        max_labelled_segments = df.shape[0]

    # Store segments and colors of segments for each column individually.
    gene_tops = {col: {} for col in vdj_cols}
    gene_colors = {col: {} for col in vdj_cols}

    # Draw a stacked bar for every gene loci and save positions on the bar
    for col_idx, col_name in enumerate(vdj_cols):
        gene_weights = (
            df.groupby(col_name, observed=True)
            .agg({"cell_weights": "sum"})
            .sort_values(by="cell_weights", ascending=False)
            .reset_index()
        )
        genes = gene_weights[col_name].tolist()
        segment_sizes = gene_weights["cell_weights"]

        # Draw "other genes" segment that sucks up the genes omitted due to
        # `max_segments`.
        if segment_sizes.size > max_segments:
            other_genes_segment_size = segment_sizes[max_segments:].sum()
            if draw_bars:
                ax.bar(
                    col_idx + 1,
                    other_genes_segment_size,
                    width=barwidth,
                    color="lightgrey",
                    edgecolor="black",
                )
            gene_tops[col_name]["other"] = other_genes_segment_size
            bottom = other_genes_segment_size
        else:
            gene_tops[col_name]["other"] = 0
            bottom = 0

        # Draw gene segments
        for i, (segment_size, gene) in list(enumerate(zip(segment_sizes, genes, strict=False)))[:max_segments][::-1]:
            if _is_na(gene):
                gene = "none"
            gene_tops[col_name][gene] = bottom + segment_size
            if draw_bars:
                bar_colors = ax.bar(
                    col_idx + 1,
                    segment_size,
                    width=barwidth,
                    bottom=bottom,
                    color="lightgrey" if full_combination else None,
                    edgecolor="black",
                )
                gene_colors[col_name][gene] = bar_colors.patches[-1].get_facecolor()

                gene_text = _sanitize_gene_name(gene)
                if i < max_labelled_segments:
                    ax.text(
                        1 + col_idx - barwidth / 2 + 0.05,
                        bottom + 0.05,
                        gene_text,
                    )
            bottom += segment_size

    # Create a case for full combinations or just the neighbours
    if full_combination:
        draw_mat = [list(vdj_cols)]
    else:
        draw_mat = []
        for lc in range(1, len(vdj_cols)):
            draw_mat.append(vdj_cols[lc - 1 : lc + 1])

    # Draw ribbons
    for ribbon_start_x_coord, target_pair in enumerate(draw_mat, start=1):
        # Count occurance of individual VDJ combinations
        gene_combinations = (
            df.groupby(target_pair, observed=True)
            .agg({"cell_weights": "sum"})
            .reset_index()
            .sort_values(
                by=[
                    "cell_weights",
                    target_pair[0],
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        # need to work on a copy, otherwise the ribbons will appear shifted if
        # full_combination=False
        tmp_gene_tops = deepcopy(gene_tops)

        for _, ribbon in gene_combinations.iloc[0:max_ribbons, :].iterrows():
            ribbon_breakpoints = []
            height = ribbon["cell_weights"]

            # set ribbon color to the color of the first column in a column pair.
            # In case of `full_combination`, assign new colors.
            if full_combination:
                ribbon_color = None
            else:
                try:
                    tmp_gene = ribbon[target_pair[0]]
                    tmp_col = target_pair[0]
                    tmp_gene = "none" if _is_na(tmp_gene) else tmp_gene
                    ribbon_color = gene_colors[tmp_col][tmp_gene]
                except KeyError:
                    # Don't draw ribbon if the source gene is not drawn.
                    continue

            for col_name in target_pair:
                gene = ribbon[col_name]
                if _is_na(gene):
                    gene = "none"
                if gene not in tmp_gene_tops[col_name]:
                    gene = "other"
                top = tmp_gene_tops[col_name][gene]
                ribbon_breakpoints.append((top - height, top))
                # this is in case there are multiple ribbons originating from
                # a single segment.
                tmp_gene_tops[col_name][gene] = top - height

            if draw_bars:
                gapwidth = barwidth
                xstart = ribbon_start_x_coord + (barwidth / 2)
            else:
                gapwidth = 0.1
                xstart = ribbon_start_x_coord + 0.05

            _gapped_ribbons(
                ribbon_breakpoints,
                ax=ax,
                gapwidth=gapwidth,
                xstart=xstart,
                ribcol=ribbon_color,
            )

    ax.set_xticks(range(1, len(vdj_cols) + 1))
    ax.set_xticklabels([x.replace("IR_", "").replace("_gene", "") for x in vdj_cols])
    ax.grid(False)

    return ax


def _gapped_ribbons(
    data: list,
    *,
    ax: plt.Axes | list | None = None,
    xstart: float = 1.2,
    gapfreq: float = 1.0,
    gapwidth: float = 0.4,
    ribcol: str | tuple | None = None,
    fun: Callable = lambda x: x[3] + (x[4] / (1 + np.exp(-((x[5] / x[2]) * (x[0] - x[1]))))),
    figsize: tuple[float, float] = (3.44, 2.58),
    dpi: int = 300,
) -> plt.Axes:
    """Draws ribbons using `fill_between`
    Called by VDJ usage plot to connect bars.

    Parameters
    ----------
    data
        Breakpoints defining the ribbon as a 2D matrix. Each row is an x position,
        columns are the lower and upper extent of the ribbon at that position.
    ax
        Custom axis, almost always called with an axis supplied.
    xstart
        The midpoint of the first bar.
    gapfreq
        Frequency of bars. Normally a bar would be drawn at every integer x position,
        hence default is 1.
    gapwidth
        At every bar position, there will be a gap. The width of this gap is identical
         to bar widths, but could also be set to 0 if we need continous ribbons.
    ribcol
        Face color of the ribbon.
    fun
        A function defining the curve of each ribbon segment from breakpoint to
        breakpoint. By default, it is a sigmoid with 6 parameters:
         * range between x position of bars,
         * curve start on the x axis,
         * slope of curve,
         * curve start y position,
         * curve end y position,
         * compression factor of the sigmoid curve
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi.

    Returns
    -------
    Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        if isinstance(ax, list):
            ax = ax[0]

    ax = cast(plt.Axes, ax)

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
        ax.fill_between(x, y1, y2, alpha=0.6)  # type: ignore
    else:
        ax.fill_between(x, y1, y2, color=ribcol, alpha=0.6)  # type: ignore

    return ax
