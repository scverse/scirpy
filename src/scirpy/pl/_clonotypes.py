import warnings
from collections.abc import Sequence
from typing import cast

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib import patheffects, rcParams, ticker
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, is_color_like
from mudata import MuData
from pandas.api.types import is_categorical_dtype
from scanpy import settings
from scanpy.plotting._utils import ticks_formatter
from scipy.sparse import issparse

from scirpy.tl._clonotypes import _doc_clonotype_network, _graph_from_coordinates
from scirpy.util import DataHandler, read_cell_indices
from scirpy.util.graph import _distance_to_connectivity

from .styling import _get_colors, _init_ax

COLORMAP_EDGES = matplotlib.colors.LinearSegmentedColormap.from_list("grey2", ["#CCCCCC", "#000000"])


@DataHandler.inject_param_docs(clonotype_network=_doc_clonotype_network)
def clonotype_network(
    adata: DataHandler.TYPE,
    *,
    color: str | Sequence[str] | None = None,
    basis: str = "clonotype_network",
    panel_size: tuple[float, float] = (10, 10),
    color_by_n_cells: bool = False,
    scale_by_n_cells: bool = True,
    base_size: float | None = None,
    size_power: float | None = None,
    use_raw: bool | None = None,
    show_labels: bool = True,
    label_fontsize: int | None = None,
    label_fontweight: str = "bold",
    label_fontoutline: int = 3,
    label_alpha: float = 0.6,
    label_y_offset: float = 2,
    legend_fontsize=None,
    legend_width: float = 2,
    show_legend: bool | None = None,
    show_size_legend: bool = True,
    palette: str | Sequence[str] | Cycler | None = None,
    cmap: str | Colormap | None = None,
    edges_color: str | None = None,
    edges_cmap: Colormap | str = COLORMAP_EDGES,
    edges: bool = True,
    edges_width: float = 0.4,
    frameon: bool | None = None,
    title: str | Sequence[str] | None = None,
    ax: Axes | None = None,
    fig_kws: dict | None = None,
    airr_mod: str = "airr",
) -> plt.Axes:
    """\
    Plot the :term:`Clonotype` network.

    Requires running :func:`scirpy.tl.clonotype_network` first, to
    compute the layout.

    {clonotype_network}

    When the network is colored by continuous variables (genes, or numeric columns
    from `obs`), the average of the cells in each dot is computed. When the network
    is colored by categorical variables (categorical columns from `obs`), different
    categories per dot are visualized as pie chart.

    The layouting algorithm of :func:`scirpy.tl.clonotype_network` takes point sizes
    into account. For this reason, we recommend providing `base_size` and `size_power`
    already to the tool function.

    Parameters
    ----------
    {adata}
    color
        Keys for annotations of observations/cells or variables/genes,
        e.g. `patient` or `CD8A`.
    basis
        Key under which the graph layout coordinates are stored in `adata.obsm`.
    panel_size
        Size of the main figure panel in inches.
    color_by_n_cells
        Color the nodes by the number of cells they represent. This overrides
        the `color` option.
    scale_by_n_cells
        Scale the nodes by the number of cells they represent. If this is
        set to `True`, we recommend using a "size-aware" layout in
        :func:`scirpy.tl.clonotype_network` to avoid overlapping nodes (default).
    base_size
        Size of a point representing 1 cell. Per default, the value provided
        to :func:`scirpy.tl.clonotype_network` is used. This option allows to
        override this value without recomputing the layout.
    size_power
        Point sizes are raised to the power of this value. Per default, the
        value provided to :func:`scirpy.tl.clonotype_network` is used. This option
        allows to override this value without recomputing the layout.
    use_raw
        Use `adata.raw` for plotting gene expression values. Default: Use `adata.raw`
        if it exists, and `adata` otherwise.
    show_labels
        If `True` plot clonotype ids on top of the subnetworks.
    label_fontsize
        Fontsize for the clonotype labels
    label_fontweight
        Fontweight for the clonotype labels
    label_fontoutline
        Size of the fontoutline added to the clonotype labels. Set to `None` to disable.
    label_alpha
        Transparency of the clonotype labels
    label_y_offset
        Offset the clonotype label on the y axis for better visibility of the
        subnetworks.
    legend_fontsize
        Font-size for the legend.
    show_legend
        Whether to show a legend (when plotting categorical variables)
        or a colorbar (when plotting continuous variables) on the right margin.
        Per default, a legend is shown if the number of categories is smaller than
        50, other wise no legend is shown.
    show_legend_size
        Whether to show a legend for dot sizes on the right margin.
        This option is only applicable if `scale_by_n_cells` is `True`.
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …) or a :class:`~cycler.Cycler` object.
        a different color map for each panel.
    cmap
        Colormap to use for plotting continuous variables.
    edges_color
        Color of the edges. Set to `None` to color by connectivity and use the
        color map provided by `edges_cmap`.
    edges_cmap
        Colormap to use for coloring edges by connectivity.
    edges
        Whether to show the edges or not.
    edges_width
        width of the edges
    frameon
        Whether to show a frame around the plot
    title
        The main plot title
    ax
        Add the plot to a predefined Axes object.
    cax
        Add the colorbar (if any) to this predefined Axes object.
    fig_kws
        Parameters passed to the :func:`matplotlib.pyplot.figure` call
        if no `ax` is specified.
    {airr_mod}

    Returns
    -------
    A list of axes objects, containing one
    element for each `color`, or None if `show == True`.

    """
    # The plotting code borrows a lot from scanpy.plotting._tools.paga._paga_graph.
    params = DataHandler(adata, airr_mod)
    params.strings_to_categoricals()
    try:
        clonotype_key = params.adata.uns[basis]["clonotype_key"]
        base_size = params.adata.uns[basis]["base_size"] if base_size is None else base_size
        size_power = params.adata.uns[basis]["size_power"] if size_power is None else size_power
    except KeyError:
        raise KeyError(f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?") from None
    if f"X_{basis}" not in params.adata.obsm_keys():
        raise KeyError(f"X_{basis} not found in `adata.obsm`. Did you run `tl.clonotype_network`?")
    if clonotype_key not in params.adata.obs.columns:
        raise KeyError(f"{clonotype_key} not found in adata.obs.")
    if clonotype_key not in params.adata.uns:
        raise KeyError(f"{clonotype_key} not found in adata.uns.")

    if frameon is None:
        frameon = settings._frameon

    if show_legend is None:
        show_legend = True
        if color is not None:
            try:
                color_col = params.get_obs(color)
                if is_categorical_dtype(color_col) and color_col.nunique() >= 50:
                    show_legend = False
            except KeyError:
                pass

    clonotype_res = params.adata.uns[clonotype_key]
    coords, adj_mat = _graph_from_coordinates(params.adata, clonotype_key, basis)
    nx_graph = nx.Graph(_distance_to_connectivity(adj_mat))
    # in 2.6 networkx added functionality to draw self-loops. We don't want
    # them plotted, so we remove them here
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))

    # Prepare figure
    if ax is None:
        fig_kws = {} if fig_kws is None else fig_kws
        fig_width = panel_size[0] if not (show_legend or show_size_legend) else panel_size[0] + legend_width + 0.5
        fig_kws.update({"figsize": (fig_width, panel_size[1])})
        ax = _init_ax(fig_kws)

    if title is None and color is not None:
        title = color
    ax.set_frame_on(frameon)
    ax.set_xticks([])
    ax.set_yticks([])

    _plot_clonotype_network_panel(
        params,
        ax,
        legend_width=legend_width,
        color=color,
        coords=coords,
        use_raw=use_raw,
        cell_indices=clonotype_res["cell_indices"],
        nx_graph=nx_graph,
        show_legend=show_legend,
        show_size_legend=show_size_legend,
        show_labels=show_labels,
        label_fontsize=label_fontsize,
        label_fontoutline=label_fontoutline,
        label_fontweight=label_fontweight,
        legend_fontsize=legend_fontsize,
        base_size=base_size,
        size_power=size_power,
        cmap=cmap,
        edges=edges,
        edges_width=edges_width,
        edges_color=edges_color,
        edges_cmap=edges_cmap,
        title=title,
        palette=palette,
        label_alpha=label_alpha,
        label_y_offset=label_y_offset,
        scale_by_n_cells=scale_by_n_cells,
        color_by_n_cells=color_by_n_cells,
    )
    return ax


def _plot_size_legend(size_legend_ax: Axes, *, sizes, size_power, base_size, n_dots=4):
    sizes = np.unique(sizes)
    dot_sizes = sizes**size_power * base_size
    n_dots = min(n_dots, len(dot_sizes))
    min_size = np.min(dot_sizes)
    max_size = min(np.max(dot_sizes), 800)
    diff = max_size - min_size
    # special case if only one dot size.
    if n_dots <= 1:
        dot_sizes = np.array([base_size**size_power])
    else:
        step = diff / (n_dots - 1)
        dot_sizes = np.array(list(np.arange(min_size, max_size, step)) + [max_size])
    sizes = (dot_sizes / base_size) ** (1 / size_power)

    # plot size bar
    x_pos = np.cumsum(dot_sizes) + 10
    size_legend_ax.scatter(
        x_pos,
        np.repeat(0, n_dots),
        s=dot_sizes,
        color="gray",
        edgecolor="black",
        linewidth=0.2,
        zorder=100,
    )
    size_legend_ax.set_xticks(x_pos)
    labels = [str(int(x)) for x in np.rint(sizes)]
    size_legend_ax.set_xticklabels(labels, fontsize="small")
    size_legend_ax.set_xlim(-2 * base_size, max(x_pos) + max(dot_sizes))

    # remove y ticks and labels
    size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)

    # remove surrounding lines
    size_legend_ax.spines["right"].set_visible(False)
    size_legend_ax.spines["top"].set_visible(False)
    size_legend_ax.spines["left"].set_visible(False)
    size_legend_ax.spines["bottom"].set_visible(False)
    size_legend_ax.grid(False)

    size_legend_ax.set_title("# cells")

    xmin, xmax = size_legend_ax.get_xlim()
    size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)


# TODO: maybe this can become a public function in muon instead
def _fetch_features_mudata(
    params: DataHandler,
    keys: Sequence[str],
    use_raw: bool | None = None,
    layer: str | None = None,
) -> pd.DataFrame:
    """Fetch a feature from the corresponding modality.

    Taken from https://github.com/scverse/muon/blob/ed96be64b07957152382e9acc1c48010e94ee615/muon/_core/plot.py#L94
    """
    data = params.mdata
    obs = data.obs.loc[params.adata.obs.index.values]

    # Fetch respective features
    if not all(key in obs for key in keys):
        # {'rna': [True, False], 'prot': [False, True]}
        keys_in_mod = {m: [key in data.mod[m].var_names for key in keys] for m in data.mod}

        # .raw slots might have exclusive var_names
        if use_raw is None or use_raw:
            for i, k in enumerate(keys):
                for m in data.mod:
                    if keys_in_mod[m][i] is False and data.mod[m].raw is not None:
                        keys_in_mod[m][i] = k in data.mod[m].raw.var_names

        # e.g. color="rna:CD8A" - especially relevant for mdata.axis == -1
        mod_key_modifier: dict[str, str] = {}
        for i, k in enumerate(keys):
            mod_key_modifier[k] = k
            for m in data.mod:
                if not keys_in_mod[m][i]:
                    k_clean = k
                    if k.startswith(f"{m}:"):
                        k_clean = k.split(":", 1)[1]
                        mod_key_modifier[k] = k_clean

                    keys_in_mod[m][i] = k_clean in data.mod[m].var_names
                    if use_raw is None or use_raw:
                        if keys_in_mod[m][i] is False and data.mod[m].raw is not None:
                            keys_in_mod[m][i] = k_clean in data.mod[m].raw.var_names

        for m in data.mod:
            if np.sum(keys_in_mod[m]) > 0:
                mod_keys = np.array(keys)[keys_in_mod[m]]
                mod_keys = np.array([mod_key_modifier[k] for k in mod_keys])

                if use_raw is None or use_raw:
                    if data.mod[m].raw is not None:
                        keysidx = data.mod[m].raw.var.index.get_indexer_for(mod_keys)
                        fmod_adata = AnnData(
                            X=data.mod[m].raw.X[:, keysidx],
                            var=pd.DataFrame(index=mod_keys),
                            obs=data.mod[m].obs,
                        )
                    else:
                        if use_raw:
                            warnings.warn(f"Attibute .raw is None for the modality {m}, using .X instead")
                        fmod_adata = data.mod[m][:, mod_keys]
                else:
                    fmod_adata = data.mod[m][:, mod_keys]

                if layer is not None:
                    if isinstance(layer, dict):
                        m_layer = layer.get(m, None)
                        if m_layer is not None:
                            x = data.mod[m][:, mod_keys].layers[m_layer]
                            fmod_adata.X = x.todense() if issparse(x) else x
                            if use_raw:
                                warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    elif layer in data.mod[m].layers:
                        x = data.mod[m][:, mod_keys].layers[layer]
                        fmod_adata.X = x.todense() if issparse(x) else x
                        if use_raw:
                            warnings.warn(f"Layer='{layer}' superseded use_raw={use_raw}")
                    else:
                        warnings.warn(f"Layer {layer} is not present for the modality {m}, using count matrix instead")
                x = cast(sp.spmatrix, fmod_adata.X).toarray() if issparse(fmod_adata.X) else fmod_adata.X
                obs = obs.join(
                    pd.DataFrame(x, columns=mod_keys, index=fmod_adata.obs_names),
                    how="left",
                )

    return obs


def _plot_clonotype_network_panel(
    params: DataHandler,
    ax,
    *,
    color,
    coords,
    use_raw,
    cell_indices,
    legend_width,
    nx_graph,
    show_labels,
    show_legend,
    show_size_legend,
    label_fontsize,
    label_fontoutline,
    label_fontweight,
    legend_fontsize,
    title,
    base_size,
    size_power,
    cmap,
    edges_color,
    edges_cmap,
    edges_width,
    edges,
    palette,
    label_alpha,
    label_y_offset,
    scale_by_n_cells,
    color_by_n_cells,
):
    cell_indices = read_cell_indices(cell_indices)

    colorbar_title = "mean per dot"
    pie_colors = None
    cat_colors = None
    colorbar = False

    # uniform color
    if isinstance(color, str) and is_color_like(color):
        color = [color for c in range(coords.shape[0])]

    if isinstance(params.data, MuData):
        _color_lst = [color] if isinstance(color, str) else []
        # in the mudata case, we use a function internally used by muon
        obs = _fetch_features_mudata(params, _color_lst, use_raw)
        # special case for scirpy: we also search mdata.mod["airr"].obs by default:
        for c in _color_lst:
            if c not in obs.columns:
                try:
                    obs[c] = params.get_obs(c)
                except (AttributeError, KeyError):
                    pass
    else:
        # ... in the anndata case, we retrieve expression from X or raw manually
        if use_raw is None:
            use_raw = params.adata.raw is not None
        obs = params.adata.obs
        # store gene expression in obs
        if isinstance(color, str) and color not in obs.columns:
            tmp_expr = (params.adata.raw if use_raw else params.adata)[:, color].X
            if sp.issparse(tmp_expr):
                tmp_expr = cast(np.matrix, cast(sp.spmatrix, tmp_expr).todense()).A1
            else:
                tmp_expr = np.ravel(cast(np.ndarray, tmp_expr))
            obs[color] = np.array(tmp_expr)

    def _aggregate_per_dot_continuous(values):
        x_color = []
        for dist_idx in coords["dist_idx"]:
            cell_ids = cell_indices[dist_idx]
            x_color.append(np.mean(values[obs.index.isin(cell_ids)]))
        return x_color

    if color_by_n_cells:
        color = coords["size"]
        colorbar = True
        colorbar_title = "# cells"

    if not scale_by_n_cells:
        show_size_legend = False

    # plot continuous values
    if isinstance(color, str) and color in obs and not is_categorical_dtype(obs[color]):
        color = _aggregate_per_dot_continuous(obs[color])
        colorbar = True

    # plot categorical variables
    if isinstance(color, str) and color in obs and is_categorical_dtype(obs[color]):
        pie_colors = []
        values = obs[color]
        if "nan" not in values.cat.categories:
            values = values.cat.add_categories("nan")
        values = values.fillna("nan").values
        # cycle colors for categories with many values instead of
        # coloring them in grey
        if palette is None:
            if obs[color].nunique() > len(sc.pl.palettes.default_102):
                palette = cycler(color=sc.pl.palettes.default_102)
        cat_colors = _get_colors(
            params,
            obs_key=color,
            palette=palette,
        )
        if "nan" not in cat_colors:
            cat_colors["nan"] = "lightgrey"
        for dist_idx in coords["dist_idx"]:
            cell_ids = cell_indices[dist_idx]
            unique, counts = np.unique(values[obs.index.isin(cell_ids)], return_counts=True)
            fracs = counts / np.sum(counts)
            if cat_colors is not None:
                pie_colors.append({cat_colors[c]: f for c, f in zip(unique, fracs, strict=False)})

    # create panel for legend(s)
    legend_ax = None
    size_legend_ax = None
    if show_legend or show_size_legend:
        fig = ax.get_figure()
        hspace = 0.5
        colorbar_height = 0.5
        size_legend_height = 1
        fig_width, fig_height = fig.get_size_inches()
        if fig_height < colorbar_height + size_legend_height:
            raise ValueError("Figure size too small. Must be at least 3 inches high.")
        if colorbar:
            spacer_height = (fig_height - colorbar_height - size_legend_height) / 2 / fig_height
            height_ratios = [
                spacer_height * 2,
                colorbar_height / fig_height,
                0,
                size_legend_height / fig_height,
            ]
        else:
            # categorical labels may take up as much space as they want
            height_ratios = [
                0,
                (fig_height - size_legend_height) / fig_height,
                0,
                size_legend_height / fig_height,
            ]
        ax.axis("off")
        gs = ax.get_subplotspec().subgridspec(
            4,  # rows = [spacer, legend/colorbar, spacer, size_legend]
            2,  # cols = [main panel, legend panel]
            width_ratios=[
                (fig_width - hspace - legend_width) / fig_width,
                legend_width / fig_width,
            ],
            height_ratios=height_ratios,
        )
        legend_ax = fig.add_subplot(gs[1, 1])
        size_legend_ax = fig.add_subplot(gs[3, 1])
        ax = fig.add_subplot(gs[:, 0])
        ax.grid(False)
        # remove ticks if added back with the new subplot
        ax.set_xticks([])
        ax.set_yticks([])

    # Generate plot
    sct = None
    if scale_by_n_cells:
        sizes = coords["size"] ** size_power * base_size
    else:
        sizes = np.full(coords["size"].values.shape, fill_value=base_size)
    if pie_colors is None:
        # standard scatter
        sct = ax.scatter(coords["x"], coords["y"], s=sizes, c=color, cmap=cmap)

        if colorbar and show_legend:
            plt.colorbar(
                sct,
                format=ticker.FuncFormatter(ticks_formatter),
                cax=legend_ax,
                orientation="horizontal",
            )
            legend_ax.set_title(colorbar_title)
            legend_ax.xaxis.set_tick_params(labelsize="small")

    else:
        for xx, yy, tmp_size, tmp_color in zip(coords["x"], coords["y"], sizes, pie_colors, strict=False):
            # tmp_color is a mapping (color) -> (fraction)
            cumsum = np.cumsum(list(tmp_color.values()))
            cumsum = cumsum / cumsum[-1]
            cumsum = [0] + cumsum.tolist()

            for r1, r2, color in zip(cumsum[:-1], cumsum[1:], tmp_color.keys(), strict=False):
                angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 20)
                x = [0] + np.cos(angles).tolist()
                y = [0] + np.sin(angles).tolist()

                xy = np.column_stack([x, y])
                s = np.abs(xy).max()

                sct = ax.scatter([xx], [yy], marker=xy, color=color, s=s**2 * tmp_size)

    # plot edges
    if edges:
        if edges_color is None:
            if edges_cmap is not None:
                edges_color = [nx_graph.get_edge_data(*x)["weight"] for x in nx_graph.edges]
            else:
                edges_color = "grey"
        edge_collection = nx.draw_networkx_edges(
            nx_graph,
            coords.loc[:, ["x", "y"]].values,
            ax=ax,
            width=edges_width,
            edge_color=edges_color,
            edge_cmap=edges_cmap,
        )
        if edge_collection != []:
            edge_collection.set_zorder(-1)
            edge_collection.set_rasterized(sc.settings._vector_friendly)

    # add clonotype labels
    if show_labels:
        text_kwds = {}
        if label_fontsize is None:
            label_fontsize = rcParams["legend.fontsize"]
        if label_fontoutline is not None:
            text_kwds["path_effects"] = [patheffects.withStroke(linewidth=label_fontoutline, foreground="w")]
        for label, group_df in coords.groupby("label", observed=True):
            # add label at centroid
            ax.text(
                np.mean(group_df["x"]),
                np.mean(group_df["y"]) + label_y_offset,
                label,
                verticalalignment="center",
                horizontalalignment="center",
                size=label_fontsize,
                fontweight=label_fontweight,
                alpha=label_alpha,
                **text_kwds,
            )

    # add legend for categorical colors
    if cat_colors is not None and show_legend:
        for cat, color in cat_colors.items():
            # use empty scatter to set labels
            legend_ax.scatter([], [], c=color, label=cat)
        legend_ax.legend(
            frameon=False,
            loc="center left",
            # bbox_to_anchor=(1, 0.5),
            fontsize=legend_fontsize,
            ncol=(1 if len(cat_colors) <= 14 else 2 if len(cat_colors) <= 30 else 3),
        )
        legend_ax.axis("off")

    if not show_legend and legend_ax is not None:
        legend_ax.axis("off")

    # add size legend
    if show_size_legend:
        _plot_size_legend(
            size_legend_ax,
            sizes=coords["size"],
            size_power=size_power,
            base_size=base_size,
        )
    elif legend_ax is not None:
        size_legend_ax.axis("off")

    ax.set_title(title)

    return sct
