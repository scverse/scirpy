from matplotlib.axes import Axes
import scanpy as sc
import pandas as pd
from anndata import AnnData
import numpy as np
from typing import Sized, Union, Collection, Sequence, Tuple, List, Optional, Mapping
from contextlib import contextmanager
import collections.abc as cabc
import warnings
from scanpy import settings
from matplotlib.colors import Colormap
from cycler import Cycler
import matplotlib
from . import base
import matplotlib.pyplot as plt
from ..util.graph import _distance_to_connectivity
import networkx as nx
import itertools
from matplotlib import rcParams, patheffects, ticker
import matplotlib.patches as mpatches
from matplotlib.colors import is_color_like
import scipy.sparse as sp
from pandas.api.types import is_categorical_dtype
from .styling import _get_colors, _init_ax
from scanpy.plotting._utils import setup_axes, ticks_formatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


COLORMAP_EDGES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey2", ["#DDDDDD", "#000000"]
)


def clonotype_network(
    adata: AnnData,
    *,
    color: Union[str, Sequence[str], None] = None,
    basis: str = "clonotype_network",
    show_labels: bool = True,
    label_fontsize: Optional[int] = None,
    label_fontweight: str = "bold",
    label_fontoutline: int = 3,
    use_raw: Optional[bool] = None,
    panel_size: Tuple[float, float] = (10, 10),
    legend_loc: str = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    edges_color: Union[str, None] = None,
    edges_cmap: Union[Colormap, str] = COLORMAP_EDGES,
    edges: bool = True,
    edges_width: float = 0.4,
    node_size: bool = True,
    cmap: Union[str, Colormap] = None,
    base_size: float = None,
    size_power: float = 0.5,
    ax: Optional[Axes] = None,
    cax: Optional[Axes] = None,
    frameon: Optional[bool] = None,
    title: Optional[str] = None,
    fig_kws: Optional[dict] = None,
    legend_fontsize=None,
) -> List[plt.Axes]:
    """\
    Plot the :term:`Clonotype` network.

    Requires running :func:`scirpy.tl.clonotype_network` before, to
    compute the layout.

    Parameters
    ----------
    adata
        Annotated data matrix.
    color
        Keys for annotations of observations/cells or variables/genes, e.g.,
        `'ann1'` or `['ann1', 'ann2']`.
    panel_size
        Size tuple (`width`, `height`) of a single panel in inches.
    legend_loc
        Location of legend, either `'on data'`, `'right margin'` or a valid keyword
        for the `loc` parameter of :class:`~matplotlib.legend.Legend`.
        When set to `None` automatically determines the legend position based on the
        following criteria: (1) `'on data'` when coloring by `clonotype`,
        (2) `'right margin'` if the number of categories < 50, and
        (3) and `'none'` otherwise.
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …) or a :class:`~cycler.Cycler` object.
        It is possible to specify a list of the same size as `color` to choose
        a different color map for each panel.
    basis
        Key under which the graph layout coordinates are stored in `adata.obsm`
    edges_color
        Color of the edges. Set to `None` to color by connectivity and use the
        color map provided by `edges_cmap`.
    edges_cmap
        Colors to use for coloring edges by connectivity
    edges
        Whether to show the edges or not. Defaults to True for < 1000 displayed
        cells, to False otherwise.
    edges_width
        width of the edges
    size
        Point size. If `None` it is automatically computed as 24000 / `n_cells`.
        Can be a sequence containing the size for each cell. The order should be the
        same as in adata.obs. Other than in the default `scanpy` implementation
        this respects that some cells might not be shown in the plot.
    **kwargs
        Additional arguments which are passed to :func:`scirpy.pl.embedding`.

    Returns
    -------
    A list of axes objects, containing one
    element for each `color`, or None if `show == True`.

    See also
    --------
    :func:`scirpy.pl.embedding` and :func:`scanpy.pl.embedding`
    """
    # The plotting code borrows a lot from scanpy.plotting._tools.paga._paga_graph.
    adata._sanitize()
    try:
        clonotype_key = adata.uns[basis]["clonotype_key"]
    except KeyError:
        raise KeyError(
            f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?"
        )
    if f"X_{basis}" not in adata.obsm_keys():
        raise KeyError(
            f"X_{basis} not found in `adata.obsm`. Did you run `tl.clonotype_network`?"
        )
    if clonotype_key not in adata.obs.columns:
        raise KeyError(f"{clonotype_key} not found in adata.obs.")
    if clonotype_key not in adata.uns:
        raise KeyError(f"{clonotype_key} not found in adata.uns.")

    if use_raw is None:
        use_raw = adata.raw is not None

    if frameon is None:
        frameon = settings._frameon

    if legend_loc is None:
        if color in adata.obs.columns and is_categorical_dtype(adata.obs[color]):
            legend_loc = "right margin" if adata.obs[color].nunique() < 50 else "none"

    clonotype_res = adata.uns[clonotype_key]

    # map the cell-id to the corresponding row/col in the clonotype distance matrix
    dist_idx, obs_names = zip(
        *itertools.chain.from_iterable(
            zip(itertools.repeat(i), obs_names)
            for i, obs_names in enumerate(clonotype_res["cell_indices"])
        )
    )
    dist_idx_lookup = pd.DataFrame(index=obs_names, data=dist_idx, columns=["dist_idx"])
    clonotype_label_lookup = adata.obs.loc[:, [clonotype_key]].rename(
        columns={clonotype_key: "label"}
    )

    # Retrieve coordinates and reduce them to one coordinate per node
    coords = (
        adata.obsm["X_clonotype_network"]
        .dropna(axis=0, how="any")
        .join(dist_idx_lookup)
        .join(clonotype_label_lookup)
        .groupby(by=["label", "dist_idx", "x", "y"], observed=True)
        .size()
        .reset_index(name="size")
    )
    if not node_size:
        coords["size"] = 1

    if base_size is None:
        base_size = 2000 / coords.shape[0]

    # Networkx graph object for plotting edges
    adj_mat = clonotype_res["distances"][coords["dist_idx"].values, :][
        :, coords["dist_idx"].values
    ]

    nx_graph = nx.Graph(_distance_to_connectivity(adj_mat))

    # Prepare figure
    if ax is None:
        fig_kws = dict() if fig_kws is None else fig_kws
        fig_kws.update({"figsize": panel_size})
        ax = _init_ax(fig_kws)

    if title is not None:
        ax.set_title(title)
    ax.set_frame_on(frameon)
    ax.set_xticks([])
    ax.set_yticks([])
    sct = _plot_clonotype_network_panel(
        adata,
        ax,
        cax,
        color=color,
        coords=coords,
        use_raw=use_raw,
        cell_indices=clonotype_res["cell_indices"],
        nx_graph=nx_graph,
        legend_loc=legend_loc,
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
    )


def _plot_clonotype_network_panel(
    adata,
    ax,
    cax,
    *,
    color,
    coords,
    use_raw,
    cell_indices,
    legend_loc,
    nx_graph,
    show_labels,
    label_fontsize,
    label_fontoutline,
    label_fontweight,
    legend_fontsize,
    base_size,
    size_power,
    cmap,
    edges_color,
    edges_cmap,
    edges_width,
    edges,
):
    pie_colors = None
    cat_colors = None
    colorbar = False
    # uniform color
    if isinstance(color, str) and is_color_like(color):
        color = [color for c in range(coords.shape[0])]

    def _aggregate_per_dot_continuous(values):
        x_color = []
        for dist_idx in coords["dist_idx"]:
            cell_ids = cell_indices[dist_idx]
            x_color.append(np.mean(values[adata.obs_names.isin(cell_ids)]))
        return x_color

    # plot gene expression
    var_names = adata.raw.var_names if use_raw else adata.var_names
    if isinstance(color, str) and color in var_names:
        x_color = []
        tmp_expr = (adata.raw if use_raw else adata)[:, color].X
        # densify expression vector - less expensive than slicing sparse every iteration.
        if sp.issparse(tmp_expr):
            tmp_expr = tmp_expr.todense().A1
        else:
            tmp_expr = np.ravel(tmp_expr)

        color = _aggregate_per_dot_continuous(tmp_expr)
        colorbar = True

    # plot continuous values
    if (
        isinstance(color, str)
        and color in adata.obs
        and not is_categorical_dtype(adata.obs[color])
    ):
        color = _aggregate_per_dot_continuous(adata.obs[color])
        colorbar = True

    # plot categorical variables
    if (
        isinstance(color, str)
        and color in adata.obs
        and is_categorical_dtype(adata.obs[color])
    ):
        pie_colors = []
        values = adata.obs[color].values
        cat_colors = _get_colors(adata, obs_key=color)
        for dist_idx in coords["dist_idx"]:
            cell_ids = cell_indices[dist_idx]
            unique, counts = np.unique(
                values[adata.obs_names.isin(cell_ids)], return_counts=True
            )
            fracs = counts / np.sum(counts)
            pie_colors.append({cat_colors[c]: f for c, f in zip(unique, fracs)})

    # Generate plot
    sct = None
    sizes = coords["size"] ** size_power * base_size
    if pie_colors is None:
        # standard scatter
        sct = ax.scatter(coords["x"], coords["y"], s=sizes, c=color, cmap=cmap)

        if colorbar and legend_loc != "none":
            if cax is None:
                fig = ax.get_figure()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)

            cb = plt.colorbar(
                sct,
                format=ticker.FuncFormatter(ticks_formatter),
                cax=cax,
            )

    else:
        for xx, yy, tmp_size, tmp_color in zip(
            coords["x"], coords["y"], sizes, pie_colors
        ):
            # tmp_color is a mapping (color) -> (fraction)
            cumsum = np.cumsum(list(tmp_color.values()))
            cumsum = cumsum / cumsum[-1]
            cumsum = [0] + cumsum.tolist()

            for r1, r2, color in zip(cumsum[:-1], cumsum[1:], tmp_color.keys()):
                angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 20)
                x = [0] + np.cos(angles).tolist()
                y = [0] + np.sin(angles).tolist()

                xy = np.column_stack([x, y])
                s = np.abs(xy).max()

                sct = ax.scatter(
                    [xx], [yy], marker=xy, color=color, s=s ** 2 * tmp_size
                )

    # plot edges
    if edges:
        if edges_color is None:
            if edges_cmap is not None:
                edges_color = [
                    nx_graph.get_edge_data(*x)["weight"] for x in nx_graph.edges
                ]
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
        edge_collection.set_zorder(-1)
        edge_collection.set_rasterized(sc.settings._vector_friendly)

    # add clonotype labels
    if show_labels:
        text_kwds = dict()
        if label_fontsize is None:
            label_fontsize = rcParams["legend.fontsize"]
        if label_fontoutline is not None:
            text_kwds["path_effects"] = [
                patheffects.withStroke(linewidth=label_fontoutline, foreground="w")
            ]
        for label, group_df in coords.groupby("label", observed=True):
            # add label at centroid
            ax.text(
                np.mean(group_df["x"]),
                np.mean(group_df["y"]),
                label,
                verticalalignment="center",
                horizontalalignment="center",
                size=label_fontsize,
                fontweight=label_fontweight,
                **text_kwds,
            )

    # add legend for categorical colors
    if cat_colors is not None and legend_loc == "right margin":
        for cat, color in cat_colors.items():
            # use empty scatter to set labels
            ax.scatter([], [], c=color, label=cat)
        legend1 = ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=legend_fontsize,
            ncol=(1 if len(cat_colors) <= 14 else 2 if len(cat_colors) <= 30 else 3),
        )
        ax.add_artist(legend1)

    return sct
