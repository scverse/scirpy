import scanpy as sc
import pandas as pd
from anndata import AnnData
import numpy as np
from typing import Union, Collection, Sequence, Tuple, List, Optional
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
from matplotlib import rcParams, patheffects
from matplotlib.colors import is_color_like
import scipy.sparse as sp
from pandas.api.types import is_categorical_dtype
from .styling import _get_colors

COLORMAP_EDGES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey2", ["#DDDDDD", "#000000"]
)


def _decide_legend_pos(adata, color):
    """Decide about the default legend position"""
    if color is None:
        return "none"
    elif color == "clonotype":
        return "on data"
    elif adata.obs[color].unique().size <= 50:
        return "right margin"
    else:
        return "none"


def clonotype_network(
    adata: AnnData,
    *,
    colors: Union[str, Collection[str], None] = None,
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
    edges: Union[bool, None] = None,
    edges_width: float = 0.4,
    size: Union[float, Sequence[float], None] = None,
    **kwargs,
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

    # color = [color] if isinstance(color, str) or color is None else list(color)

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

    # Networkx graph object for plotting edges
    adj_mat = clonotype_res["distances"][coords["dist_idx"].values, :][
        :, coords["dist_idx"].values
    ]
    graph = nx.Graph(_distance_to_connectivity(adj_mat))

    # uniform color
    if isinstance(colors, str) and is_color_like(colors):
        colors = [colors for c in range(coords.shape[0])]

    def _aggregate_per_dot_continuous(values):
        x_color = []
        for dist_idx in coords["dist_idx"]:
            cell_ids = clonotype_res["cell_indices"][dist_idx]
            x_color.append(np.mean(values[adata.obs_names.isin(cell_ids)]))
        return x_color

    def _aggregate_per_dot_categorical(values):
        for dist_idx in coords["dist_idx"]:
            cell_ids = clonotype_res["cell_indices"][dist_idx]
            unique, counts = np.unique(
                values[adata.obs_names.isin(cell_ids)], return_counts=True
            )

    # plot gene expression
    if use_raw is None:
        use_raw = adata.raw is not None
    var_names = adata.raw.var_names if use_raw else adata.var_names
    if isinstance(colors, str) and colors in var_names:
        x_color = []
        tmp_expr = (adata.raw if use_raw else adata)[:, colors].X
        # densify expression vector - less expensive than slicing sparse every iteration.
        if sp.issparse(tmp_expr):
            tmp_expr = tmp_expr.todense().A1
        else:
            tmp_expr = np.ravel(tmp_expr)

        colors = _aggregate_per_dot_continuous(tmp_expr)

    # plot continuous values
    if (
        isinstance(colors, str)
        and colors in adata.obs
        and not is_categorical_dtype(adata.obs[colors])
    ):
        colors = _aggregate_per_dot_continuous(adata.obs[colors])

    # plot categorical variables
    pie = False
    if (
        isinstance(colors, str)
        and colors in adata.obs
        and is_categorical_dtype(adata.obs[colors])
    ):
        cat_colors = _get_colors(adata, obs_key=colors)
        pie = True

    # Generate plot
    if not pie:
        # standard scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(coords["x"], coords["y"], s=coords["size"], c=colors)
    else:
        for ix, (xx, yy) in enumerate(zip(coords["x"], coords["y"])):
            # TODO
            pass

    # plot edges
    edge_collection = nx.draw_networkx_edges(
        graph,
        coords.loc[:, ["x", "y"]].values,
        ax=ax,
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
        for label, group_df in coords.groupby("label"):
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

    return coords

    # TODO legend
    # TODO size legend
    # TODO categorical variables (pie chart)
    # TODO continuous variables (mean)
    # TODO axis labels and title
    # TODO respect settings, frameon/off
    # TODO base size and override size

    # # for clonotype, use "on data" as default
    # if legend_loc is None:
    #     try:
    #         legend_loc = [_decide_legend_pos(adata, c) for c in color]
    #     except KeyError:
    #         raise KeyError(f"column '{color}' not found in `adata.obs`. ")

    # if isinstance(edges_cmap, str):
    #     edges_cmap = matplotlib.cm.get_cmap(edges_cmap)

    # n_displayed_cells = np.sum(~np.any(np.isnan(adata.obsm[f"X_{basis}"]), axis=1))

    # if edges is None:
    #     edges = n_displayed_cells < 1000

    # if size is None:
    #     size = 24000 / n_displayed_cells
