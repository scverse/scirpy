import scanpy as sc
from anndata import AnnData
import numpy as np
from typing import Union, Collection, Sequence, Tuple, List
from contextlib import contextmanager
import collections.abc as cabc
import warnings
from scanpy import settings
from matplotlib.colors import Colormap
from cycler import Cycler
import matplotlib
from . import base
import matplotlib.pyplot as plt


COLORMAP_EDGES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey2", ["#DDDDDD", "#000000"]
)


@contextmanager
def _patch_plot_edges(neighbors_key, edges_cmap=None):
    """Monkey-patch scanpy's plot_edges to take our adjacency matrices"""
    scanpy_plot_edges = sc.plotting._utils.plot_edges

    def plot_edges(*args, **kwargs):
        return _plot_edges(*args, edges_cmap=edges_cmap, **kwargs)

    sc.plotting._utils.plot_edges = plot_edges
    try:
        yield
    finally:
        sc.plotting._utils.plot_edges = scanpy_plot_edges


def _plot_edges(
    axs, adata, basis, edges_width, edges_color, neighbors_key, edges_cmap=None
):
    """Add edges from a scatterplot. 

    Adapted from https://github.com/theislab/scanpy/blob/master/scanpy/plotting/_tools/scatterplots.py
    """
    import networkx as nx

    if not isinstance(axs, cabc.Sequence):
        axs = [axs]

    if neighbors_key is None:
        neighbors_key = "neighbors"
    if neighbors_key not in adata.uns:
        raise ValueError("`edges=True` requires `pp.neighbors` to be run before.")
    neighbors = adata.uns[neighbors_key]
    idx = np.where(~np.any(np.isnan(adata.obsm["X_" + basis]), axis=1))[0]
    g = nx.Graph(neighbors["connectivities"]).subgraph(idx)

    if edges_color is None:
        if edges_cmap is not None:
            edges_color = [g.get_edge_data(*x)["weight"] for x in g.edges]
        else:
            edges_color = "grey"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax in axs:
            edge_collection = nx.draw_networkx_edges(
                g,
                adata.obsm["X_" + basis],
                ax=ax,
                width=edges_width,
                edge_color=edges_color,
                edge_cmap=edges_cmap,
            )
            edge_collection.set_zorder(-2)
            edge_collection.set_rasterized(settings._vector_friendly)


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
    color: Union[str, Collection[str], None] = None,
    panel_size: Tuple[float, float] = (10, 10),
    legend_loc: str = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    basis: str = "clonotype_network",
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
    try:
        neighbors_key = adata.uns[basis]["neighbors_key"]
    except KeyError:
        raise KeyError(
            f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?"
        )
    color = [color] if isinstance(color, str) or color is None else list(color)

    if f"X_{basis}" not in adata.obsm_keys():
        raise KeyError(
            f"X_{basis} not found in `adata.obsm`. Did you run `tl.clonotype_network`?"
        )

    # for clonotype, use "on data" as default
    if legend_loc is None:
        try:
            legend_loc = [_decide_legend_pos(adata, c) for c in color]
        except KeyError:
            raise KeyError(f"column '{color}' not found in `adata.obs`. ")

    if isinstance(edges_cmap, str):
        edges_cmap = matplotlib.cm.get_cmap(edges_cmap)

    n_displayed_cells = np.sum(~np.any(np.isnan(adata.obsm[f"X_{basis}"]), axis=1))

    if edges is None:
        edges = n_displayed_cells < 1000

    if size is None:
        size = 24000 / n_displayed_cells

    with _patch_plot_edges(edges_cmap):
        return base.embedding(
            adata,
            basis=basis,
            panel_size=panel_size,
            color=color,
            legend_loc=legend_loc,
            palette=palette,
            edges_color=edges_color,
            edges_width=edges_width,
            edges=edges,
            size=size,
            neighbors_key=neighbors_key,
            **kwargs,
        )
