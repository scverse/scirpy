import scanpy as sc
from anndata import AnnData
import numpy as np
from typing import Union, Collection, Sequence
from contextlib import contextmanager
import collections.abc as cabc
import warnings
from scanpy import settings
from matplotlib.colors import Colormap
from cycler import Cycler
import matplotlib
from . import base

COLORMAP_EDGES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "grey2", ["#DDDDDD", "#000000"]
)


@contextmanager
def _patch_plot_edges(neighbors_key, edges_cmap=None):
    """Monkey-patch scanpy's plot_edges to take our adjacency matrices"""
    scanpy_plot_edges = sc.plotting._utils.plot_edges

    def plot_edges(*args, **kwargs):
        return _plot_edges(
            *args, neighbors_key=neighbors_key, edges_cmap=edges_cmap, **kwargs
        )

    sc.plotting._utils.plot_edges = plot_edges
    try:
        yield
    finally:
        sc.plotting._utils.plot_edges = scanpy_plot_edges


def _plot_edges(
    axs, adata, basis, edges_width, edges_color, edges_cmap=None, neighbors_key=None
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


def clonotype_network(
    adata: AnnData,
    *,
    color: Union[str, Collection[str], None] = None,
    panel_size=(10, 10),
    legend_loc: str = None,
    palette: Union[str, Sequence[str], Cycler, None] = None,
    neighbors_key="tcr_neighbors",
    basis="clonotype_network",
    edges_color: Union[str, None] = None,
    edges_cmap: Union[Colormap, str] = COLORMAP_EDGES,
    edges: bool = True,
    edges_width=0.4,
    **kwargs
):
    """\
    Plot the clonotype network
    ----------
    adata 
        annotated data matrix
    color
        Keys for annotations of observations/cells or variables/genes, e.g.,
        `'ann1'` or `['ann1', 'ann2']`.
    panel_size
        Size tuple (`width`, `height`) of a single panel in inches
    legend_loc
        Location of legend, either `'on data'`, `'right margin'` or a valid keyword
        for the `loc` parameter of :class:`~matplotlib.legend.Legend`.
        Defaults to "on data" when coloring by `clonotype` or "right margin" in all
        other cases. 
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …) or a :class:`~cycler.Cycler` object. 
        It is possible to specify a list of the same size as `color` to choose 
        a different color map for each panel. 
    neighbors_key
        Key under which the tcr neighborhood matrix is stored in `adata.uns`
    basis
        Key under which the graph layout coordinates are stored in `adata.obsm`
    edges_color
        Color of the edges. Set to `None` to color by connectivity and use the 
        color map provided by `edges_cmap`. 
    edges_cmap  
        Colors to use for coloring edges by connectivity
    edges
        Whether to show the edges or not
    edges_width
        width of the edges
    kwargs  
        Additional arguments are passed to :func:`base.embedding`. 

    Returns
    -------
     axes
        A list of axes objects, containing one
        element for each `color`, or None if `show == True`. 

    See also
    --------
    :func:`pl.embedding` and :func:`scanpy.pl.embedding`
    """
    color = [color] if isinstance(color, str) or color is None else list(color)

    # for clonotype, use "on data" as default
    if legend_loc is None:
        legend_loc = ["on data" if c == "clonotype" else "right margin" for c in color]

    if isinstance(edges_cmap, str):
        edges_cmap = matplotlib.cm.get_cmap(edges_cmap)

    with _patch_plot_edges(neighbors_key, edges_cmap):
        return base.embedding(
            adata,
            basis="clonotype_network",
            panel_size=panel_size,
            color=color,
            edges=edges,
            legend_loc=legend_loc,
            palette=palette,
            edges_color=edges_color,
            edges_width=edges_width,
            **kwargs,
        )


def clonotype_network_igraph(
    adata, neighbors_key="tcr_neighbors", basis="clonotype_network"
):
    """
    Get an `igraph` object representing the clonotype network.

    Parameters
    ----------
    adata
        annotated data matrix
    neighbors_key
        key in `adata.uns` where tcr neighborhood information is located
    basis
        key in `adata.obsm` where the network layout is stored. 
    
    Returns
    -------
    graph
        igraph object
    layout 
        corresponding igraph Layout object. 
    """
    import igraph as ig
    from .._util._graph import get_igraph_from_adjacency

    conn = adata.uns[neighbors_key]["connectivities"]
    idx = np.where(~np.any(np.isnan(adata.obsm["X_" + basis]), axis=1))[0]
    g = get_igraph_from_adjacency(conn).subgraph(idx)
    layout = ig.Layout(coords=adata.obsm["X_" + basis][idx, :].tolist())
    return g, layout
