import scipy
from scanpy import logging
import igraph as ig
import networkx
import numpy as np


def get_igraph_from_adjacency(adj: scipy.sparse.csr_matrix, edge_type: str = None):
    """Get igraph graph from adjacency matrix.
    Better than Graph.Adjacency for sparse matrices

    Parameters
    ----------
    adj
        (weighted) adjacency matrix
    edge_type
        A type attribute added to all edges
    """
    g = ig.Graph(directed=False)
    g.add_vertices(adj.shape[0])  # this adds adjacency.shape[0] vertices

    sources, targets = scipy.sparse.triu(adj, k=1).nonzero()
    weights = adj[sources, targets].astype("float")
    g.add_edges(list(zip(sources, targets)))

    g.es["weight"] = weights
    if edge_type is not None:
        g.es["type"] = edge_type

    if g.vcount() != adj.shape[0]:
        logging.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


def layout_many_components(
    graph: ig.Graph,
    component_layout_func: str = "fr",
    pad_x: float = 1.0,
    pad_y: float = 1.0,
):
    """
    Arguments:
    ----------
    graph
        The graph to plot.
    component_layout_func
        Function used to layout individual components. 
    pad_x, pad_y
        Padding between subgraphs in the x and y dimension.

    Returns:
    --------
    pos : dict node : (float x, float y)
        The layout of the graph.

    """

    components = _get_components_sorted_by_size(graph)
    component_sizes = [component.vcount() for component in components]
    bboxes = _get_component_bboxes(component_sizes, pad_x, pad_y)

    return np.vstack(
        [
            _layout_component(component, bbox, component_layout_func)
            for component, bbox in zip(components, bboxes)
        ]
    )


def _get_components_sorted_by_size(g):
    subgraphs = g.decompose(mode="weak")
    return sorted(subgraphs, key=lambda x: x.vcount())


def _get_component_bboxes(component_sizes, pad_x=1.0, pad_y=1.0):
    bboxes = []
    x, y = (0, 0)
    current_n = 1
    for n in component_sizes:
        width, height = _get_bbox_dimensions(n, power=0.8)

        if not n == current_n:  # create a "new line"
            x = 0  # reset x
            y += height + pad_y  # shift y up
            current_n = n

        bbox = x, y, width, height
        bboxes.append(bbox)
        x += width + pad_x  # shift x down the line
    return bboxes


def _get_bbox_dimensions(n, power=0.5):
    # return (np.sqrt(n), np.sqrt(n))
    return (n ** power, n ** power)


def _layout_component(component, bbox, component_layout_func):
    layout = component.layout(component_layout_func)
    rescaled_pos = _rescale_layout(np.array(layout.coords), bbox)
    return rescaled_pos


def _rescale_layout(coords, bbox):

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)

    if not min_x == max_x:
        delta_x = max_x - min_x
    else:  # graph probably only has a single node
        delta_x = 1.0

    if not min_y == max_y:
        delta_y = max_y - min_y
    else:  # graph probably only has a single node
        delta_y = 1.0

    new_min_x, new_min_y, new_delta_x, new_delta_y = bbox

    new_coords_x = (coords[:, 0] - min_x) / delta_x * new_delta_x + new_min_x
    new_coords_y = (coords[:, 1] - min_y) / delta_y * new_delta_y + new_min_y

    return np.vstack([new_coords_x, new_coords_y]).T


def test():
    from itertools import combinations

    g = ig.Graph()

    # add 100 unconnected nodes
    g.add_vertices(100)

    # add 50 2-node components
    g.add_edges([(ii, ii + 1) for ii in range(100, 200, 2)])

    # add 33 3-node components
    for ii in range(200, 300, 3):
        g.add_edges([(ii, ii + 1), (ii, ii + 2), (ii + 1, ii + 2)])

    # add a couple of larger components
    n = 300
    for ii in np.random.randint(4, 30, size=10):
        g.add_edges(combinations(range(n, n + ii), 2))
        n += ii

    layout = layout_many_components(g, component_layout_func="fr")

    ig.plot(g, layout)
