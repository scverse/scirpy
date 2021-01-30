import scipy.sparse
from scanpy import logging
import igraph as ig
import numpy as np
from .._compat import Literal
import sys


def _get_igraph_from_adjacency(adj: scipy.sparse.csr_matrix, edge_type: str = None):
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

    if isinstance(weights, np.matrix):
        weights = weights.A1
    if isinstance(weights, scipy.sparse.csr_matrix):
        # this is the case when len(sources) == len(targets) == 0, see #236
        weights = weights.toarray()

    g.es["weight"] = weights
    if edge_type is not None:
        g.es["type"] = edge_type

    if g.vcount() != adj.shape[0]:
        logging.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


def layout_components(
    graph: ig.Graph,
    component_layout: str = "fr",
    arrange_boxes: Literal["size", "rpack", "squarify"] = "squarify",
    pad_x: float = 1.0,
    pad_y: float = 1.0,
) -> np.ndarray:
    """
    Compute a graph layout by layouting all connected components individually.

    Adapted from https://stackoverflow.com/questions/53120739/lots-of-edges-on-a-graph-plot-in-python

    Parameters
    ----------
    graph
        The graph to plot.
    component_layout
        Layout function used to layout individual components.
        Can be anything that can be passed to `igraph.Graph.layout`
    arrange_boxes
        How to arrange the individual components. Can be "size"
        to arange them by the component size, or "rpack" to pack them as densly
        as possible, or "squarify" to arrange them using a treemap algorithm.
    pad_x
        Padding between subgraphs in the x dimension.
    pad_y
        Padding between subgraphs in the y dimension.

    Returns
    -------
    pos
        n_nodes x dim array containing the layout coordinates

    """
    # assign the original vertex id, it will otherwise get lost by decomposition
    for i, v in enumerate(graph.vs):
        v["id"] = i
    components = np.array(graph.decompose(mode="weak"))
    component_sizes = np.array([component.vcount() for component in components])
    order = np.argsort(component_sizes)
    components = components[order]
    component_sizes = component_sizes[order]
    vertex_ids = [v["id"] for comp in components for v in comp.vs]
    vertex_sorter = np.argsort(vertex_ids)

    bbox_fun = {"rpack": _bbox_rpack, "size": _bbox_sorted, "squarify": _bbox_squarify}[
        arrange_boxes
    ]
    bboxes = bbox_fun(component_sizes, pad_x, pad_y)

    component_layouts = [
        _layout_component(component, bbox, component_layout)
        for component, bbox in zip(components, bboxes)
    ]
    # get vertexes back into their original order
    coords = np.vstack(component_layouts)[vertex_sorter, :]
    return coords


def _bbox_rpack(component_sizes, pad_x=1.0, pad_y=1.0):
    """Compute bounding boxes for individual components
    by arranging them as densly as possible.

    Depends on `rectangle-packer`.
    """
    try:
        import rpack
    except ImportError:
        raise ImportError(
            "Using the 'components layout' requires the installation of "
            "the `rectangle-packer`. You can install it with "
            "`pip install rectangle-packer`."
        )

    dimensions = [_get_bbox_dimensions(n, power=0.8) for n in component_sizes]
    # rpack only works on integers; sizes should be in descending order
    dimensions = [
        (int(width + pad_x), int(height + pad_y))
        for (width, height) in dimensions[::-1]
    ]
    origins = rpack.pack(dimensions)
    outer_dimensions = rpack.enclosing_size(dimensions, origins)
    aspect_ratio = outer_dimensions[0] / outer_dimensions[1]
    if aspect_ratio > 1:
        scale_width, scale_height = 1, aspect_ratio
    else:
        scale_width, scale_height = aspect_ratio, 1
    bboxes = [
        (
            x,
            y,
            width * scale_width - pad_x,
            height * scale_height - pad_y,
        )
        for (x, y), (width, height) in zip(origins, dimensions)
    ]
    return bboxes[::-1]


def _bbox_squarify(component_sizes, pad_x=10, pad_y=10):
    """Arrange bounding boxes using the `squarify` implementation for treemaps"""
    try:
        import squarify
    except ImportError:
        raise ImportError(
            "Using the 'components layout' requires the installation"
            "of the `squarify` package. You can install it with "
            "`pip install squarify`"
        )
    order = np.argsort(-component_sizes)
    undo_order = np.argsort(order)
    component_sizes = component_sizes[order]
    component_sizes = squarify.normalize_sizes(component_sizes, 100, 100)
    rects = squarify.padded_squarify(component_sizes, 0, 0, 100, 100)

    bboxes = []
    for r in rects:
        width = r["dx"]
        height = r["dy"]
        offset_x = r["x"]
        offset_y = r["y"]
        delta = abs(width - height)
        if width > height:
            width = height
            offset_x += delta / 2
        else:
            height = width
            offset_y += delta / 2
        bboxes.append((offset_x, offset_y, width - pad_x, height - pad_y))

    return [bboxes[i] for i in undo_order]


def _bbox_sorted(component_sizes, pad_x=1.0, pad_y=1.0):
    """Compute bounding boxes for individual components
    by arranging them by component size"""
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
    """Compute layout for an individual component"""
    layout = component.layout(component_layout_func)
    rescaled_pos = _rescale_layout(np.array(layout.coords), bbox)
    return rescaled_pos


def _rescale_layout(coords, bbox):
    """Transpose the layout of a component into its bounding box"""
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
