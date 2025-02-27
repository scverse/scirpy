import itertools
from typing import Literal

import igraph as ig
import numpy as np
from scanpy import logging
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix

from ._component_layout import layout_components
from ._fr_size_aware_layout import layout_fr_size_aware


def igraph_from_sparse_matrix(
    matrix: spmatrix,
    *,
    matrix_type: Literal["connectivity", "distance"] = "distance",
    max_value: float = None,
    simplify: bool = True,
) -> ig.Graph:
    """
    Get an igraph object from an adjacency or distance matrix.

    Parameters
    ----------
    matrix
        A sparse matrix that represents the connectivity or distance matrix for the graph.
        Zero-entries mean "no edge between the two nodes".
    matrix_type
        Whether the `sparse_matrix` represents connectivities (higher value = smaller distance)
        or distances (higher value = higher distance). Distance matrices will be
        converted into connectivities. A connectivity matrix is also known as
        weighted adjacency matrix.
    max_value
        When converting distances to connectivities, this will be considered the
        maximum distance. This defaults to `numpy.max(sparse_matrix)`.
    simplify
        Make an undirected graph and remove circular edges (i.e. edges from
        a node to itself).

    Returns
    -------
    igraph object
    """
    matrix = matrix.tocsr()

    if matrix_type == "distance":
        matrix = _distance_to_connectivity(matrix, max_value=max_value)

    return _get_igraph_from_adjacency(matrix, simplify=simplify)


def _distance_to_connectivity(distances: csr_matrix, *, max_value: float = None) -> csr_matrix:
    """Get a weighted adjacency matrix from a distance matrix.

    A distance of 1 (in the sparse matrix) corresponds to an actual distance of 0.
    An actual distance of 0 corresponds to a connectivity of 1.

    A distance of 0 (in the sparse matrix) corresponds to an actual distance of
    infinity. An actual distance of infinity corresponds to a connectivity of 0.

    Parameters
    ----------
    distances
        sparse distance matrix
    max_value
        The max_value is used to normalize the distances, i.e. distances
        are divided by this value. If not specified it will
        be the max. of the input matrix.
    """
    if not isinstance(distances, csr_matrix):
        raise ValueError("Distance matrix must be in CSR format.")

    if max_value is None:
        max_value = np.max(distances)

    connectivities = distances.copy()
    d = connectivities.data - 1

    # structure of the matrix stays the same, we can safely change the data only
    connectivities.data = (max_value - d) / max_value
    connectivities.eliminate_zeros()

    return connectivities


def _get_igraph_from_adjacency(adj: csr_matrix, simplify=True):
    """Get an undirected igraph graph from adjacency matrix.
    Better than Graph.Adjacency for sparse matrices.

    Parameters
    ----------
    adj
        sparse, weighted, symmetrical adjacency matrix.
    """
    sources, targets = adj.nonzero()
    weights = adj[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    if isinstance(weights, csr_matrix):
        # this is the case when len(sources) == len(targets) == 0, see #236
        weights = weights.toarray()

    g = ig.Graph(directed=not simplify)
    g.add_vertices(adj.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets, strict=False)))

    g.es["weight"] = weights

    if g.vcount() != adj.shape[0]:
        logging.warning(
            f"The constructed graph has only {g.vcount()} nodes. Your adjacency matrix contained redundant nodes."
        )  # type: ignore

    if simplify:
        # since we start from a symmetrical matrix, and the graph is undirected,
        # it is fine to take either of the two edges when simplifying.
        g.simplify(combine_edges="first")

    return g


def _get_sparse_from_igraph(graph, *, simplified, weight_attr=None):
    """
    Convert an igraph back to a sparse adjacency matrix

    Parameters
    ----------
    simplified
        The graph was created with the `simplify` option, i.e. the graph is undirected
        and circular edges (i.e. edges from a node to itself) are not included.
    weight_attr
        Edge attribute of the igraph object that contains the edge weight.

    Returns
    -------
    Square adjacency matrix. If the graph was directed, the matrix will be symmetric.
    """
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        weights = graph.es[weight_attr]
    shape = graph.vcount()
    shape = (shape, shape)
    if len(edges) > 0:
        adj_mat = csr_matrix((weights, list(zip(*edges, strict=False))), shape=shape)
        if simplified:
            # make symmetrical and add diagonal
            adj_mat = adj_mat + adj_mat.T - sparse.diags(adj_mat.diagonal()) + sparse.diags(np.ones(adj_mat.shape[0]))
        return adj_mat
    else:
        return csr_matrix(shape)
