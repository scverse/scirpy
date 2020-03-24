import scanpy as sc
from anndata import AnnData
import igraph as ig
import numpy as np
from .._util import get_igraph_from_adjacency


def clonotype_network(
    adata, color, *, neighbors_key="neighbors", obsm_key="X_clonotype_network"
):
    """Plot the clonotype network"""
    idx = np.where(~np.any(np.isnan(adata.obsm[obsm_key]), axis=1))[0]
    adj = adata.uns["sctcrpy"][neighbors_key]["connectivities"][idx, :][:, idx]
    g = get_igraph_from_adjacency(adj)
    layout = ig.Layout(adata.obsm[obsm_key][idx, :].tolist())
    # return g, layout
    categories = adata.obs.iloc[idx, :][color].astype("category").cat.codes.values
    coloring = ig.VertexClustering(g, membership=categories)
    return ig.plot(coloring, layout=layout, edge_arrow_size=0, vertex_size=4)
