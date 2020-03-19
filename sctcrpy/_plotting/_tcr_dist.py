import scanpy as sc
from anndata import AnnData
import igraph as ig


def clonotype_network(adata, color):
    """Plot the clonotype network"""
    g = adata.uns["sctcrpy"]["clonotype_subgraph"]
    layout = adata.uns["sctcrpy"]["clonotype_layout"]
    idx = adata.uns["sctcrpy"]["clonotype_subgraph_idx"]
    categories = adata.obs.iloc[idx, :][color].astype("category").cat.codes.values
    coloring = ig.VertexClustering(g, membership=categories)
    return ig.plot(coloring, layout=layout, edge_arrow_size=0, vertex_size=4)
