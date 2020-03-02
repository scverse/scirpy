import scanpy as sc
from anndata import AnnData


def umap_tra(adata: AnnData, **kwargs):
    return sc.pl.embedding(adata, "X_umap_tra", **kwargs)


def umap_trb(adata: AnnData, **kwargs):
    return sc.pl.embedding(adata, "X_umap_trb", **kwargs)
