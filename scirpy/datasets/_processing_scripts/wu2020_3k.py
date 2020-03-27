import scanpy as sc

adata = sc.read_h5ad("wu2020.h5ad")
adata = adata[adata.obs["has_tcr"] == "True", :]
sc.pp.subsample(adata, n_obs=3000)
adata.write_h5ad("wu2020_3k.h5ad")
