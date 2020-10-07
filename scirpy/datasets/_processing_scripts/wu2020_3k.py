import scanpy as sc
import pandas as pd

# Use this list of 3k barcodes for consistency with previous versions
barcodes = pd.read_csv("./3k_barcodes.csv", header=None)[0].values

adata = sc.read_h5ad("wu2020.h5ad")
adata = adata[barcodes, :].copy()

adata.write_h5ad("wu2020_3k.h5ad", compression="lzf")
