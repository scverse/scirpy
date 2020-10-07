# %load_ext autoreload
# %autoreload 2
import scanpy as sc
import sys

sys.path.insert(0, "../../..")
import scirpy as ir
from multiprocessing import Pool
import os
import pandas as pd
from glob import glob
import numpy as np

# + language="bash"
# mkdir -p data
# cd data
# wget -O GSE139555_raw.tar "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file"
# wget -O GSE139555_tcell_metadata.txt.gz "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file&file=GSE139555%5Ftcell%5Fmetadata%2Etxt%2Egz"
# tar xvf GSE139555_raw.tar

# + language="bash"
# cd data
# for f in *.matrix.mtx.gz; do
#   dirname=${f/\.matrix\.mtx\.gz/}
#   mkdir $dirname
#   mv $dirname.genes.tsv.gz $dirname/genes.tsv.gz
#   mv $dirname.matrix.mtx.gz $dirname/matrix.mtx.gz
#   mv $dirname.barcodes.tsv.gz $dirname/barcodes.tsv.gz
#   mv $dirname.filtered_contig_annotations.csv.gz $dirname/filtered_contig_annotations.csv.gz
#   # fix missing feature type column
#   zcat $dirname/genes.tsv.gz |  awk '{print $0 "\tGene Expression"}' | gzip > $dirname/features.tsv.gz
# done
# -

mtx_paths = glob("data/GSM*")

mtx_paths

metadata_all = pd.read_csv(
    "data/GSE139555_tcell_metadata.txt.gz", sep="\t", index_col=0
)

umap = metadata_all[["UMAP_1", "UMAP_2"]]

metadata = metadata_all[["ident", "patient", "sample", "source", "clonotype"]]

metadata = metadata.rename(
    columns={"clonotype": "clonotype_orig", "ident": "cluster_orig"}
)

metadata


def _load_adata(path):
    sample_id = path.split("-")[-1].upper()
    obs = metadata.loc[metadata["sample"] == sample_id, :]
    umap_coords = umap.loc[metadata["sample"] == sample_id, :].values
    adata = sc.read_10x_mtx(path)
    adata_tcr = ir.io.read_10x_vdj(
        os.path.join(path, "filtered_contig_annotations.csv.gz")
    )
    adata.obs_names = [
        "{}_{}".format(sample_id, barcode) for barcode in adata.obs_names
    ]
    adata_tcr.obs_names = [
        "{}_{}".format(sample_id, barcode) for barcode in adata_tcr.obs_names
    ]
    # subset to T cells only
    adata = adata[obs.index, :].copy()
    adata.obs = adata.obs.join(obs, how="inner")
    assert adata.shape[0] == umap_coords.shape[0]
    adata.obsm["X_umap_orig"] = umap_coords
    ir.pp.merge_with_ir(adata, adata_tcr)
    return adata


p = Pool()
adatas = p.map(_load_adata, mtx_paths)
p.close()

adata = adatas[0].concatenate(adatas[1:])

# inverse umap X -coordinate
adata.obsm["X_umap_orig"][:, 0] = (
    np.max(adata.obsm["X_umap_orig"][:, 0]) - adata.obsm["X_umap_orig"][:, 0]
)

np.sum(adata.obs["has_ir"] == "True")

adata.write_h5ad("wu2020.h5ad", compression="lzf")

adata.obs

sc.pl.embedding(adata, "umap_orig", color="cluster_orig", legend_loc="on data")
