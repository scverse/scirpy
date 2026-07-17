# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# +
# %load_ext autoreload
# %autoreload 2
import sys

import scanpy as sc

# +
sys.path.insert(0, "../../..")
import os
from glob import glob
from multiprocessing import Pool

import anndata
import numpy as np
import pandas as pd
from mudata import MuData

import scirpy as ir

# + language="bash"
# mkdir -p data
# cd data
# wget --no-verbose -O GSE139555_raw.tar "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file"
# wget --no-verbose -O GSE139555_tcell_metadata.txt.gz "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file&file=GSE139555%5Ftcell%5Fmetadata%2Etxt%2Egz"
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

metadata_all = pd.read_csv("data/GSE139555_tcell_metadata.txt.gz", sep="\t", index_col=0)

umap = metadata_all[["UMAP_1", "UMAP_2"]]

metadata = metadata_all[["ident", "patient", "sample", "source", "clonotype"]]

metadata = metadata.rename(columns={"clonotype": "clonotype_orig", "ident": "cluster_orig"})

metadata


def _load_adata(path):
    sample_id = path.split("-")[-1].upper()
    obs = metadata.loc[metadata["sample"] == sample_id, :]
    umap_coords = umap.loc[metadata["sample"] == sample_id, :].values
    adata = sc.read_10x_mtx(path)
    adata_tcr = ir.io.read_10x_vdj(os.path.join(path, "filtered_contig_annotations.csv.gz"))
    adata.obs_names = [f"{sample_id}_{barcode}" for barcode in adata.obs_names]
    adata_tcr.obs_names = [f"{sample_id}_{barcode}" for barcode in adata_tcr.obs_names]
    # subset to cells with annotated metadata only
    adata = adata[obs.index, :].copy()
    # all metadata except clonotyp_orig in GEX modality
    adata.obs = adata.obs.join(obs.drop(columns=["clonotype_orig"]), how="inner")
    assert adata.shape[0] == umap_coords.shape[0]
    adata.obsm["X_umap_orig"] = umap_coords
    # #356: workaround for https://github.com/scverse/muon/issues/93
    adata_tcr.X = np.ones((adata_tcr.shape[0], 0))
    # clonotype orig column in TCR modality
    adata_tcr.obs = adata_tcr.obs.join(obs.loc[:, ["clonotype_orig"]], how="left", validate="one_to_one")
    return adata, adata_tcr


p = Pool()
adatas = p.map(_load_adata, mtx_paths)
p.close()

adatas, adatas_airr = zip(*adatas, strict=False)

adata = anndata.concat(adatas)

adata_airr = anndata.concat(adatas_airr)

# inverse umap X -coordinate
adata.obsm["X_umap_orig"][:, 0] = np.max(adata.obsm["X_umap_orig"][:, 0]) - adata.obsm["X_umap_orig"][:, 0]

mdata = MuData({"gex": adata, "airr": adata_airr})

mdata

adata.obs

adata_airr.obs

mdata.obs

sc.pl.embedding(adata, "umap_orig", color="cluster_orig", legend_loc="on data")

mdata.write_h5mu("wu2020.h5mu", compression="lzf")
