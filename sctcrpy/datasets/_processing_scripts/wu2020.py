# %load_ext autoreload
# %autoreload 2
import scanpy as sc
import sys

sys.path.append("..")
import sctcrpy as st
from multiprocessing import Pool
import os
import pandas as pd
from glob import glob

# + language="bash"
# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file"
# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file&file=GSE139555%5Fall%5Fmetadata%2Etxt%2Egz"
# tar xvf GSE139555_raw.tar

# + language="bash"
# cd GSM4143662
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

mtx_paths = glob("GSM4143662/GSM*")

mtx_paths

metadata = pd.read_csv(
    "GSM4143662/GSE139555_all_metadata.txt.gz", sep="\t", index_col=0
)

metadata = metadata[["ident", "patient", "sample", "source", "clonotype"]]

metadata = metadata.rename(columns={"clonotype": "clonotype_orig"})

metadata


def _load_adata(path):
    sample_id = path.split("-")[-1].upper()
    obs = metadata.loc[metadata["sample"] == sample_id, :]
    adata = sc.read_10x_mtx(path)
    adata_tcr = st.read_10x_vdj_csv(
        os.path.join(path, "filtered_contig_annotations.csv.gz")
    )
    adata.obs_names = [
        "{}_{}".format(sample_id, barcode) for barcode in adata.obs_names
    ]
    adata_tcr.obs_names = [
        "{}_{}".format(sample_id, barcode) for barcode in adata_tcr.obs_names
    ]
    adata.obs = adata.obs.join(obs, how="left")
    st.pp.merge_with_tcr(adata, adata_tcr)
    return adata


p = Pool()

adatas = p.map(_load_adata, mtx_paths)

adata = adatas[0].concatenate(adatas[1:])

import numpy as np

np.sum(adata.obs["has_tcr"])

adata.write_h5ad("wu2020.h5ad", compression="lzf")

adata.obs
