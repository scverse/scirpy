# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
# ---

# %load_ext autoreload
# %autoreload 2

# +
# %env OPENBLAS_NUM_THREADS=16
# %env OMP_NUM_THREADS=16
# %env MKL_NUM_THREADS=16
# %env OMP_NUM_cpus=16
# %env MKL_NUM_cpus=16
# %env OPENBLAS_NUM_cpus=16
import sys

sys.path.insert(0, "../../..")

import scirpy as ir
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

# -

# The dataset has been downloaded from ENA and then processed using the Smart-seq2 Pipeline:
# https://github.com/nf-core/smartseq2/

DATASET_DIR = Path("/data/datasets/Maynard_Bivona_2020_NSCLC/")

# ### Read counts and TPMs

count_mat = pd.read_csv(
    DATASET_DIR / "smartseq2_pipeline/resultCOUNT.txt",
    sep="\t",
    low_memory=False,
    index_col="Geneid",
)

tpm_mat = pd.read_csv(
    DATASET_DIR / "smartseq2_pipeline/resultTPM.txt", sep="\t", low_memory=False
)

# summarize to gene symbol for the ~300 duplicated symbols.
tpm_mat_symbol = tpm_mat.drop("gene_id", axis="columns").groupby("gene_symbol").sum()

# ### Read and sanitize metadata

# +
sample_info = pd.read_csv(DATASET_DIR / "scripts/sra_sample_info.csv", low_memory=False)
cell_metadata = pd.read_csv(
    DATASET_DIR / "scripts/cell_metadata.csv", low_memory=False, index_col=0
)

# combine metadata
meta = sample_info.merge(
    cell_metadata, left_on="cell_ID", right_on="cell_id"
).set_index("Run")
# -

meta = meta.drop(
    [
        "Assay Type",
        "AvgSpotLen",
        "SRA Study",
        "ReleaseDate",
        "Bases",
        "disease",
        "Biomaterial_provider",
        "BioProject",
        "Isolate",
        "Sample Name",
        "BioSample",
        "BioSampleModel",
        "Bytes",
        "Center Name",
        "Consent",
        "DATASTORE filetype",
        "DATASTORE provider",
        "DATASTORE region",
        "Experiment",
        "Instrument",
        "LibraryLayout",
        "Library Name",
        "LibrarySelection",
        "cell_ID",
        "LibrarySource",
        "Organism",
        "Platform",
        "gender",
        "SAMPLE_TYPE",
        "TISSUE",
    ],
    axis="columns",
).rename(
    {
        "Age": "age",
        "smokingHx": "smoking_status",
        "stage.at.dx": "stage_at_diagnosis",
    },
    axis="columns",
)

meta.tail()

# ### Find all cells for which we have both counts, TPM and annotation

has_counts = set(count_mat.columns)
has_tpm = set(tpm_mat.columns)
has_meta = set(meta.index.values)

cell_ids = np.array(list(has_counts & has_tpm & has_meta))

# ### Build adata

var = (
    pd.DataFrame(count_mat.index)
    .rename({"Geneid": "gene_symbol"}, axis="columns")
    .set_index("gene_symbol")
    .sort_index()
)

adata = sc.AnnData(
    X=csr_matrix(tpm_mat_symbol.loc[var.index, cell_ids].values.T),
    layers={"raw_counts": csr_matrix(count_mat.loc[var.index, cell_ids].values.T)},
    var=var,
    obs=meta.loc[cell_ids, :],
)

adata_tcr = ir.io.read_tracer(
    "/data/datasets/Maynard_Bivona_2020_NSCLC/smartseq2_pipeline/TraCeR"
)

adata_bcr = ir.io.read_bracer(
    "/data/datasets/Maynard_Bivona_2020_NSCLC/smartseq2_pipeline/BraCeR/filtered_BCR_summary/changeodb.tab"
)

ir.pp.merge_with_ir(adata, adata_tcr)

ir.pp.merge_with_ir(adata, adata_bcr)

# Write out the dataset
adata.write_h5ad("maynard2020.h5ad", compression="lzf")

# ### Some quick peeks at the data

ir.tl.chain_qc(adata)
ir.pl.group_abundance(adata, groupby="receptor_type")

# Quality control - calculate QC covariates
adata.obs["n_counts"] = adata.layers["raw_counts"].sum(axis=1)
adata.obs["log_counts"] = np.log(adata.obs["n_counts"])
adata.obs["n_genes"] = (adata.layers["raw_counts"] > 0).sum(1)
mt_gene_mask = np.array([gene.lower().startswith("mt-") for gene in adata.var_names])
adata.layers["raw_counts"][:, mt_gene_mask]
adata.obs["mt_counts"] = adata.layers["raw_counts"][:, mt_gene_mask].sum(axis=1)
adata.obs["mt_frac"] = np.divide(adata.obs["mt_counts"], adata.obs["n_counts"])

import seaborn as sns

# Quality control - plot QC metrics
# Sample quality plots
t1 = sc.pl.violin(adata, "n_counts", groupby="sample_name", size=2, log=True, cut=0)
t2 = sc.pl.violin(adata, "mt_frac", groupby="sample_name")

adata_pp = adata[adata.obs["mt_frac"] < 0.2, :].copy()

adata_pp.shape

# don't need this als already TPM
# sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e6)
sc.pp.log1p(adata_pp)

sc.pp.pca(adata_pp, n_comps=50, svd_solver="arpack")

sc.pp.neighbors(adata_pp)

sc.tl.umap(adata_pp)

sc.pl.umap(adata_pp, color=["sample_name", "patient_id"])

sc.pl.umap(adata_pp, color=["CD8A", "CD3E", "PDCD1"])

sc.pl.umap(adata_pp, color=["receptor_type"])

sc.pl.umap(adata_pp, color="receptor_type", groups=["ambiguous", "multichain"])

from IPython.display import display

with pd.option_context("display.max_rows", 30, "display.max_columns", 9999):
    display(adata_pp.obs.loc[adata_pp.obs["receptor_type"] == "ambiguous", :])

ir.pp.ir_neighbors(adata_pp, receptor_arms="all", dual_ir="primary_only")

ir.pp.ir_neighbors(
    adata_pp,
    metric="alignment",
    sequence="aa",
    receptor_arms="all",
    dual_ir="primary_only",
)

ir.tl.define_clonotypes(adata_pp)

ir.tl.clonotype_network(adata_pp, min_size=2)
ir.pl.clonotype_network(adata_pp, color="clonotype", legend_loc="none")

ir.pl.clonotype_network(adata_pp, color=["receptor_subtype"])
