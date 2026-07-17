# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: endofcell,-all
#     formats: py:light,ipynb
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# %load_ext autoreload
# %autoreload 2

# +
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import scanpy as sc
import scipy.sparse as sp
from mudata import MuData

import scirpy as ir

DATASET_DIR = Path("/data/datasets/Maynard_Bivona_2020_NSCLC/")
# -

# The dataset has been downloaded from ENA and then processed using BraCer, TraCeR and the nf-core RNA-seq pipeline
# using `salmon`.
#
# We previously processed this dataset with STAR + featureCounts, but following up the discussion
# on nf-core, featureCounts is not state-of-the art any more for estimating transcript abundances.
# We, therefore, switched to the nf-core RNA-seq pipeline and Salmon.

with open(
    "/data/genomes/hg38/annotation/gencode/gencode.v33.primary_assembly.annotation.gtf",
) as gtf:
    entries = {}
    for line in gtf:
        if line.startswith("#"):
            continue
        attrs = line.split("\t")[8].strip("\n")
        attrs = [item.strip().split(" ") for item in attrs.split(";")]
        attrs = [(x[0], x[1].strip('"')) for x in attrs if len(x) == 2]
        attrs = dict(attrs)
        entries[attrs["gene_id"]] = attrs["gene_name"]


ensg2symbol = pd.DataFrame.from_dict(entries, orient="index", columns=["symbol"])
# remove PAR_ genes and duplicated symbols (~80)
ensg2symbol = ensg2symbol.loc[~ensg2symbol.index.str.contains("PAR_"), :]
ensg2symbol = ensg2symbol.loc[~ensg2symbol.duplicated(), :]
ensg2symbol

sample_paths = glob(str(DATASET_DIR / "rnaseq_pipeline/salmon/SRR*"))

len(sample_paths)


def read_salmon(path):
    """Quant type can be one of "tpm", "count", "count_scaled"."""
    path = Path(path)
    df = pd.read_csv(Path(path / "quant.genes.sf"), sep="\t", index_col=0)
    df = df.join(ensg2symbol, how="inner")
    res = {}
    res["sample_id"] = path.name
    res["var"] = df.reset_index().rename(columns={"index": "ensg"}).loc[:, ["ensg", "symbol"]]
    res["count"] = sp.csc_matrix(df["NumReads"].values)
    res["count_scaled"] = sp.csc_matrix(df["NumReads"].values / df["EffectiveLength"].values)
    res["tpm"] = sp.csc_matrix(df["TPM"].values)

    return res


with Pool(16) as p:
    res = p.map(read_salmon, sample_paths[:], chunksize=20)

# check that gene symbols are the same in all arrays
pdt.assert_frame_equal(res[0]["var"], res[-1]["var"])

sample_ids = np.array([x["sample_id"].split("_")[0] for x in res])
count_mat = sp.vstack([x["count"] for x in res]).tocsr()
tpm_mat = sp.vstack([x["tpm"] for x in res]).tocsr()
count_mat_scaled = sp.vstack([x["count_scaled"] for x in res]).tocsr()

# ## Read and sanitize metadata

# + endofcell="--"
# # +
sample_info = pd.read_csv(DATASET_DIR / "scripts/make_h5ad" / "sra_sample_info.csv", low_memory=False)
cell_metadata = pd.read_csv(
    DATASET_DIR / "scripts/make_h5ad" / "cell_metadata.csv",
    low_memory=False,
    index_col=0,
)

# combine metadata
meta = sample_info.merge(cell_metadata, left_on="cell_ID", right_on="cell_id").set_index("Run")
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
# --

meta.rename(
    columns={
        "sample_name": "sample",
        "histolgy": "condition",
        "patient_id": "patient",
        "biopsy_site": "tissue",
        "primary_or_metastaic": "origin",
    },
    inplace=True,
)

meta["condition"] = [{"Adenocarcinoma": "LUAD", "Squamous": "LSCC"}[x] for x in meta["condition"]]

meta["tissue"] = ["lymph_node" if x == "LN" else x.lower() for x in meta["tissue"]]

meta.loc[meta["origin"].isnull(), ["sample", "patient"]].drop_duplicates()

meta["origin"] = [{"Primary": "tumor_primary", "Metastatic": "tumor_metastasis"}.get(x, np.nan) for x in meta["origin"]]

meta.loc[:, ["sample", "patient", "tissue", "condition", "origin"]]

for col in ["sample", "patient", "tissue", "condition", "origin"]:
    print(col, meta[col].unique())

# ## build adata object

has_tx = set(sample_ids)
has_meta = set(meta.index.values)

has_all = has_tx & has_meta

len(has_tx), len(has_meta), len(has_all)

sample_id_mask = np.isin(sample_ids, list(has_all))

adata = sc.AnnData(
    var=res[0]["var"],
    X=tpm_mat[sample_id_mask, :],
    obs=meta.loc[sample_ids[sample_id_mask], :],
)

adata.layers["counts"] = count_mat[sample_id_mask, :]
adata.layers["counts_length_scaled"] = count_mat_scaled[sample_id_mask, :]

adata.var.set_index("symbol", inplace=True)

# ## add IR information

adata_tcr = ir.io.read_tracer(DATASET_DIR / "smartseq2_pipeline/TraCeR")

adata_bcr = ir.io.read_bracer(DATASET_DIR / "smartseq2_pipeline/BraCeR/filtered_BCR_summary/changeodb.tab")

adata_airr = ir.pp.merge_airr(adata_tcr, adata_bcr)

mdata = MuData({"gex": adata, "airr": adata_airr})

# ## check that all is right

mdata_vis = mdata.copy()

adata.layers

adata_vis = mdata_vis["gex"].copy()
sc.pp.log1p(adata_vis)
sc.pp.highly_variable_genes(adata_vis, n_top_genes=4000, flavor="cell_ranger")
sc.tl.pca(adata_vis)
sc.pp.neighbors(adata_vis)
sc.tl.umap(adata_vis)

sc.pl.umap(adata_vis, color=["sample", "patient", "origin", "CD8A", "CD14"])

ir.tl.chain_qc(mdata_vis)

mdata_vis.update_obs()

_ = ir.pl.group_abundance(mdata_vis, groupby="airr:receptor_subtype", target_col="gex:patient")

_ = ir.pl.group_abundance(mdata_vis, groupby="airr:chain_pairing", target_col="gex:patient")

# ## save MuData

mdata.write_h5mu("maynard2020.h5mu", compression="lzf")
