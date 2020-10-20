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
