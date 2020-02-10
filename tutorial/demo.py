# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python [conda env:.conda-scTCR]
#     language: python
#     name: conda-env-.conda-scTCR-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import sys

sys.path.append("..")
import sctcrpy as st
import pandas as pd
import numpy as np
import scanpy as sc

# %% [markdown]
# ## Read in the TCR data

# %%
adata_vdj = st.read_10x_vdj("../tutorial/example_data/10x/all_contig_annotations.json")

# %%
adata_tracer = st.read_tracer("../tutorial/example_data/tracer/tracer_100/")

# %% [markdown]
# The sample is a `AnnData` object with all TCR information stored in `.obs` and an empty gene expression matrix `X`.

# %%
adata_tracer

# %%
adata_tracer.obs

# %% [markdown]
# ## Combine with Transcriptomics data
#
# Let's now read in the corresponding transcriptomics data (for the 10x sample) with scanpy and combine it with TCR data.

# %%
adata = sc.read_10x_h5(
    "../tutorial/example_data/10x/vdbg_141_gex_filtered_feature_bc_matrix.h5"
)

# %%
adata

# %%
st.pp.merge_with_tcr(adata, adata_vdj)

# %%
adata

# %%
adata.obs


# %% [markdown]
# Just to make some plottings clearer, let us simulate we have more than one samples

# %%
def f(x):
    return np.random.randint(6)
adata.obs['sample'] = adata.obs.apply(f, axis=1)
adata.obs.head()

# %% [markdown]
# ## Standard-preprocessing for the transcriptomics data

# %%
sc.pp.filter_cells(adata, min_genes=700)
sc.pp.filter_cells(adata, min_counts=2000)

# %%
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1000)
sc.pp.log1p(adata)

# %% [markdown]
# ### Compute PCA + UMAP

# %%
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %% [markdown]
# ### Leiden clustering

# %%
sc.tl.leiden(adata, resolution=0.5)

# %% [markdown]
# ## Visualize Genes, clusters and TCRs

# %%
sc.pl.umap(adata, color=["CD3E", "CD8A", "leiden", "has_tcr"], ncols=2)

# %% [markdown]
# ## Filter cells with TCR only and re-compute UMAP

# %%
adata = adata[adata.obs["has_tcr"] == "True", :]

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.tl.leiden(adata, resolution=0.5)

# %%
sc.pl.umap(
    adata,
    color=[
        "CD3E",
        "CD8A",
        "FOXP3",
        "CD4",
        "leiden",
        "multi_chain",
        "TRA_1_cdr3_len",
        "TRA_1_junction_ins",
    ],
    ncols=3,
)

# %%
sc.pl.violin(adata, ["TRA_1_cdr3_len", "TRA_1_junction_ins"], groupby="leiden")

# %% [markdown]
# ## Apply TCR-basesd tools

# %%
st.tl.define_clonotypes(adata)

# %%
st.tl.alpha_diversity(adata, groupby="leiden")

# %%
st.pl.alpha_diversity(adata, groupby="leiden")

# %%
st.pl.clonal_expansion(adata, groupby="leiden", clip_at=4, fraction=False)

# %%
st.pl.clonal_expansion(adata, groupby="leiden")

# %% [markdown]
# ## Clonotype abundances

# %%
ax = st.pl.group_abundance(adata, groupby="leiden")
#ax = ax[0]
#fig = ax.get_figure()
#fig.savefig('/data/scratch/szabo/RepertoireSeq/singlecell_tcr/tutorial/abundance.png')

# %% [markdown]
# Perhaps an even more straightforward question would be comparing clonotype composition of samples

# %%
st.pl.group_abundance(adata, groupby="sample", fraction=False)

# %%
st.pl.group_abundance(adata, groupby="sample")

# %%
st.pl.group_abundance(adata, groupby="sample", viztype='stacked')

# %%
st.pl.group_abundance(adata, groupby="sample", viztype='table')

# %% [markdown]
# If cell types are considered, it is still probably better to normalize to cell numbers in a sample.

# %%
st.pl.group_abundance(adata, groupby="leiden", fraction='sample')

# %%
I would simply use group abundance plots to show shain usage (it would of course not check if chain usage tool has been run then)

# %%
st.tl.chain_pairing(adata)

# %%
adata.obs.loc[:,[
            "has_tcr",
            "multichain",
            "TRA_1_cdr3",
            "TRA_2_cdr3",
            "TRB_1_cdr3",
            "TRB_2_cdr3",
        ]]

# %%
st.pl.group_abundance(adata, groupby='sample', target_col='chain_pairing', viztype='stacked')

# %%
st.pl.group_abundance(adata, groupby='chain_pairing', target_col='sample', viztype='stacked', fraction='sample')

# %%
Group abundance plots can also give some information on VDJ usage

# %%
st.pl.group_abundance(adata, groupby='leiden', target_col='TRB_1_v_gene', fraction='sample', viztype='stacked')

# %%
tb = st.pl.group_abundance(adata, groupby='TRB_1_v_gene', target_col='leiden', fraction='sample', viztype='table')
tb.head()

# %%
st.pl.group_abundance(adata, groupby='TRB_1_v_gene', target_col='leiden', fraction='sample', viztype='stacked', group_order=tb.columns.tolist()[1:10])

# %% [markdown]
# ## Spectratype plots

# %%
st.pl.spectratype(adata, groupby='leiden')

# %%
st.pl.spectratype(adata, groupby='leiden', fraction='sample')

# %%
st.pl.spectratype(adata, groupby='leiden', fraction='sample', viztype='line')

# %%

# %%

# %%
z = adata.uns['sctcrpy'].pop('group_abundance')

# %%
adata.obs.to_csv('/data/scratch/szabo/RepertoireSeq/singlecell_tcr/tutorial/toytable.csv')

# %%
