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
# To plot clonotype abundances, I experimented with two approaches. One uses Pandas and precomputes before passing data to the plotting function and a more lazy one that relies on Seaborn.

# %%
st.pl.group_abundance(adata, groupby="leiden", fraction=False)

# %%
st.pl.group_abundance(adata, groupby="leiden")

# %%
st.pl.group_abundance(adata, target_col='TRB_1_v_gene', label_col='TRB_1_v_gene', groupby="leiden")

# %%
st.pl.group_abundance_lazy(adata, groupby="leiden")

# %%
st.tl.group_abundance(adata, groupby='leiden')

# %%
adata.obs.loc[~_is_na(adata.obs[groupby]), [groupby, target_col]]

# %%
tcr_obs.groupby([target_col]).size().reset_index(name="count").sort_values(by="count", ascending=False)

# %%
overall_abundaces = tcr_obs.groupby([target_col]).size().reset_index(name="count").sort_values(by="count", ascending=False)

# %%
z = tcr_obs.groupby([target_col]).size().reset_index(name="count").sort_values(by="count", ascending=False)
z['size'] = 

# %%
groupsizes = tcr_obs.loc[:, groupby].value_counts().to_dict()
clonotype_counts = (tcr_obs.groupby([groupby, target_col]).size().reset_index(name="count"))
clonotype_counts['groupsize'] = clonotype_counts[groupby].map(groupsizes)
clonotype_counts.groupsize = clonotype_counts.groupsize.astype('int32')
#clonotype_counts['count'] = clonotype_counts['count']/clonotype_counts['groupsize']
clonotype_counts

# %%
clonotype_counts.groupby([target_col]).sum().loc['clonotype_919', :]

# %%
clonotype_counts.groupby([target_col]).sum().sort_values(by="count", ascending=False).index.values

# %%
piw = clonotype_counts.pivot(index='clonotype', columns='leiden', values='count').fillna(0.0)
piw


# %%
sns.countplot(y='clonotype', data=piw)

# %%
piw.plot.bar()

# %%
adata.uns['sctcrpy'].keys()

# %%
tcr_obs.loc[:,groupby].value_counts().to_dict()

# %%
z = adata.uns['sctcrpy']['group_abundance_lazy']

# %%
z = adata.uns['sctcrpy'].pop('group_abundance')

# %%
obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
        ],
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
zadata = sc.AnnData(obs=obs)

# %%
st.tl.group_abundance_lazy(zadata, groupby="group", inplace=False)['df'].to_dict()

# %%
st.pl.group_abundance()
