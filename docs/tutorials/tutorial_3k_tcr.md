---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
---

# Analysis of 3k T cells from cancer

<!-- #raw raw_mimetype="text/restructuredtext" -->
In this tutorial, we re-analyze single-cell TCR/RNA-seq data from Wu et al. (:cite:`Wu2020`)
generated on the 10x Genomics platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer.
For this tutorial, to speed up computations, we use a downsampled version of 3k cells.
<!-- #endraw -->

```python
# This cell is for development only. Don't copy this to your notebook.
%load_ext autoreload
%autoreload 2

# Temporarily suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

```python
import numpy as np
import pandas as pd
import scanpy as sc
import scirpy as ir
from matplotlib import pyplot as plt, cm as mpl_cm
from cycler import cycler

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = 2  # verbosity: errors (0), warnings (1), info (2), hints (3)
```

```python
sc.logging.print_header()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The dataset ships with the `scirpy` package. We can conveniently load it from the :mod:`~scirpy.datasets` module:
<!-- #endraw -->

```python
adata = ir.datasets.wu2020_3k()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. note:: 

    `adata` is a regular :class:`~anndata.AnnData` object with additional, Immune Receptor (IR)-specific columns in `obs` that are named according to the `AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`__. For more information, check the page about 

     * Scirpy's :ref:`data structure <data-structure>`, and
     * Scirpy's :ref:`working model of immune receptors <receptor-model>`.
     
.. warning::

    **scirpy's data structure has updated in v0.7.0 to be fully AIRR-compliant.**
    
    `AnnData` objects created with older versions of scirpy can be upgraded with :func:`scirpy.io.upgrade_schema` to be compatible with the latest version of scirpy. 
<!-- #endraw -->

```python
adata.shape
```

```python
adata.obs
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. note:: **Importing data**

    `scirpy` natively supports reading :term:`IR` data from `Cellranger <https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger>`_ (10x), `TraCeR <https://github.com/Teichlab/tracer>`_ (Smart-seq2)
    or the `AIRR rearrangement schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`__ and provides helper functions to import other data types. We provide a :ref:`dedicated tutorial on data loading <importing-data>` with more details.

    This particular dataset has been imported using :func:`scirpy.io.read_10x_vdj` and merged
    with transcriptomics data using :func:`scirpy.pp.merge_with_ir`. The exact procedure
    is described in :func:`scirpy.datasets.wu2020`.

<!-- #endraw -->

## Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
We recommend following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
and the best practice paper by [Luecken et al.](https://www.embopress.org/doi/10.15252/msb.20188746)

```python
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=100)
```

```python
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1000)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=5000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
```

For the _Wu2020_ dataset, the authors already provide clusters and UMAP coordinates.
Instead of performing clustering and cluster annotation ourselves, we will just use
the provided data. The clustering and annotation methodology is
described in [their paper](https://doi.org/10.1038/s41586-020-2056-8).

```python
adata.obsm["X_umap"] = adata.obsm["X_umap_orig"]
```

```python
mapping = {
    "nan": "other",
    "3.1-MT": "other",
    "4.1-Trm": "CD4_Trm",
    "4.2-RPL32": "CD4_RPL32",
    "4.3-TCF7": "CD4_TCF7",
    "4.4-FOS": "CD4_FOSS",
    "4.5-IL6ST": "CD4_IL6ST",
    "4.6a-Treg": "CD4_Treg",
    "4.6b-Treg": "CD4_Treg",
    "8.1-Teff": "CD8_Teff",
    "8.2-Tem": "CD8_Tem",
    "8.3a-Trm": "CD8_Trm",
    "8.3b-Trm": "CD8_Trm",
    "8.3c-Trm": "CD8_Trm",
    "8.4-Chrom": "other",
    "8.5-Mitosis": "other",
    "8.6-KLRB1": "other",
}
adata.obs["cluster"] = [mapping[x] for x in adata.obs["cluster_orig"]]
```

Let's inspect the UMAP plots. The first three panels show the UMAP plot colored by sample, patient and cluster.
We don't observe any clustering of samples or patients that could hint at batch effects.

The last three panels show the UMAP colored by the T cell markers _CD8_, _CD4_, and _FOXP3_.
We can confirm that the markers correspond to their respective cluster labels.

```python
sc.pl.umap(
    adata,
    color=["sample", "patient", "cluster", "CD8A", "CD4", "FOXP3"],
    ncols=2,
    wspace=0.7,
)
```

## TCR Quality Control

<!-- #raw raw_mimetype="text/restructuredtext" -->
While most of T cell receptors have exactly one pair of α and β chains, up to one third of
T cells can have *dual TCRs*, i.e. two pairs of receptors originating from different alleles (:cite:`Schuldt2019`).

Using the :func:`scirpy.tl.chain_qc` function, we can add a summary
about the Immune cell-receptor compositions to `adata.obs`. We can visualize it using :func:`scirpy.pl.group_abundance`.

.. note:: **chain pairing**

    - *Orphan chain* refers to cells that have either a single alpha or beta receptor chain.
    - *Extra chain* refers to cells that have a full alpha/beta receptor pair, and an additional chain.
    - :term:`Multichain <Multichain-cell>` refers to cells with more than two receptor pairs detected. 
      These cells are likely doublets.


.. note:: **receptor type and receptor subtype**

    - `receptor_type` refers to a coarse classification into `BCR` and `TCR`. Cells with both `BCR` and `TCR` chains
      are labelled as `ambiguous`.
    - `receptor_subtype` refers to a more specific classification into α/β, ɣ/δ, IG-λ, and IG-κ chain configurations.

    For more details, see :func:`scirpy.tl.chain_qc`.
<!-- #endraw -->

```python
ir.tl.chain_qc(adata)
```

As expected, the dataset contains only α/β T-cell receptors:

```python
ax = ir.pl.group_abundance(adata, groupby="receptor_subtype", target_col="source")
```

```python
ax = ir.pl.group_abundance(adata, groupby="chain_pairing", target_col="source")
```

Indeed, in this dataset, ~6% of cells have more than
one pair of productive T-cell receptors:

```python
print(
    "Fraction of cells with more than one pair of TCRs: {:.2f}".format(
        np.sum(
            adata.obs["chain_pairing"].isin(
                ["extra VJ", "extra VDJ", "two full chains"]
            )
        )
        / adata.n_obs
    )
)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
Next, we visualize the :term:`Multichain-cells <Multichain-cell>` on the UMAP plot and exclude them from downstream analysis:
<!-- #endraw -->

```python
sc.pl.umap(adata, color="chain_pairing", groups="multichain")
```

```python
adata = adata[adata.obs["chain_pairing"] != "multichain", :].copy()
```

Similarly, we can use the `chain_pairing` information to exclude all cells that don't have at least 
one full pair of receptor sequences:

```python
adata = adata[~adata.obs["chain_pairing"].isin(["orphan VDJ", "orphan VJ"]), :].copy()
```

Finally, we re-create the chain-pairing plot to ensure that the filtering worked 
as expected: 

```python
ax = ir.pl.group_abundance(adata, groupby="chain_pairing", target_col="source")
```

## Define clonotypes and clonotype clusters

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. warning::

    **The way scirpy computes clonotypes and clonotype clusters has been changed in v0.7.0.**
    
    Code written based on scirpy versions prior to v0.7.0 needs to be updated. Please read the `release notes <https://github.com/icbi-lab/scirpy/releases/tag/v0.7.0>`_ and follow the description in this section. 
    
.. _define-clonotypes-tutorial:

In this section, we will define and visualize :term:`clonotypes <Clonotype>` and :term:`clonotype clusters <Clonotype cluster>`.

*Scirpy* implements a network-based approach for clonotype definition. The steps to create and visualize the clonotype-network are analogous to the construction of a neighborhood graph from transcriptomics data with *scanpy*.

.. list-table:: Analysis steps on IR data
    :widths: 40 60
    :header-rows: 1

    - - scirpy function
      - objective
    - - :func:`scirpy.pp.ir_dist`
      - Compute sequence-based distance matrices for all :term:`VJ <Chain locus>` and
        :term:`VDJ <Chain locus>` sequences. 
    - - :func:`scirpy.tl.define_clonotypes`
      - Define :term:`clonotypes <Clonotype>` by nucleotide
        sequence identity.
    - - :func:`scirpy.tl.define_clonotype_clusters`
      - Cluster cells by the similarity of their CDR3-sequences.
    - - :func:`scirpy.tl.clonotype_network`
      - Compute layout of the clonotype network.
    - - :func:`scirpy.pl.clonotype_network`
      - Plot clonotype network colored by different parameters.

<!-- #endraw -->

### Compute CDR3 neighborhood graph and define clonotypes

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`scirpy.pp.ir_dist` computes distances between :term:`CDR3 <CDR>` nucleotide (`nt`) or amino acid (`aa`) sequences, either based on sequence identity or similarity. It creates two distance matrices: one for all unique :term:`VJ <Chain locus>` sequences and one for all unique :term:`VDJ <Chain locus>` sequences. The distance matrices are added to `adata.uns`. 

The function :func:`scirpy.tl.define_clonotypes` matches cells based on the distances of their 
`VJ` and `VDJ` CDR3-sequences and value of the function parameters `dual_ir` and `receptor_arms`. Finally, it
detects connected modules in the graph and annotates them as clonotypes. This will add a `clone_id` and
`clone_id_size` column to `adata.obs`.

The `dual_ir` parameter defines how scirpy handles cells with :term:`more than one pair of receptors <Dual IR>`. The default value is `any` which implies that cells with any of their primary or secondary receptor chain matching will be considered to be of the same clonotype. 

Here, we define :term:`clonotypes <Clonotype>` based on nt-sequence identity.
In a later step, we will define :term:`clonotype clusters <Clonotype cluster>` based on
amino-acid similarity.
<!-- #endraw -->

```python
# using default parameters, `ir_dist` will compute nucleotide sequence identity
ir.pp.ir_dist(adata)
ir.tl.define_clonotypes(adata, receptor_arms="all", dual_ir="primary_only")
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
To visualize the network we first call :func:`scirpy.tl.clonotype_network` to compute the layout.
We can then visualize it using :func:`scirpy.pl.clonotype_network`. We recommend setting the
`min_cells` parameter to `>=2`, to prevent the singleton clonotypes from cluttering the network.
<!-- #endraw -->

```python
ir.tl.clonotype_network(adata, min_cells=2)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The resulting plot is a network, where each dot represents cells with identical 
receptor configurations. As we define :term:`clonotypes <Clonotype>` as cells with identical CDR3-sequences, each 
dot is also a clonotype. For each clonotype, the numeric clonotype id is shown in the graph. 
The size of each dot refers to the number of cells with the same receptor configurations. 
Categorical variables can be visualized as pie charts. 
<!-- #endraw -->

```python
ir.pl.clonotype_network(
    adata, color="source", base_size=20, label_fontsize=9, panel_size=(7, 7)
)
```

### Re-compute CDR3 neighborhood graph and define clonotype clusters

<!-- #raw raw_mimetype="text/restructuredtext" -->
We can now re-compute the clonotype network based on amino-acid sequence similarity
and define :term:`clonotype clusters <Clonotype cluster>`.

To this end, we need to change set `metric="alignment"` and specify a `cutoff` parameter.
The distance is based on the `BLOSUM62 <https://en.wikipedia.org/wiki/BLOSUM>`__ matrix.
For instance, a distance of `10` is equivalent to 2 `R`s mutating into `N`.
This appoach was initially proposed as *TCRdist* by Dash et al. (:cite:`TCRdist`).

All cells with a distance between their CDR3 sequences lower than `cutoff` will be connected in the network.
<!-- #endraw -->

```python
ir.pp.ir_dist(
    adata,
    metric="alignment",
    sequence="aa",
    cutoff=15,
)
```

```python
ir.tl.define_clonotype_clusters(
    adata, sequence="aa", metric="alignment", receptor_arms="all", dual_ir="any"
)
```

```python
ir.tl.clonotype_network(adata, min_cells=3, sequence="aa", metric="alignment")
```

Compared to the previous plot, we observere several connected dots. 
Each fully connected subnetwork represents a "clonotype cluster", each dot 
still represents cells with identical receptor configurations.

The dots are colored by patient. We observe, that for instance, clonotypes `101` and `68` (left top and bottom) are *private*, i.e. they contain cells from a single patient only. On the other hand, clonotype `159` (left middle) is 
*public*, i.e. it is shared across patients *Lung1* and *Lung3*. 

```python
ir.pl.clonotype_network(
    adata, color="patient", label_fontsize=9, panel_size=(7, 7), base_size=20
)
```

We can now extract information (e.g. CDR3-sequences) from a specific clonotype cluster by subsetting `AnnData`.
When extracting the CDR3 sequences of clonotype cluster `159`, we retreive five different receptor configurations with different numbers of cells, corresponding to the five points in the graph. 

```python
adata.obs.loc[adata.obs["cc_aa_alignment"] == "159", :].groupby(
    [
        "IR_VJ_1_junction_aa",
        "IR_VJ_2_junction_aa",
        "IR_VDJ_1_junction_aa",
        "IR_VDJ_2_junction_aa",
        "receptor_subtype",
    ],
    observed=True,
).size().reset_index(name="n_cells")
```

### Including the V-gene in clonotype definition

<!-- #raw raw_mimetype="text/restructuredtext" -->
Using the paramter `use_v_gene` in :func:`~scirpy.tl.define_clonotypes`, we can enforce
clonotypes (or clonotype clusters) to have the same :term:`V-gene <V(D)J>`, and, therefore, the same :term:`CDR1 and 2 <CDR>`
regions. Let's look for clonotype clusters with different V genes:
<!-- #endraw -->

```python
ir.tl.define_clonotype_clusters(
    adata,
    sequence="aa",
    metric="alignment",
    receptor_arms="all",
    dual_ir="any",
    same_v_gene=True,
    key_added="cc_aa_alignment_same_v",
)
```

```python
# find clonotypes with more than one `clonotype_same_v`
ct_different_v = adata.obs.groupby("cc_aa_alignment").apply(
    lambda x: x["cc_aa_alignment_same_v"].nunique() > 1
)
ct_different_v = ct_different_v[ct_different_v].index.values.tolist()
ct_different_v
```

Here, we see that the clonotype clusters `280` and `765` get split into `(280, 788)` and `(765, 1071)`, respectively, when the `same_v_gene` flag is set. 

```python
adata.obs.loc[
    adata.obs["cc_aa_alignment"].isin(ct_different_v),
    [
        "cc_aa_alignment",
        "cc_aa_alignment_same_v",
        "IR_VJ_1_v_call",
        "IR_VDJ_1_v_call",
    ],
].sort_values("cc_aa_alignment").drop_duplicates().reset_index(drop=True)
```

## Clonotype analysis

### Clonal expansion

<!-- #raw raw_mimetype="text/restructuredtext" -->
Let's visualize the number of expanded clonotypes (i.e. clonotypes consisting
of more than one cell) by cell-type. The first option is to add a column with the :func:`scirpy.tl.clonal_expansion`
to `adata.obs` and overlay it on the UMAP plot.
<!-- #endraw -->

```python
ir.tl.clonal_expansion(adata)
```

`clonal_expansion` refers to expansion categories, i.e singleton clonotypes, clonotypes with 2 cells and more than 2 cells.
The `clonotype_size` refers to the absolute number of cells in a clonotype.

```python
sc.pl.umap(adata, color=["clonal_expansion", "clone_id_size"])
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The second option is to show the number of cells belonging to an expanded clonotype per category
in a stacked bar plot, using the :func:`scirpy.pl.clonal_expansion` plotting function.
<!-- #endraw -->

```python
ir.pl.clonal_expansion(adata, groupby="cluster", clip_at=4, normalize=False)
```

The same plot, normalized to cluster size. Clonal expansion is a sign of positive selection
for a certain, reactive T-cell clone. It, therefore, makes sense that CD8+ effector T-cells
have the largest fraction of expanded clonotypes.

```python
ir.pl.clonal_expansion(adata, "cluster")
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
Expectedly, the CD8+ effector T cells have the largest fraction of expanded clonotypes.

Consistent with this observation, they have the lowest :func:`scirpy.pl.alpha_diversity` of clonotypes.
<!-- #endraw -->

```python
ax = ir.pl.alpha_diversity(
    adata, metric="normalized_shannon_entropy", groupby="cluster"
)
```

### Clonotype abundance

<!-- #raw raw_mimetype="text/restructuredtext" -->
The function :func:`scirpy.pl.group_abundance` allows us to create bar charts for
arbitrary categorial from `obs`. Here, we use it to show the distribution of the
ten largest clonotypes across the cell-type clusters.
<!-- #endraw -->

```python
ir.pl.group_abundance(adata, groupby="clone_id", target_col="cluster", max_cols=10)
```

It might be beneficial to normalize the counts
to the number of cells per sample to mitigate biases due to different sample sizes:

```python
ir.pl.group_abundance(
    adata, groupby="clone_id", target_col="cluster", max_cols=10, normalize="sample"
)
```

Coloring the bars by patient gives us information about public and private clonotypes:
Some clonotypes are *private*, i.e. specific to a certain tissue,
others are *public*, i.e. they are shared across different tissues.

```python
ax = ir.pl.group_abundance(
    adata, groupby="clone_id", target_col="source", max_cols=15, figsize=(5, 3)
)
```

However, clonotypes that are shared between *patients* are rare:

```python
ax = ir.pl.group_abundance(
    adata, groupby="clone_id", target_col="patient", max_cols=15, figsize=(5, 3)
)
```

## Gene usage

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`scirpy.tl.group_abundance` can also give us some information on VDJ usage.
We can choose any of the `{TRA,TRB}_{1,2}_{v,d,j,c}_gene` columns to make a stacked bar plot.
We use `max_col` to limit the plot to the 10 most abundant V-genes.
<!-- #endraw -->

```python
ax = ir.pl.group_abundance(
    adata, groupby="IR_VJ_1_v_call", target_col="cluster", normalize=True, max_cols=10
)
```

We can pre-select groups by filtering `adata`:

```python
ax = ir.pl.group_abundance(
    adata[
        adata.obs["IR_VDJ_1_v_call"].isin(
            ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]
        ),
        :,
    ],
    groupby="cluster",
    target_col="IR_VDJ_1_v_call",
    normalize=True,
)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The exact combinations of VDJ genes can be visualized as a Sankey-plot using :func:`scirpy.pl.vdj_usage`.
<!-- #endraw -->

```python
ax = ir.pl.vdj_usage(adata, full_combination=False, max_segments=None, max_ribbons=30)
```

We can also use this plot to investigate the exact VDJ composition of one (or several) clonotypes:

```python
ir.pl.vdj_usage(
    adata[adata.obs["clone_id"].isin(["68", "101", "127", "161"]), :],
    max_ribbons=None,
    max_segments=100,
)
```

### Spectratype plots

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`~scirpy.pl.spectratype` plots give us information about the length distribution of CDR3 regions.
<!-- #endraw -->

```python
ir.pl.spectratype(adata, color="cluster", viztype="bar", fig_kws={"dpi": 120})
```

The same chart visualized as "ridge"-plot:

```python
ir.pl.spectratype(
    adata,
    color="cluster",
    viztype="curve",
    curve_layout="shifted",
    fig_kws={"dpi": 120},
    kde_kws={"kde_norm": False},
)
```

A spectratype-plot by gene usage. To pre-select specific genes, we can simply filter the `adata` object before plotting.

```python
ir.pl.spectratype(
    adata[
        adata.obs["IR_VDJ_1_v_call"].isin(
            ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]
        ),
        :,
    ],
    cdr3_col="IR_VDJ_1_junction_aa",
    color="IR_VDJ_1_v_call",
    normalize="sample",
    fig_kws={"dpi": 120},
)
```

## Comparing repertoires

### Repertoire simlarity and overlaps

<!-- #raw raw_mimetype="text/restructuredtext" -->
Overlaps in the adaptive immune receptor repertoire of samples or sample groups enables to pinpoint important clonotype groups, as well as to provide a measure of similarity between samples.
Running Scirpy's :func:`~scirpy.tl.repertoire_overlap` tool creates a matrix featuring the abundance of clonotypes in each sample. Additionally, it also computes a (Jaccard) distance matrix of samples as well as the linkage of hierarchical clustering.
<!-- #endraw -->

```python
df, dst, lk = ir.tl.repertoire_overlap(adata, "sample", inplace=False)
df.head()
```

The distance matrix can be shown as a heatmap, while samples are reordered based on hierarchical clustering.

```python
ir.pl.repertoire_overlap(adata, "sample", heatmap_cats=["patient", "source"])
```

A specific pair of samples can be compared on a scatterplot, where dot size corresponds to the number of clonotypes at a given coordinate.

```python
ir.pl.repertoire_overlap(
    adata, "sample", pair_to_plot=["LN2", "LT2"], fig_kws={"dpi": 120}
)
```

## Integrating gene expression data

Leveraging the opportunity offered by close integeration with scanpy, transcriptomics-based data can be utilized alongside immune receptor data.

### Clonotype modularity

<!-- #raw raw_mimetype="text/restructuredtext" -->
Using the :term:`Clonotype modularity` we can identify clonotypes consisting of 
cells that are transcriptionally more similar than expected by random. 

The clonotype modularity score represents the log2 fold change of the 
number of edges in the cell-cell neighborhood graph compared to 
the random background model. Clonotypes (or clonotype clusters) with 
a high modularity score consist of cells that have a similar molecular phenotype. 
<!-- #endraw -->

```python
ir.tl.clonotype_modularity(adata, target_col="cc_aa_alignment")
```

We can plot the clonotype modularity on top of a umap of clonotype network plot

```python
sc.pl.umap(adata, color="clonotype_modularity")
```

```python
ir.pl.clonotype_network(
    adata,
    color="clonotype_modularity",
    label_fontsize=9,
    panel_size=(6, 6),
    base_size=20,
)
```

We can also visualize the clonotype modularity together with the associated
FDR as a sort of "one sided volcano plot":

```python
ir.pl.clonotype_modularity(adata, base_size=20)
```

Let's further inspect the two top scoring candidates. We can extract that information from `adata.obs["clonotype_modularity"]`. 

```python
clonotypes_top_modularity = list(
    adata.obs.set_index("cc_aa_alignment")["clonotype_modularity"]
    .sort_values(ascending=False)
    .index.unique()
    .values[:2]
)
```

```python
sc.pl.umap(
    adata,
    color="cc_aa_alignment",
    groups=clonotypes_top_modularity,
    palette=cycler(color=mpl_cm.Dark2_r.colors),
)
```

We observe that they are (mostly) restricted to a single cluster. By leveraging
scanpy's differential expression module, we can compare the gene expression 
of the cells in the two clonotypes to the rest. 

```python
sc.tl.rank_genes_groups(
    adata,
    "clone_id",
    groups=clonotypes_top_modularity,
    reference="rest",
    method="wilcoxon",
)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
for ct, ax in zip(clonotypes_top_modularity, axs):
    sc.pl.rank_genes_groups_violin(
        adata, groups=[ct], n_genes=15, ax=ax, show=False, strip=False
    )
```

### Clonotype imbalance among cell clusters

Using cell type annotation inferred from gene expression clusters, for example, clonotypes belonging to CD8+ effector T-cells and CD8+ tissue-resident memory T cells, can be compared.

```python
freq, stat = ir.tl.clonotype_imbalance(
    adata,
    replicate_col="sample",
    groupby="cluster",
    case_label="CD8_Teff",
    control_label="CD8_Trm",
    inplace=False,
)
top_differential_clonotypes = stat["clone_id"].tolist()[:3]
```

Showing top clonotypes on a UMAP clearly shows that clonotype 101 is featured by CD8+ tissue-resident memory T cells, while clonotype 68 by CD8+ effector and effector memory cells. 

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.6})
sc.pl.umap(adata, color="cluster", ax=ax1, show=False)
sc.pl.umap(
    adata,
    color="clone_id",
    groups=top_differential_clonotypes,
    ax=ax2,
    # increase size of highlighted dots
    size=[
        80 if c in top_differential_clonotypes else 30 for c in adata.obs["clone_id"]
    ],
    palette=cycler(color=mpl_cm.Dark2_r.colors),
)
```

### Repertoire overlap of cell types

Just like comparing repertoire overlap among samples, Scirpy also offers comparison between gene expression clusters or cell subpopulations. As an example, repertoire overlap of the two cell types compared above is shown.

```python
ir.tl.repertoire_overlap(adata, "cluster")
ir.pl.repertoire_overlap(
    adata, "cluster", pair_to_plot=["CD8_Teff", "CD8_Trm"], fig_kws={"dpi": 120}
)
```

### Marker genes in top clonotypes

Gene expression of cells belonging to individual clonotypes can also be compared using Scanpy. As an example, differential gene expression of two clonotypes, found to be specific to cell type clusters can also be analysed.

```python
sc.tl.rank_genes_groups(
    adata, "clone_id", groups=["101"], reference="68", method="wilcoxon"
)
sc.pl.rank_genes_groups_violin(adata, groups="101", n_genes=15)
```

## Query epitope databases

<!-- #raw raw_mimetype="text/restructuredtext" tags=[] -->
We can use scirpy to query reference databases or datasets to annotate :term:`IRs <IR>` with certain
features, such as epitope specificity. The reference database can be any dataset in :ref:`scirpy's AnnData format <data-structure>` and you can follow the instructions in the :ref:`data loading tutorial <importing-custom-formats>`
to build a custom reference database, if it is not available from :mod:`scirpy.datasets` yet. 

Querying reference datasets uses the same logic as :ref:`defining clonotypes <define-clonotypes-tutorial>`: 

.. list-table:: Analysis steps on IR data
    :widths: 40 60
    :header-rows: 1

    - - scirpy function
      - objective
    - - :func:`scirpy.pp.ir_dist`
      - Compute sequence-based distance matrices .
    - - :func:`scirpy.tl.ir_query`
      - For each cell, identify matching entries in a reference database.
    - - :func:`scirpy.tl.ir_query_annotate`
      - Transfer annotations from reference database to `adata.obs`. 
    - - :func:`scirpy.tl.ir_query_annotate_df`
      - Return a dataframe with all matching annotations. 
      
Here, we obtain the :func:`VDJDB <scirpy.datasets.vdjdb>` and annotate epitopes based
on amino acid sequence identity. For demonstration purposes on this toy dataset we use rather lenient settings: For a match, we specify that it is enough that either of the :term:`VJ <V(D)J>` and :term:`VDJ <V(D)J>` sequences, and either of the primary or secondary receptor chains matches the database. 
<!-- #endraw -->

```python
vdjdb = ir.datasets.vdjdb()
```

```python
ir.pp.ir_dist(adata, vdjdb, metric="identity", sequence="aa")
```

```python
ir.tl.ir_query(
    adata, vdjdb, metric="identity", sequence="aa", receptor_arms="any", dual_ir="any"
)
```

<!-- #raw raw_mimetype="text/restructuredtext" tags=[] -->
:func:`~scirpy.tl.ir_query_annotate_df` allows us to retrieve *all pairs cells with their of matching entries*. If a cell matches multiple entires from the reference database, the resulting data frame will contain multiple rows for the same cell. 
<!-- #endraw -->

```python
ir.tl.ir_query_annotate_df(
    adata,
    vdjdb,
    metric="identity",
    sequence="aa",
    include_ref_cols=["antigen.species", "antigen.gene"],
).tail()
```

<!-- #raw raw_mimetype="text/restructuredtext" tags=[] -->
Alternatively, to break down the annotation to a single-value per cell, you can use
:func:`~scirpy.tl.ir_query_annotate`. Depending on the specified `strategy` it will only label unambiguous matches, or use the most frequent value. 
<!-- #endraw -->

```python
ir.tl.ir_query_annotate(
    adata,
    vdjdb,
    metric="identity",
    sequence="aa",
    include_ref_cols=["antigen.species"],
    strategy="most-frequent",
)
```

```python
sc.pl.umap(adata, color="antigen.species")
```
