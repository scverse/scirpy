---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
---

# Analysis of 3k T cells from cancer

<!-- #raw raw_mimetype="text/restructuredtext" -->
In this tutorial, we re-analize single-cell TCR/RNA-seq data from Wu et al (:cite:`Wu2020`)
generated on the 10x Genomics platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer.
For this tutorial, to speed up computations, we use a downsampled version of 3k cells.
<!-- #endraw -->

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.append("../..")
import scirpy as ir
import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
```

```python
sc.logging.print_versions()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The dataset ships with the `scirpy` package. We can conveniently load it from the `dataset` module:
<!-- #endraw -->

```python
adata = ir.datasets.wu2020_3k()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
`adata` is a regular :class:`~anndata.AnnData` object:
<!-- #endraw -->

```python
adata.shape
```

It just has additional TCR-related columns in `obs`:

 * `has_tcr`: `True` for all cells with a T-cell receptor
 * `TRA_1_<attr>`/`TRA_2_<attr>`: columns related to the primary and secondary TCR-alpha chain
 * `TRB_1_<attr>`/`TRB_2_<attr>`: columns related to the primary and secondary TCR-beta chain

The list of attributes available are:

 * `c_gene`, `v_gene`, `d_gene`, `j_gene`: The gene symbols of the respective genes
 * `cdr3` and `cdr3_nt`: The amino acoid and nucleotide sequences of the CDR3 regions
 * `junction_ins`: The number of nucleotides inserted in the `VD`/`DJ`/`VJ` junctions. 

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. note:: **T cell receptors**
  
  For more information about our T-cell receptor model, see :ref:`tcr-model`. 
<!-- #endraw -->


```python
adata.obs
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. note:: **Importing data**

    `scirpy` supports importing TCR data from `Cellranger <https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger>`_ (10x)
    or `TraCeR <https://github.com/Teichlab/tracer>`_. (SMARTseq2). 
    See :ref:`api-io` for more details.

    This particular dataset has been imported using :func:`scirpy.read_10x_vdj_csv` and merged
    with transcriptomics data using :func:`scirpy.pp.merge_with_tcr`. The exact procedure
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
```

For the _Wu2020_ dataset, the authors already provide clusters and UMAP coordinates.
Instead of performing clustering and cluster annotation ourselves, we will just use
provided data.

```python
adata.obsm["X_umap"] = adata.obsm["X_umap_orig"]
```

```python
mapping = {
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

The lower three panels show the UMAP colored by the T cell markers _CD8_, _CD4_, and _FOXP3_.
We can confirm that the markers correspond to their respective cluster labels.

```python
sc.pl.umap(adata, color=["sample", "patient", "cluster", "CD8A", "CD4", "FOXP3"], ncols=2, wspace=.5)
```

## TCR Quality Control

<!-- #raw raw_mimetype="text/restructuredtext" -->
While most of T cell receptors have exactly one pair of α and β chains, up to one third of 
T cells can have *dual TCRs*, i.e. two pairs of receptors originating from different alleles (:cite:`Schuldt2019`).

Using the :func:`scirpy.tl.chain_pairing` function, we can add a summary
about the T cell receptor compositions to `adata.obs`. We can visualize it using :func:`scirpy.pl.group_abundance`.

.. note:: **chain pairing**

    - *Orphan chain* refers to cells that have either a single alpha or beta receptor chain.
    - *Extra chain* refers to cells that have a full alpha/beta receptor pair, and an additional chain.
    - *Multichain* refers to cells with more than two receptor pairs detected. These cells are likely doublets.
<!-- #endraw -->

```python
ir.tl.chain_pairing(adata)
```

```python
ir.pl.group_abundance(
    adata, groupby="chain_pairing", target_col="source",
)
```

Indeed, in this dataset, ~7% of cells have more than a one pair of productive T-cell receptors:

```python
print("Fraction of cells with more than one pair of TCRs: {:.2f}".format(
    np.sum(adata.obs["chain_pairing"].isin(["Extra beta", "Extra alpha", "Two full chains"])) / adata.n_obs
))
```

Next, we visualize the _Multichain_ cells on the UMAP plot and exclude them from downstream analysis:

```python
sc.pl.umap(adata, color="multi_chain")
```

```python
adata = adata[adata.obs["multi_chain"] != "True", :].copy()
```

## Define clonotypes

<!-- #raw raw_mimetype="text/restructuredtext" -->

In this section, we will define and visualize clonotypes.

*Scirpy* implements a network-based approach for clonotype definition. The steps to create and visualize the clonotype-network are analogous to the construction of a neighborhood graph from transcriptomics data with *scanpy*.

.. list-table:: Analysis steps on transcriptomics data
    :widths: 40 60
    :header-rows: 1

    * - scanpy function
      - objective
    * - :func:`scanpy.pp.neighbors`
      - Compute a nearest-neighbor graph based on gene expression.
    * - :func:`scanpy.tl.leiden`
      - Cluster cells by the similarity of their transcriptional profiles.
    * - :func:`scanpy.tl.umap`
      - Compute positions of cells in UMAP embedding.
    * - :func:`scanpy.pl.umap`
      - Plot UMAP colored by different parameters.

.. list-table:: Analysis steps on TCR data
    :widths: 40 60
    :header-rows: 1

    - - scirpy function
      - objective
    - - :func:`scirpy.pp.tcr_neighbors`
      - Compute a neighborhood graph of CDR3-sequences.
    - - :func:`scirpy.tl.define_clonotypes`
      - Cluster cells by the similarity of their CDR3-sequences.
    - - :func:`scirpy.tl.clonotype_network`
      - Compute positions of cells in clonotype network.
    - - :func:`scirpy.pl.clonotype_network`
      - Plot clonotype network colored by different parameters.

<!-- #endraw -->

### Compute CDR3 neighborhood graph

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`scirpy.pp.tcr_neighbors` computes the pairwise sequence alignment of all CDR3 sequences and
derives a distance from the alignment score. This approach was originally proposed as *TCRdist* by Dash et al. (:cite:`TCRdist`).

The function requires to specify a `cutoff` parameter. All cells with a distance between their
CDR3 sequences lower than `cutoff` will be connected in the network. In the first example,
we set the cutoff to `0`, to define clontypes as cells with **identical** CDR3 sequences.
When the cutoff is `0` no alignment will be performed.

Then, the function :func:`scirpy.tl.define_clonotypes` will detect connected modules
in the graph and annotate them as clonotypes. This will add a `clonotype` and
`clonotype_size` column to `adata.obs`.
<!-- #endraw -->

```python
ir.pp.tcr_neighbors(adata, strategy="all", merge_chains="primary_only", cutoff=0)
ir.tl.define_clonotypes(adata)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->

To visualize the network we first call :func:`scirpy.tl.clonotype_network` to compute the layout.
We can then visualize it using :func:`scirpy.pl.clonotype_network`. We recommend setting the
`min_size` parameter to `>=2`, to prevent the singleton clonotypes from cluttering the network.

<!-- #endraw -->

```python
ir.tl.clonotype_network(adata, min_size=2)
ir.pl.clonotype_network(adata, color="clonotype", legend_loc="none")
```

Let's re-compute the network with a `cutoff` of `15`.
That's the equivalent of 3 `R`s mutating into `N` (using the BLOSUM62 distance matrix).

Additionally, we set `chains` to `all`. This results in the distances not being only
computed between the most abundant pair of T-cell receptors, but instead, will
take the minimal distance between any pair of T-cell receptors.

```python
sc.settings.verbosity = 4
```

```python
ir.pp.tcr_neighbors(adata, cutoff=15, merge_chains="all")
ir.tl.define_clonotypes(adata, partitions="connected")
```

```python
ir.tl.clonotype_network(adata, min_size=3)
```

Compared to the previous plot, we observe slightly larger clusters that are not necessarily fully connected any more. 

```python
ir.pl.clonotype_network(adata, color="clonotype", legend_fontoutline=3)
```

Now we show the same graph, colored by sample.
We observe that for instance clonotypes 247 and 293 are _private_, i.e. they contain cells from
a single sample only. On the other hand, for instance clonotype 106 is _public_, i.e.
it is shared across tissues and/or patients.

```python
ir.pl.clonotype_network(adata, color="sample")
```

## Clonotype analysis

### Clonal expansion

Let's visualize the number of expanded clonotypes (i.e. clonotypes consisting
of more than one cell) by cell-type. The first option is to add a column with the *clonal expansion*
to `adata.obs` and plot it on the UMAP plot. 

```python
ir.tl.clonal_expansion(adata)
```

```python
sc.pl.umap(adata, color=["clonal_expansion", "clonotype_size"])
```

The second option is to show the number of cells belonging to an expanded clonotype per category
in a stacked bar plot: 

```python
ir.pl.clonal_expansion(adata, groupby="cluster", clip_at=4, fraction=False)
```

The same plot, normalized to cluster size: 

```python
ir.pl.clonal_expansion(adata, "cluster")
```

Expectedly, the CD8+ effector T cells have the largest fraction of expanded clonotypes. 

Consistent with this observation, they have the lowest alpha diversity of clonotypes: 

```python
ax = ir.pl.alpha_diversity(adata, groupby="cluster")
```

### Clonotype abundance

<!-- #raw raw_mimetype="text/restructuredtext" -->
The function :func:`scirpy.pl.group_abundance` allows us to create bar charts for
arbitrary categorial from `obs`. Here, we use it to show the distribution of the 
ten largest clonotypes across the cell-type clusters.
<!-- #endraw -->

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="cluster", max_cols=10
)
```

When cell-types are considered, it might be benefitial to normalize the counts
to the sample size: 

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="cluster", max_cols=10, fraction="sample"
)
```

Coloring the bars by patient gives us information about public and private clonotypes: 
While most clonotypes are private, i.e. specific to a certain tissue, 
some of them are public, i.e. they are shared across different tissues. 

```python
ax = ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="sample", max_cols=10
)
ax.legend(loc=(1.1, 0.01), ncol=4, fontsize="x-small") 
```

However, none of them is shared across patients.  
This is consistent with the observation we made earlier on the clonotype network. 

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="patient", max_cols=10
)
```

## Gene usage

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`scirpy.tl.group_abundance` can also give us some information on VDJ usage. 
We can choose any of the `{TRA,TRB}_{1,2]_{v,d,j,c}_gene` columns to make a stacked bar plot. 
We use `max_col` to limit the plot to the 10 most abundant V-genes. 
<!-- #endraw -->

```python
ir.pl.group_abundance(
    adata,
    groupby="TRB_1_v_gene",
    target_col="cluster",
    fraction=True,
    max_cols=10
)
```

We can pre-select groups by filtering `adata`:

```python
ir.pl.group_abundance(
    adata[adata.obs["TRB_1_v_gene"].isin(
        ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]
    ),:],
    groupby="cluster",
    target_col="TRB_1_v_gene",
    fraction=True,
)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The exact combinations of VDJ genes can be visualized as a Sankey-plot using :func:`scirpy.pl.vdj_usage`. 
<!-- #endraw -->

```python
ir.pl.vdj_usage(adata, full_combination=False, top_n=30)
```

### Spectratype plots

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`~scirpy.pl.spectratype` plots give us information about the length distribution of CDR3 regions. 
<!-- #endraw -->

```python
ir.pl.spectratype(adata, target_col="cluster", fig_kws={"dpi": 120})
```

The same as line chart, normalized to cluster size: 

```python
ir.pl.spectratype(adata, target_col="cluster", fraction="cluster", viztype="line")
```

Again, to pre-select specific genes, we can simply filter the `adata` object before plotting. 

```python
ir.pl.spectratype(
    adata[adata.obs["TRB_1_v_gene"].isin(["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]),:], 
    groupby="TRB_1_cdr3",
    target_col="TRB_1_v_gene",
    fraction="sample",
    fig_kws={'dpi': 150}
)
```

```python

```
