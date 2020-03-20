---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.2"
      jupytext_version: 1.4.1
---

# Analysis of 3k T cells from cancer

In this tutorial, we re-analize single-cell TCR/RNA-seq data from <cite data-cite="Wu2020">Wu2020</cite> generated on the 10x VDJ platform. The original dataset consists of >140k T cells from 14 treatment-naive patients across four different types of cancer.

For this tutorial, to speed up computations, we use a downsampled version of 3k cells.

<div class="alert alert-warning">

**Warning:** This tutorial is under construction!

</div>

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.append("../..")
import sctcrpy as st
import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
```

The Dataset ships with the `sctcrpy` package:

```python
adata = st.datasets.wu2020_3k()
```

`adata` is a regular `scanpy` AnnData object:

```python
adata.shape
```

It just has additional TCR-related columns in `obs`:

```python
adata.obs
```

```python

```

## Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
Here, we quickly preprocess transcriptomics data, roughly following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html).

```python
sc.pp.filter_cells(adata, min_genes=700)
sc.pp.filter_cells(adata, min_counts=2000)
sc.pp.filter_genes(adata, min_cells=10)
```

```python
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1000)
sc.pp.log1p(adata)
```

```python
sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=3000)
```

```python
sc.tl.pca(adata, svd_solver='arpack')
```

```python
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=.5)
```

```python
sc.pl.umap(adata, color=["sample", "patient", "leiden", "CD3E", "CD8A", "CD4"], ncols=3)
```

## TCR QC

It is thought that a single cell can have up to two alpha and two beta chains -- one from each Allele.
Let's check how many TCR chains have been detected

```python
st.tl.chain_pairing(adata)
```

```python
st.pl.group_abundance(
    adata, groupby="chain_pairing", target_col="sample", fraction="has_tcr",
)
```

The `multi_chain` key stores which cells have more than 2 TRA or 2 TRB chains. They are likely doublets and we will exclude them from the downstream analysis

```python
sc.pl.umap(adata, color="multi_chain")
```

```python
adata = adata[adata.obs["multi_chain"] != "True", :].copy()
```

## Define clonotypes

Next, we need to define clonotypes

<div class="alert alert-warning">

**Warning:** Clonotype definition will soon change to a graph-based approach

</div>

```python
st.tl.define_clonotypes(adata)
```

```python
st.pl.clonal_expansion(adata, groupby="leiden", clip_at=4, fraction=False)
```

```python
st.pl.clonal_expansion(adata, "leiden")
```

```python
st.pl.alpha_diversity(adata, groupby="leiden")
```

### Clonotype abundance

```python
st.pl.group_abundance(
    adata, groupby="clonotype", target_col="leiden", max_cols=10, fraction=False
)
```

Perhaps an even more straightforward question would be comparing clonotype composition of samples

```python
st.pl.group_abundance(
    adata, groupby="clonotype", target_col="sample", max_cols=10, stacked=False
)
```

If cell types are considered, it is still probably better to normalize to cell numbers in a sample.

```python
st.pl.group_abundance(
    adata, groupby="clonotype", target_col="leiden", fraction="sample", max_cols=10
)
```

## Gene usage

Group abundance plots can also give some information on VDJ usage

```python
st.pl.group_abundance(
    adata,
    groupby="TRB_1_v_gene",
    target_col="leiden",
    fraction="sample",
    max_cols=10,
    fig_kws={"dpi": 170},
)
```

```python
vdj_usage = st.tl.group_abundance(
    adata, groupby="leiden", target_col="TRB_1_v_gene", fraction=True
)
```

```python
vdj_usage = vdj_usage.loc[:, ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]]
```

```python
st.pl.base.bar(vdj_usage)
```

### Spectratype plots

```python
st.pl.spectratype(adata, target_col="leiden",     fig_kws={"dpi": 170},)
```

```python
st.pl.spectratype(adata, target_col="leiden", fraction="sample", viztype="line")
```

```python
st.pl.spectratype(adata, target_col="leiden", fraction=False, viztype="line")
```

```python
st.pl.spectratype(
    adata, groupby="TRB_1_cdr3_len", target_col="TRB_1_v_gene", fraction="sample", fig_kws={'dpi': 150}
)
```
