---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
---

# Testing the repertoire overlap function

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
adata = adata[adata.obs["multi_chain"] != "True", :].copy()
```

```python
ir.pp.tcr_neighbors(adata, strategy="all", merge_chains="primary_only", cutoff=0)
ir.tl.define_clonotypes(adata)
```

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
ir.pp.tcr_neighbors(adata, cutoff=15, merge_chains="all")
ir.tl.define_clonotypes(adata, partitions="connected")
```

```python

```

```python
adata.obs.head()
```

```python
df, dst, lk = ir.tl.repertoire_overlap(adata, 'sample', inplace=False)
```

```python
df.head()
```

```python
ir.pl.repertoire_overlap(adata, 'sample')
```

```python
ir.pl.repertoire_overlap(adata, 'sample', heatmap_cats=['patient', 'source'])
```

```python
ir.pl.repertoire_overlap(adata, 'sample', dendro_only=True, heatmap_cats=['source'])
```

```python
ir.pl.repertoire_overlap(adata, 'sample', pair_to_plot=('LN2', 'LT2'))
```

```python

```
