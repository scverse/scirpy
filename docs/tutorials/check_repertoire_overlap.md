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
df = ir.tl.repertoire_overlap(adata, 'sample')
```

```python
df.head()
```

```python
adata.uns['repertoire_overlap'].keys()
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
from .fixtures import adata_clonotype, adata_tra
```

```python
obs = {
    "AAGGTTCCACCCAGTG-1": {
        "TRA_1_cdr3_len": 15.0,
        "TRA_1_cdr3": "CALSDPNTNAGKSTF",
        "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
        "sample": 3,
        "clonotype": "clonotype_458",
        "chain_pairing": "Extra alpha",
    },
    "ACTATCTAGGGCTTCC-1": {
        "TRA_1_cdr3_len": 14.0,
        "TRA_1_cdr3": "CAVDGGTSYGKLTF",
        "TRA_1_cdr3_nt": "TGTGCCGTGGACGGTGGTACTAGCTATGGAAAGCTGACATTT",
        "sample": 1,
        "clonotype": "clonotype_739",
        "chain_pairing": "Extra alpha",
    },
    "CAGTAACAGGCATGTG-1": {
        "TRA_1_cdr3_len": 12.0,
        "TRA_1_cdr3": "CAVRDSNYQLIW",
        "TRA_1_cdr3_nt": "TGTGCTGTGAGAGATAGCAACTATCAGTTAATCTGG",
        "sample": 1,
        "clonotype": "clonotype_986",
        "chain_pairing": "Two full chains",
    },
    "CCTTACGGTCATCCCT-1": {
        "TRA_1_cdr3_len": 12.0,
        "TRA_1_cdr3": "CAVRDSNYQLIW",
        "TRA_1_cdr3_nt": "TGTGCTGTGAGGGATAGCAACTATCAGTTAATCTGG",
        "sample": 1,
        "clonotype": "clonotype_987",
        "chain_pairing": "Single pair",
    },
    "CGTCCATTCATAACCG-1": {
        "TRA_1_cdr3_len": 17.0,
        "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
        "TRA_1_cdr3_nt": "TGTGCAGCAAGTCGCAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
        "sample": 5,
        "clonotype": "clonotype_158",
        "chain_pairing": "Single pair",
    },
    "CTTAGGAAGGGCATGT-1": {
        "TRA_1_cdr3_len": 15.0,
        "TRA_1_cdr3": "CALSDPNTNAGKSTF",
        "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
        "sample": 1,
        "clonotype": "clonotype_459",
        "chain_pairing": "Single pair",
    },
    "GCAAACTGTTGATTGC-1": {
        "TRA_1_cdr3_len": 14.0,
        "TRA_1_cdr3": "CAVDGGTSYGKLTF",
        "TRA_1_cdr3_nt": "TGTGCCGTGGATGGTGGTACTAGCTATGGAAAGCTGACATTT",
        "sample": 1,
        "clonotype": "clonotype_738",
        "chain_pairing": "Single pair",
    },
    "GCTCCTACAAATTGCC-1": {
        "TRA_1_cdr3_len": 15.0,
        "TRA_1_cdr3": "CALSDPNTNAGKSTF",
        "TRA_1_cdr3_nt": "TGTGCTCTGAGTGATCCCAACACCAATGCAGGCAAATCAACCTTT",
        "sample": 3,
        "clonotype": "clonotype_460",
        "chain_pairing": "Two full chains",
    },
    "GGAATAATCCGATATG-1": {
        "TRA_1_cdr3_len": 17.0,
        "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
        "TRA_1_cdr3_nt": "TGTGCAGCAAGTAGGAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
        "sample": 5,
        "clonotype": "clonotype_157",
        "chain_pairing": "Single pair",
    },
    "AAACCTGAGATAGCAT-1": {
        "TRA_1_cdr3_len": 13.0,
        "TRA_1_cdr3": "CAGGGSGTYKYIF",
        "TRA_1_cdr3_nt": "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
        "sample": 3,
        "clonotype": "clonotype_330",
        "chain_pairing": "Single pair",
    },
    "AAACCTGAGTACGCCC-1": {
        "TRA_1_cdr3_len": 14.0,
        "TRA_1_cdr3": "CAMRVGGSQGNLIF",
        "TRA_1_cdr3_nt": "TGTGCAATGAGGGTCGGAGGAAGCCAAGGAAATCTCATCTTT",
        "sample": 5,
        "clonotype": "clonotype_592",
        "chain_pairing": "Two full chains",
    },
    "AAACCTGCATAGAAAC-1": {
        "TRA_1_cdr3_len": 15.0,
        "TRA_1_cdr3": "CAFMKPFTAGNQFYF",
        "TRA_1_cdr3_nt": "TGTGCTTTCATGAAGCCTTTTACCGCCGGTAACCAGTTCTATTTT",
        "sample": 5,
        "clonotype": "clonotype_284",
        "chain_pairing": "Extra alpha",
    },
    "AAACCTGGTCCGTTAA-1": {
        "TRA_1_cdr3_len": 12.0,
        "TRA_1_cdr3": "CALNTGGFKTIF",
        "TRA_1_cdr3_nt": "TGTGCTCTCAATACTGGAGGCTTCAAAACTATCTTT",
        "sample": 3,
        "clonotype": "clonotype_425",
        "chain_pairing": "Extra alpha",
    },
    "AAACCTGGTTTGTGTG-1": {
        "TRA_1_cdr3_len": 13.0,
        "TRA_1_cdr3": "CALRGGRDDKIIF",
        "TRA_1_cdr3_nt": "TGTGCTCTGAGAGGGGGTAGAGATGACAAGATCATCTTT",
        "sample": 3,
        "clonotype": "clonotype_430",
        "chain_pairing": "Single pair",
    },
}
obs = pd.DataFrame.from_dict(obs, orient="index")
zadata = sc.AnnData(obs=obs)
```

```python
zadata
```

```python
ir.tl.repertoire_overlap(zadata, 'sample').to_dict(orient="index")
```

```python

```
