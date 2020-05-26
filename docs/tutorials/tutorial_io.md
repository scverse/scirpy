---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0.rc1
  kernelspec:
    display_name: Python [conda env:sctcrpy2]
    language: python
    name: conda-env-sctcrpy2-py
---

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.insert(0, "../..")
import scirpy as ir
import scanpy as sc
from glob import glob
import pandas as pd
import tarfile
import anndata

# suppress "storing XXX as categorical" warnings. 
anndata.logging.anndata_logger.setLevel("ERROR")
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. _importing-data:

Loading TCR data with scirpy
============================

In this notebook, we demonstrate how single-cell TCR data can be imported into 
an :class:`anndata.AnnData` object for the use with Scirpy. To learn more about
AnnData and how Scirpy makes use of it, check out the :ref:`data-structure` section. 


Loading data from *10x Genomics CellRanger* or *TraCeR*
-------------------------------------------------------

We provide convenience functions to load data from *CellRanger* or *TraCeR* with a single function call
supporting both data generated on the *10x* and *Smart-seq2* sequencing platforms, respectively. 

.. module:: scirpy.io

.. autosummary::
   :toctree: .

   read_10x_vdj
   read_tracer
   
Read 10x data
^^^^^^^^^^^^^

With `read_10x_vjd` we can load `filtered_contig_annotations.csv` or `contig_annotations.json` files as they are produced by *CellRanger*. 
Here, we demonstrate how to load paired single cell transcriptomics and TCR sequencing data from COVID19 patients 
from `GSE145926 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926>`_ (:cite:`Liao2020`). 
<!-- #endraw -->

```python
# Load the TCR data
adata_tcr = ir.io.read_10x_vdj(
    "example_data/liao-2019-covid19/GSM4385993_C144_filtered_contig_annotations.csv.gz"
)

# Load the associated transcriptomics data
adata = sc.read_10x_h5(
    "example_data/liao-2019-covid19/GSM4339772_C144_filtered_feature_bc_matrix.h5"
)
```

This particular sample only has a detected TCR for a small fraction of the cells: 

```python
adata_tcr.shape
```

```python
adata.shape
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
Next, we integrate both the TCR and the transcriptomics data into a single :class:`~anndata.AnnData` object
using :func:`scirpy.pp.merge_with_tcr`:
<!-- #endraw -->

```python
ir.pp.merge_with_tcr(adata, adata_tcr)
```

Now, we can use TCR-related variables together with the gene expression data. 
Here, we visualize the cells with a detected TCR on the UMAP plot. 
It is reassuring that the TCRs coincide with the T-cell marker gene CD3. 

```python
sc.pp.log1p(adata)
sc.pp.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["has_tcr", "CD3E"])
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
Read Smart-seq2 data processed with TraCeR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`TraCeR <https://github.com/Teichlab/tracer>`__ (:cite:`Stubbington2016-kh`) is a method commonly used
to extract TCR sequences from data generated with Smart-seq2 or other full-length single-cell sequencing protocols. 
`nf-core <https://nf-co.re/>`_ provides a full `pipeline for processing Smart-seq2 sequencing data <https://github.com/nf-core/smartseq2/>`__

The :func:`scirpy.io.read_tracer` function obtains its TCR information from the `.pkl` file
in the `filtered_TCR_seqs` folder TraCeR generates for each cell. 

For this example, we load data from `GSE75688 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688>`_ (:cite:`Chung2017`). 
The raw data has been processed using the `Smart-seq2 pipeline <https://github.com/nf-core/smartseq2/>`__ from nf core. 
<!-- #endraw -->

```python
# extract data
with tarfile.open("example_data/chung-park-2017.tar.bz2", 'r:bz2') as tar:
    tar.extractall("example_data/chung-park-2017")
```

```python
# Load transcriptomics data from count matrix
tmp_expr = pd.read_csv("example_data/chung-park-2017/counts.tsv", sep="\t")
# anndata needs genes in columns and samples in rows
tmp_expr = tmp_expr.set_index("Geneid").T
adata = sc.AnnData(tmp_expr)
adata.shape
```

```python
# Load TCR data and merge it with transcriptomics data
adata_tcr = ir.io.read_tracer("example_data/chung-park-2017/tracer/")
ir.pp.merge_with_tcr(adata, adata_tcr)
```

```python
sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=3000)
sc.pp.log1p(adata)
sc.pp.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["has_tcr", "CD3E"])
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
Combining multiple samples
--------------------------

It is quite common that the sequncing data is split up in multiple samples. 
To combine them into a single object, we load each sample independently using one of the approaches described
in this document. Then, we combine them using :meth:`anndata.AnnData.concatenate`. 

Here is a full example loading and combining three samples from the COVID19 study by :cite:`Liao2020`. 
<!-- #endraw -->

```python
# define sample metadata. Usually read from a file. 
samples = {
    "C144": {"group": "mild"},
    "C146": {"group": "severe"},
    "C149": {"group": "healthy control"},
}
```

```python
# Create a list of AnnData objects (one for each sample)
adatas = []
for sample, sample_meta in samples.items():
    gex_file = glob(f"example_data/liao-2019-covid19/*{sample}*.h5")[0]
    tcr_file = glob(f"example_data/liao-2019-covid19/*{sample}*.csv.gz")[0]
    adata = sc.read_10x_h5(gex_file)
    adata_tcr = ir.io.read_10x_vdj(tcr_file)
    ir.pp.merge_with_tcr(adata, adata_tcr)
    adata.obs['sample'] = sample
    adata.obs['group'] = sample_meta['group']
    # concatenation only works with unique gene names
    adata.var_names_make_unique()
    adatas.append(adata)
```

```python
# Merge anndata objects
adata = adatas[0].concatenate(adatas[1:])
```

The data is now integrated in a single object. 
Again, the detected TCRs coincide with `CD3E` gene expression. 
We clearly observe batch effects between the samples -- for a meaningful downstream analysis further 
processing steps such as highly-variable gene filtering and batch correction are necessary. 

```python
sc.pp.log1p(adata)
sc.pp.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["has_tcr", "CD3E", "sample"])
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
.. _importing-custom-formats:

Creating AnnData objects from other formats
-------------------------------------------

If you believe you are working with a commonly used format, consider sending a `feature request <https://github.com/icbi-lab/scirpy/issues>`_
for a `read_XXX` function. 
<!-- #endraw -->
