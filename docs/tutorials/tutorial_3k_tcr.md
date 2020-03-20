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

## Read in the TCR data

```python
print([x for x in range(10)])
```

```python

```
