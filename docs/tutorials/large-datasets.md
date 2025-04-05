# Working with >1M cells

Scirpy scales to millions of cells on a single workstation. This page is a work-in-progess collection with advice how to
work with large datasets.

:::{admonition} Use an up-to-date version!
:class: tip

Scalability has been a major focus of recent developments in Scirpy. Make sure you use the latest version
when working with large datasets to take advantage of all speedups.
:::

## Distance metrics

Computing pairwise sequence distances the major bottleneck for large datasets in the scirpy workflow.
Here is some advice on how to maximize the speed of this step:

## Choose an appropriate distance metric for `pp.ir_dist`

Some distance metrics are significantly faster than others. Here are the distance metrics, roughly ordered by speed:

`identity` > `gpu_hamming` > `hamming` = `normalized_hamming` > `tcrdist` > `levenshtein` > `fastalignment` > `alignment`

TCRdist, fastalignment and alignment produce very similar distance matrices, but tcrdist is by far the fastest. For this
reason, we'd always recommend to go with `tcrdist`, when looking for a metric taking into account a substitution matrix.

## Multi-machine parallelization with dask

The `hamming`, `normalized_hamming`, `tcrdist`, `levenshtein`, `fastalignment`, and `alignment` metrics are parallelized
using [joblib](https://joblib.readthedocs.io/en/stable/). This makes it very easy to switch the backend to
[dask](https://www.dask.org/) to distribute jobs across a multi machine cluster. Note that this comes with a
considerable overhead for communication between the workers. It's only worthwhile when processing on a single
machine becomes infeasible.

```python
from dask.distributed import Client, LocalCluster
import joblib

# substitute this with a multi-machine cluster...
cluster = LocalCluster(n_workers=16)
client = Client(cluster)

with joblib.parallel_config(backend="dask", n_jobs=200, verbose=10):
    ir.pp.ir_dist(
        mdata,
        metric="tcrdist",
        n_jobs=1, # jobs per worker
        n_blocks = 20, # number of blocks sent to dask
    )
```

## Using GPU acceleration for hamming distance

The Hamming distance metric supports GPU acceleration via [cupy](https://cupy.dev/).

First, install the optional `cupy` dependency:

```
!pip install scirpy[cupy]
```

Then simply run

```
ir.pp.ir_dist(mdata, metric="gpu_hamming")
```

to take advantage of GPU acceleration.
