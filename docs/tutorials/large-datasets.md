# Working with >1M cells

This page is a work-in-progess collection with advice how to scale up the scirpy workflow beyond 1M cells.

## Use an up-to-date version

Scalability has been a major focus of recent developments in Scirpy. Make sure you use the latest version
when working with large datasets to take advantage of all speedups.

## Choose an appropriate distance metric for `pp.ir_dist`

Some distance metrics are significantly faster than others. Here are the distance metrics, roughly ordered by speed:

`identity` > `gpu_hamming` > `hamming` = `normalized_hamming` > `tcrdist` > `levenshtein` > `fastalignment` > `alignment`

TCRdist, fastalignment and alignment are conceptually very similar, but tcrdist is by far the fastest. For this
reason, we'd always recommend to go with `tcrdist`, when looking for a metric taking into account a substitution matrix.

## Multi-machine paralellization with dask

## Using GPU acceleration for hamming distance

The Hamming distance metric supports GPU acceleration via [cupy](https://cupy.dev/).

First, install the optional `cupy` dependency:

```
!pip install scirpy[cupy]
```
