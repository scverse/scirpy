# Quickstart

Scirpy analyses adaptive immune receptor (IR) data next to gene expression, stored as a {class}`~mudata.MuData` object with a `gex` and an `airr` modality.
The example below loads 3k T cells from cancer patients, checks their receptor chains, defines clonotypes and plots clonal expansion.

```python
import scirpy as ir

mdata = ir.datasets.wu2020_3k()

ir.pp.index_chains(mdata)
ir.tl.chain_qc(mdata)
ir.pp.ir_dist(mdata)
ir.tl.define_clonotypes(mdata, receptor_arms="all", dual_ir="primary_only")
ir.tl.clonal_expansion(mdata)

ir.pl.group_abundance(mdata, groupby="airr:chain_pairing", target_col="gex:source")
ir.pl.clonal_expansion(mdata, target_col="clone_id", groupby="gex:source")
```

{doc}`tutorials/tutorial_3k_tcr` analyses the same data in depth, {doc}`data-structure` explains how scirpy stores receptor data, and {doc}`api` lists every function.
