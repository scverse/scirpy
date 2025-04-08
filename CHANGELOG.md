# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixes

- Ensure that clonotype network plots don't have any axis ticks ([#607](https://github.com/scverse/scirpy/pull/607)).

### Chore

- Update template to v0.5.0 ([#608](https://github.com/scverse/scirpy/pull/608/), [#606](https://github.com/scverse/scirpy/pull/606/))

## v0.22.0

### Fixes

- Fix missing kwargs in `pl.repertoire_overlap` ([#599](https://github.com/scverse/scirpy/pull/599)).

### Additions

- Add `tl.mutational_load` to compute absolute and relative mutational load on IMGT-aligned sequences.
  This is useful for assessin somatic hypermutation in B cells ([#573](https://github.com/scverse/scirpy/pull/573)).

## v0.21.0

### Additions

- Add GPU implementation of Hamming distance ([#541](https://github.com/scverse/scirpy/pull/541))

### Documentation

- Add tutorial with tips for large datasets ([#541](https://github.com/scverse/scirpy/pull/541))

## v0.20.1

### Fixes

- Exclude `logomaker` v0.8.5 ([#589](https://github.com/scverse/scirpy/pull/589))

## v0.20.0

### Backwards-incompatible changes

- The format of storing the results of `tl.define_clonotypes`/`tl.define_clonotype_clusters` in `adata.uns` has changed.
  Older versions of Scirpy won't be able to run downstream functions (e.g. `tl.clonotype_network`) on AnnData objects
  created with Scirpy v0.20 or later. This change was necessary to speed up writing results to `h5ad` when working
  with large datasets ([#556](https://github.com/scverse/scirpy/pull/556)).

### Additions

- Add `pl.logoplot_cdr3_motif` that allows to plot sequence logos of
  CDR3 sequences using [logomaker](https://logomaker.readthedocs.io/en/latest/) ([#534](https://github.com/scverse/scirpy/pull/534)).

### Fixes

- Make `datasets.vdjdb` compatible with the latest release of VDJDB ([#578](https://github.com/scverse/scirpy/pull/578)).

### Documentation

- Add a tutorial for BCR analysis with Scirpy ([#542](https://github.com/scverse/scirpy/pull/542)).
- Fix typo in `pp.index_chains` methods description ([#570](https://github.com/scverse/scirpy/pull/570))

## v0.19.0

### Additions

- Add a `mask_obs` argument to `tl.clonotype_network` that allows to compute the clonotype networks on a subset of the cells ([#557](https://github.com/scverse/scirpy/pull/557)).
- Add `datasets.stephenson2021_5k`, an example dataset for the upcoming BCR tutorial ([#565](https://github.com/scverse/scirpy/pull/565))

### Fixes

- Add all optional dependencies required for testing to the `[test]` dependency group ([#562](https://github.com/scverse/scirpy/pull/562)).
- Unpin AnnData version ([#551](https://github.com/scverse/scirpy/pull/551))

## v0.18.0

### Additions

- Isotypically included B cells are now labelled as `receptor_subtype="IGH+IGK/L"` instead of `ambiguous` in `tl.chain_qc` ([#537](https://github.com/scverse/scirpy/pull/537)).
- Added the `normalized_hamming` metric to `pp.ir_dist` that accounts for differences in CDR3 sequence length ([#512](https://github.com/scverse/scirpy/pull/512)).
- `tl.define_clonotype_clusters` now has an option to require J genes to match (`same_j_gene=True`) in addition to `same_v_gene`. ([#470](https://github.com/scverse/scirpy/pull/470)).

### Performance improvements

- The hamming distance has been reimplemented with numba, achieving a significant speedup ([#512](https://github.com/scverse/scirpy/pull/512)).
- Clonotype clustering has been accelerated leveraging sparse matrix operations ([#470](https://github.com/scverse/scirpy/pull/470)).

### Fixes

- Fix that `pl.clonotype_network` couldn't use non-standard obsm key ([#545](https://github.com/scverse/scirpy/pull/545)).

### Other changes

- Make `parasail` an optional dependency since it is hard to install it on ARM CPUs. `TCRdist` is now the
  recommended default distance metric which is much faster than parasail-based pairwise sequence alignments while
  providing very similar results ([#547](https://github.com/scverse/scirpy/pull/547)).
- Drop support for Python 3.9 in accordance with [SPEC0](https://scientific-python.org/specs/spec-0000/) ([#546](https://github.com/scverse/scirpy/pull/546))

## v0.17.2

### Fixes

- Detection of CPU count in `define_clonotype_clusters` was broken ([#527](https://github.com/scverse/scirpy/pull/527))

## v0.17.1

### Fixes

- Compatibility with numpy 2.0 ([#525](https://github.com/scverse/scirpy/pull/525))

### Chore

- scverse template update to v0.4 ([#519](https://github.com/scverse/scirpy/pull/519))

## v0.17.0

### Additions

- Add "TCRdist" as new metric ([#502](https://github.com/scverse/scirpy/pull/502))

### Fixes

- Fix issue with detecting the number of available CPUs on MacOS ([#518](https://github.com/scverse/scirpy/pull/502))

## v0.16.1

### Fixes

- Fix default value for `n_jobs` in `ir.tl.ir_query` that could lead to an error ([#498](https://github.com/scverse/scirpy/pull/498)).
- Update description of D50 diversity metric in documentation ([#499](https://github.com/scverse/scirpy/pull/498)).
- Fix `clonotype_modularity` not being able to store result in MuData in some cases ([#504](https://github.com/scverse/scirpy/pull/504)).
- Fix issue with creating sparse matrices from generators with the latest scipy version ([#504](https://github.com/scverse/scirpy/pull/504))

## v0.16.0

### Backwards-incompatible changes

- Use the `umi_count` field instead of `duplicate_count` to store UMI counts. The field `umi_count` has been added to
  the AIRR Rearrangement standard in [version 1.4](https://docs.airr-community.org/en/latest/news.html#version-1-4-1-august-27-2022) ([#487](https://github.com/scverse/scirpy/pull/487)).
  Use of `duplicate_count` for UMI counts is now discouraged. Scirpy will use `umi_count` in all `scirpy.io` functions.
  It will _not_ change AIRR data that is read through `scirpy.io.read_airr` that still uses the `duplicate_count` column.
  Scirpy remains compatible with datasets that still use `duplicate_count`. You can update your dataset using

    ```python
    adata.obsm["airr"]["umi_count"] = adata.obsm["airr"]["duplicate_count"]
    ```

### Other

- the `io.to_dandelion` and `io.from_dandelion` interoperability functions now rely on the implementation provided by Dandelion itself ([#483](https://github.com/scverse/scirpy/pull/483)).

## v0.15.0

### Fixes

- Fix incompatibility with `scipy` 1.12 ([#484](https://github.com/scverse/scirpy/pull/484))
- Fix incompatibility with `adjustText` 1.0 ([#477](https://github.com/scverse/scirpy/pull/477))
- Reduce overall importtime by deferring the import of the `airr` package until it is actually used. ([#473](https://github.com/scverse/scirpy/pull/473))

### New features

- Speed up alignment distances by pre-filtering. There are two filtering strategies: A (lossless) length-based filter
  and a heuristic based on the expected penalty per mismatch. This is implemented in the `FastAlignmentDistanceCalculator`
  class which supersedes the `AlignmentDistanceCalculator` class, which is now deprecated. Using the `"alignment"` metric
  in `pp.ir_dist` now uses the `FastAlignmentDistanceCalculator` with only the lenght-based filter activated.
  Using the `"fastalignment"` activates the heuristic, which is significantly faster, but results in some false-negatives. ([#456](https://github.com/scverse/scirpy/pull/456))
- Switch to [joblib/loky](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) as a backend for parallel
  processing in `pp.ir_dist`. Joblib enables to switch to alternative backends that support out-of-machine computing
  (e.g. `dask`, `ray`) via the `parallel_config` context manager. Additionally, chunk sizes are now adjusted dynamically based on the problem size. ([#473](https://github.com/scverse/scirpy/pull/473))

### Documentation

- The default values of the distance calculator classes in `ir_dist.metrics` was unclear. The default value is now
  set in the classes. In `pp.ir_dist` and `ir_dist.sequence_dist`, no cutoff argument is passed to the metrics
  objects, unless one is explicitly specified (previously `None` was passed by default).

## v0.14.0

### Breaking changes

- Reimplement `pp.index_chains` using numba and awkward array functions, achieving a significant speedup. This function
  behaves exactly like the previous version _except_ that callback functions passed to the `filter` arguments
  must now be vectorized over an awkward array, e.g. to check if a `junction_aa` field is present you could
  previously pass `lambda x: x['junction_aa'] is not None`, now an accepted version would be
  `lambda x: ~ak.is_none(x["junction_aa"], axis=-1)`. To learn more about native awkward array functions, please
  refer to the [awkward array documentation](https://awkward-array.org/doc/main/reference/index.html). ([#444](https://github.com/scverse/scirpy/pull/444))

### Additions

- The `clonal_expansion` function now supports a `breakpoints` argument for more flexible "expansion categories".
  The `breakpoints` argument supersedes the `clip_at` parameter, which is now deprecated. ([#439](https://github.com/scverse/scirpy/pull/439))

### Fixes

- Fix that `define_clonotype_clusters` could not retreive `within_group` columns from MuData ([#459](https://github.com/scverse/scirpy/pull/459))
- Fix that AIRR Rearrangment fields of integer types could not be written when their value was None ([#465](https://github.com/scverse/scirpy/pull/465))

## v0.13.1

### Fixes

- Fix that `clonotype_modularity` could not run with AnnData object ([#421](https://github.com/scverse/scirpy/pull/421)).
- Fix usage of wrong column in 3k tutorial ([#423](https://github.com/scverse/scirpy/pull/423))
- Change igraph dependency from `python-igraph` to `igraph` ([#436](https://github.com/scverse/scirpy/pull/436))
- Fix that `group_abundance` didn't work when AIRR data was stored in a different MuData slot than `airr` ([#438](https://github.com/scverse/scirpy/pull/438))

(v0.13)=

## v0.13.0 - new data structure based on awkward arrays

This update introduces a new datastructure based on [awkward arrays](https://awkward-array.org/doc/main/).
The new datastructure is described in more detail [in the documentation](https://scirpy.scverse.org/en/latest/data-structure.html) and is considered the "official" way of representing AIRR data for scverse core and ecosystem packages.

Benefits of the new data structure include:

- a more natural, lossless representation of [AIRR Rearrangement data](https://docs.airr-community.org/en/latest/datarep/rearrangements.html)
- separation of AIRR data and the [receptor model](https://scirpy.scverse.org/en/latest/ir-biology.html), thereby getting rid of previous limitations (e.g. "only productive chains") and enabling other use-cases (e.g. spatial AIRR data) in the future.
- clean `adata.obs` as AIRR data is not expanded into columns
- support for [MuData](https://mudata.readthedocs.io/en/latest/) for working with paired gene expression and AIRR data as separate modalities.

The overall workflow stays the same, however this update required several backwards-incompatible changes which are summarized below.

### Backwards-incompatible changes

#### New data structure

Closes issue https://github.com/scverse/scirpy/issues/327.

Changed behavior:

- there are no "has_ir" and "multichain" columns in `adata.obs` anymore
- By default all fields are imported from AIRR rearrangement and 10x data.
- The restriction that all chains added to an `AirrCell` must have the same fields has been removed. Missing fields are automatically filled with missing values.
- `io.upgrade_schema` can update from v0.7 to v0.13 schema. AnnData objects generated with scirpy `<= 0.6.x` cannot be read anymore.
- `pl.spectratype` now has a `chain` attributed and the meaning of the `cdr3_col` attribute has changed.

New functions:

- `pp.index_chains`
- `pp.merge_chains`

Removed functions:

- `pp.merge_with_ir`
- `pp.merge_airr_chains`

#### API supporting MuData

Closes issue https://github.com/scverse/scirpy/issues/383

All functions take (where applicable) the additional, optional keyword arguments

- `airr_mod`: the modality in MuData that contains AIRR information (default: "airr")
- `airr_key`: the slot in `adata.obsm` that contains AIRR rearrangement data (default: "airr")
- `chain_idx_key`: the slot in `adata.obsm` that contains indices specifying which chains in `adata.obsm[airr_key]` are the primary/secondary chains etc.

New class:

- `util.DataHandler`

#### Updated example datasets

The example datasets have been updated to be based on the new datastructure and are now based on MuData.

- The example datasets have been regenerated from scratch using the loader notebooks described in the docstring. The Maynard dataset gene expression is now based on values generated with Salmon instead of RSEM/featurecounts.
- Scirpy now uses [pooch](https://github.com/fatiando/pooch) to manage example datasets.

#### Cleanup

- Removed the deprecated functions `io.from_tcr_objs`, `io.from_ir_objs`, `io.to_ir_objs`, `pp.merge_with_tcr`, `pp.tcr_neighbors`, `pp.ir_neighbors`, `tl.chain_pairing`
- Removed the deprecated classes `TcrCell`, `AirrChain`, `TcrChain`
- Removed the function `pl.cdr_convergence` which was never public anyway.

### Additions

#### Easy-access functions (`scirpy.get`)

Closes issue https://github.com/scverse/scirpy/issues/184

New functions:

- `get.airr`
- `get.obs_context`
- `get.airr_context`

### Fixes

- Several type hints that were previously inaccurate are now updated.
- Fix x-axis labelling in `pl.clonotype_overlap` raises an error if row annotations are not unique for each group.

### Documentation

The documentation has been updated to reflect the changes described above, in particular the [tutorials](https://scirpy.scverse.org/en/latest/tutorials.html) and the page about the [data structure](https://scirpy.scverse.org/en/latest/data-structure.html).

Moreover, the documentation now uses a new design and moved from GitHub pages to ReadTheDocs.org.
Older versions of the documentation are still [accessible from github pages](https://scirpy.scverse.org/en/latest/versions.html).

### Other changes

- Scirpy now adopts the [cookiecutter-scverse](https://github.com/scverse/cookiecutter-scverse) template. The structure
  of this repository has ben adapted accordingly. Also code was reformatted in accordance with the template defaults.
- The minimum required Python version is now 3.9 in accordance with NEP 29
- Increased the minium version of tqdm to 4.63 (See https://github.com/tqdm/tqdm/issues/1082)
- `pl.repertoire_overlap` now _always_ runs `tl.repertoire_overlap` internally and doesn't rely on cached values.
- The mode `dendro_only` in `pl.repertoire_overlap` has been removed.
- Cells that have a receptor, but no CDR3 sequence have previously received a separate clonotype in `tl.define_clonotypes`. Now they are receiving no clonotype (i.e. `np.nan`) as do cells without a receptor.
- The function `tl.clonal_expansion` now returns a `pd.Series` instead of a `np.array` with `inplace=False`
- Removed deprecation for `clonotype_imbalanced`, see https://github.com/scverse/scirpy/issues/330
- The `group_abundance` tool and plotting function used `has_ir` as a default group as we could previously rely on this column being present. With the new datastructure, this is not the case. To no break old code, the `has_ir` column is tempoarily added when requested. The `group_abundance` function will have to be rewritten enitrely in the future, see https://github.com/scverse/scirpy/issues/232
- In `pl.spectratype`, the parameter `groupby` has been replaced by `chain`.
- We now use [isort](https://pycqa.github.io/isort/) to organize imports.
- Static typing has been improved internally (using pylance). It's not perfectly consistent yet, but we will keep working on this in the future.
- Fix IEDB data loader after update of IEDB data formats (#401) and add tests for database import functions.
- `io.read_airr` now tolerates if fields required according to the AIRR standard are missing. The respective fields will be initalized with `None` (#407 by @zktuong).

## v0.12.2

### Fixes

- Fix IEDB data loader after update of IEDB data formats (backport of #401)

## v0.12.1

### Fixes

- Bump min Python version to 3.8; CI update by @grst in https://github.com/scverse/scirpy/pull/381
- Temporarily pin pandas < 2 in #390

### Other Changes

- update pre-commit CI

## v0.12.0

### New Features

- Download IEDB and process it into an AnnData object by @ausserh in https://github.com/scverse/scirpy/pull/377

### Fixes

- Fix working with subplots (#378) by @grst in https://github.com/scverse/scirpy/pull/379

### Documentation

- Fix typos in IR query by @Zethson in https://github.com/scverse/scirpy/pull/374
- Fix a bunch of typos in the docs by @grst in https://github.com/scverse/scirpy/pull/375

### Internal changes

- Fix CI by @grst in https://github.com/scverse/scirpy/pull/376

### New Contributors

- @Zethson made their first contribution in https://github.com/scverse/scirpy/pull/374
- @ausserh made their first contribution in https://github.com/scverse/scirpy/pull/377

**Full Changelog**: https://github.com/scverse/scirpy/compare/v0.11.2...v0.12.0

## v0.11.2

### Fixes

- Excluded broken python-igraph version (#366)

## v0.11.1

### Fixes

- Solve incompatibility with scipy v1.9.0 (#360)

### Internal changes

- do not autodeploy docs via CI (currently broken)
- updated patched version of scikit-learn

## v0.11.0

### Additions

- Add data loader for BD Rhapsody single-cell immune-cell receptor data (`io.read_bd_rhapsody`) (#351)

### Fixes

- Fix type conversions in `from_dandelion` (#349).
- Update minimal dandelion version

### Documentation

- Rebranding to [scverse](https://scverse.org) (#324, #326)
- Add issue templates
- Fix IMGT typos (#344 by @emjbishop)

### Internal changes

- Bump default CI python version to 3.9
- Use patched version of scikit-bio in CI until https://github.com/biocore/scikit-bio/pull/1813 gets merged

## v0.10.1

### Fixes

- Fix bug in cellranger import (#310 by @ddemaeyer)
- Fix that VDJDB download failed when cache dir was not present (#311)

## v0.10.0

### Additions

This release adds a new feature to query reference databases (#298) comprising

- an extension of `pp.ir_dist` to compute distances to a reference dataset,
- `tl.ir_query`, to match immune receptors to a reference database based on the distances computed with `ir_dist`,
- `tl.ir_query_annotate` and `tl.ir_query_annotate_df` to annotate cells based on the result of `tl.ir_query`, and
- `datasets.vdjdb` which conveniently downloads and processes the latest version of [VDJDB](https://vdjdb.cdr3.net/).

### Fixes

- Bump minimal dependencies for networkx and tqdm (#300)
- Fix issue with `repertoire_overlap` (Fix #302 via #305)
- Fix issue with `define_clonotype_clusters` (Fix #303 via #305)
- Suppress `FutureWarning`s from pandas in tutorials (#307)

### Internal changes

- Update sphinx to >= 4.1 (#306)
- Update black version
- Update the internal folder structure: `tl`, `pp` etc. are now real packages instead of aliases

## v0.9.1

### Fixes

- Scirpy can now import additional columns from Cellranger 6 (#279 by @naity)
- Fix minor issue with `include_fields` in `AirrCell` (#297)

### Documentation

- Fix broken link in README (#296)
- Add developer documentation (#294)

## v0.9.0

### Additions

- Add the new "clonotype modularity" tool which ranks clonotypes by how strongly connected their gene expression neighborhood graph is. (#282).

The below example shows three clonotypes (164, 1363, 942), two of which consist of cells that are transcriptionally related.

<table>
<tr>
<th>
example clonotypes
</th>
<th>
clonotype modularity vs. FDR
</th>
</tr>
<tr>
<td>
<img src="https://user-images.githubusercontent.com/7051479/132355883-2608af74-8d2c-420f-871a-9fa006c4cfd8.png">
</td>
<td>
<img src="https://user-images.githubusercontent.com/7051479/132355941-927e0a9e-5a7c-4df4-b339-1be9535c4279.png">
</td>
</tr>
</table>

### Deprecations

- `tl.clonotype_imbalance` is now deprecated in favor of the new clonotype modularity tool.

### Fixes

- Fix calling locus from gene name in some cases (#288)
- Compatibility with `networkx>=2.6` (#292)

### Minor updates

- Fix some links in README (#284)
- Fix old instances of clonotype in docs (should be clone_id) (#287)

## v0.8.0

### Additions

- `tl.alpha_diversity` now supports all [metrics from scikit-bio](http://scikit-bio.org/docs/0.4.2/generated/skbio.diversity.alpha.html#module-skbio.diversity.alpha), the `D50` metric and custom callback functions (#277 by @naity)

### Fixes

- Handle input data with "productive" chains which don't have a `junction_aa` sequence annotated (#281)
- Fix issue with serialized "extra chains" not being imported correctly (#283 by @zktuong)

### Minor changes

- The CI can now build documentation from pull-requests from forks. PR docs are not deployed to github-pages anymore, but can be downloaded as artifact from the CI run.

## v0.7.1

### Fixes

- Ensure Compatibility with latest version of dandelion ([e78701c](https://github.com/icbi-lab/scirpy/commit/e78701ce5d3ece33688319e0f85a51b02dd06769))
- Add links to older versions of documentation (#275)
- Fix issue, where clonotype analysis couldn't be continued after saving and reloading `h5ad` object (#274)
- Allow "None" values to be present as cell-level attributes during `merge_airr_chains` (#273)

### Minor changes

- Require `anndata >= 0.7.6` in conda tests (#266)

## v0.7.0

This update features a

- change of Scirpy's data structure to improve interoperability with the AIRR standard
- a complete re-write of the clonotype definition module for improved performance.

This required several backwards-incompatible changes. Please read the release notes below and the updated tutorials.

### Backwards-incompatible changes

#### Improve Interoperability by fully supporting the AIRR standard (#241)

Scirpy stores receptor information in `adata.obs`. In this release, we updated the column names to match the [AIRR Rearrangement standard](https://docs.airr-community.org/en/latest/datarep/rearrangements.html#productive). Our data model is now much more flexible, allowing to import arbitrary immune-receptor (IR)-chain related information. Use `scirpy.io.upgrade_schema()` to update existing `AnnData` objects to the latest format.

Closed issues #240, #253, #258, #255, #242, #215.

This update includes the following changes:

- `IrCell` is now replaced by `AirrCell` which has additional functionality
- `IrChain` has been removed. Use a plain dictionary instead.
- CDR3 information is now read from the `junction` and `junction_aa` columns instead of `cdr3_nt` and `cdr3`, respectively.
- Clonotype assignments are now per default stored in the `clone_id` column.
- `expr` and `expr_raw` are now `duplicate_count` and `consensus_count`.
- `{v,d,j,c}_gene` is now `{v,d,j,c}_call`.
- There's now an `extra_chains` column containing all IR-chains that don't fit into our [receptor model](https://icbi-lab.github.io/scirpy/ir-biology.html#receptor-model). These chains are not used by scirpy, but can be re-exported to different formats.
- `merge_with_ir` is now split up into `merge_with_ir` (to merge IR data with transcriptomics data) and `merge_airr_chains` (to merge several adatas with IR information, e.g. BCR and TCR data).
- Tutorial and documentation updates, to reflect these changes
- Sequences are not converted to upper case on import. Scirpy tools that consume the sequences convert them to upper case on-the-fly.
- `{to,from}_ir_objs` has been renamed to `{to,from}_airr_cells`.

#### Refactor CDR3 network creation (#230)

Previously, `pp.ir_neighbors` constructed a `cell x cell` network based on clonotype similarity. This led to performance issues
with highly expanded clonotypes (i.e. thousands of cells with exactly the same receptor configuration). Such cells would
form dense blocks in the sparse adjacency matrix (see issue #217). Another downside was that expensive alignment-distances had
to be recomputed every time the parameters of `ir_neighbors` was changed.

The new implementation computes distances between all _unique receptor configurations_, only considering one instance of highly expanded clonotypes.

Closed issues #243, #217, #191, #192, #164.

This update includes the following changes:

- `pp.ir_neighbors` has been replaced by `pp.ir_dist`.
- The options `receptor_arms` and `dual_ir` have been moved from `pp.ir_neighbors` to `tl.define_clonotypes` and `tl.define_clonotype_clusters`.
- The default key for clonotype clusters is now `cc_{distance}_{metric}` instead of `ct_cluster_{distance}_{metric}`.
- `same_v_gene` now fully respects the options `dual_ir` and `receptor_arms`
- v-genes and receptor types were previously simply appended to clonotype ids (when `same_v_gene=True`). Now clonotypes with different v-genes get assigned a different numeric id.
- Distance metric classes have been moved from `ir_dist` to `ir_dist.metrics`.
- Distances matrices generated by `ir_dist` are now square and symmetric instead of triangular.
- The default value for `dual_ir` is now `any` instead of `primary_only` (Closes #164).
- The API of `clonotype_network` has changed.
- Clonotype network now visualizes cells with identical receptor configurations. The number of cells with identical receptor configurations is shown as point size (and optionally, as color). Clonotype network does not support plotting multiple colors at the same time any more.

| Clonotype network (previous implementation)                                                                    | Clonotype network (now)                                                                                        |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Each dot represents a cell. Cells with identical receptors form a fully connected subnetwork                   | Each dot represents cells with identical receptors. The dot size refers to the number of cells                 |
| ![image](https://user-images.githubusercontent.com/7051479/114389098-dff87800-9b94-11eb-8dd8-cc406024eaa6.png) | ![image](https://user-images.githubusercontent.com/7051479/114389105-e2f36880-9b94-11eb-9a58-68a09e67efe7.png) |

#### Drop Support for Python 3.6

- Support Python 3.9, drop support for Python 3.6, following the numpy guidelines. (#229)

### Fixes

- `tl.clonal_expansion` and `tl.clonotype_convergence` now respect cells with missing receptors and return `nan` for those cells. (#252)

### Additions

- `util.graph.igraph_from_sparse_matrix` allows to convert a sparse connectivity or distance matrix to an `igraph` object.
- `ir_dist.sequence_dist` now also works sequence arrays that contain duplicate entries (#192)
- `from_dandelion` and `to_dandelion` facilitate interaction with the [Dandelion package](https://github.com/zktuong/dandelion) (#240)
- `write_airr` allows to write scirpy's `adata.obs` back to the [AIRR Rearrangement format](https://docs.airr-community.org/en/latest/datarep/rearrangements.html).
- `read_airr` now tries to infer the locus from gene names, if no locus column is present.
- `ir.io.upgrade_schema` allows to upgrade an existing scirpy anndata object to be compatible with the latest version of scirpy
- `define_clonotypes` and `define_clonotype_clusters` now prints a logging message indicating where the results have been stored (#215)

### Minor changes

- `tqdm` now uses IPython widgets to display progress bars, if available
- the `process_map` from `tqdm` is now used to display progress bars for parallel computations instead the custom implementation used previously [f307c2b](https://github.com/icbi-lab/scirpy/pull/230/commits/f307c2b9a6a5b3e86ca0399b6142490d15511177)
- `matplotlib`s "grid lines" are now suppressed by default in all plots.
- Docs from the `master` branch are now deployed to `icbi-lab.github.io/scirpy/develop` instead of the main documentation website. The main website only gets updated on releases.
- Refactored the `_is_na` function that checks if a string evaluates to `None`.
- Fixed outdated documentation of the `receptor_arms` parameter (#264)

## v0.6.1

### Fixes

- Fix an issue where `define_clonotype` failed when the clonotype network had no edges (#236).
- Require pandas >= 1.0 and fix a pandas incompatibility in `merge_with_ir` (#238).
- Ensure consistent order of the spectratype dataframe (#238).

### Minor changes

- Fix missing `bibtex_bibfiles` option in sphinx configuration
- Work around https://github.com/takluyver/flit/issues/383.

## v0.6.0

### Backwards-incompatible changes:

- Set more sensible defaults the the `cutoff` parameter in `ir_neighbors`. The default is now `2` for `hamming` and `levenshtein` distance metrics and `10` for the `alignment` distance metric.

### Additions:

- Add Hamming-distance as additional distance metric for `ir_neighbors` (#216 by @ktpolanski)

### Minor changes:

- Fix MacOS CI (#221)
- Use mamba instead of conda in CI (#216)

## v0.5.0 - Add support for BCRs and gamma-delta TCRs

### Backwards-incompatible changes:

- The [data structure](https://icbi-lab.github.io/scirpy/usage-principles.html#data-structure) has changed. Column have been renamed from `TRA_xxx` and `TRB_xxx` to `IR_VJ_xxx` and `IR_VDJ_xxx`. Additionally a `locus` column has been added for each chain.
- All occurences of `tcr` in the function and class names have been replaced with `ir`. Aliases for the old names have been created and emit a `FutureWarning`.

### Additions:

- There's now a mixed TCR/BCR example dataset (`maynard2020`) available (#211)
- BCR-related amendments to the documentation (#206)
- `tl.chain_qc` which supersedes `chain_pairing`. It additionally provides information about the receptor type.
- `io.read_tracer` now supports gamma-delta T-cells (#207)
- `io.to_ir_objs` allows to convert adata to a list of `IrCells` (#210)
- `io.read_bracer` allows to read-in BraCeR BCR data. (#208)
- The `pp.merge_with_ir` function now can handle the case when both the left and the right `AnnData` object contain immune receptor information. This is useful when integrating both TCR and BCR data into the same dataset. (#210)

### Fixes:

- Fix a bug in `vdj_usage` which has been triggered by the new data structure (#203)

### Minor changes:

- Removed the tqdm monkey patch, as the issue has been resolved upstream (#200)
- Add AIRR badge, as scirpy is now certified to comply with the AIRR software standard v1. (#202)
- Require pycairo >1.20 which provides a windows wheel, eliminating the CI problems.

## v0.4.2

- Include tests into main package (#189)

## v0.4.1

- Fix pythonpublish CI action
- Update black version (and code style, accordingly)
- Changes for AIRR-complicance:
    - Add support level to README
    - Add Biocontainer instructions to README
    - Add a minimal test suite to be ran on conda CI

## v0.4

- Adapt tcr_dist to support second array of sequences (#166). This enables comparing CDR3 sequences against a list of reference sequences.
- Add `tl.clonotype_convergence` which helps to find evidence of convergent evolution (#168)
- Optimize parallel sequence distance calculation (#171). There is now less communication overhead with the worker processes.
- Fixed an error when runing `pp.tcr_neighbors` (#177)
- Improve packaging. Use `setuptools_scm` instead of `get_version`. Remove redundant metadata. (#180). More tests for conda (#180).

## v0.3

- More extensive CI tests (now also testing on Windows, MacOS and testing the conda recipe) (#136, #138)
- Add example images to API documentation (#140)
- Refactor IO to expose TcrCell and TcrChain (#139)
- Create data loading tutorial (#139)
- Add a progressbar to TCR neighbors (#143)
- Move clonotype_network_igraph to tools (#144)
- Add `read_airr` to support the AIRR rearrangement format (#147)
- Add option to take v-gene into account during clonotype definition (#148)
- Store colors in AnnData to ensure consistent coloring across plots (#151)
- Divide `define_clontoypes` into `define_clonotypes` and `define_clonotype_clusters` (#152). Now, the user has to specify explicitly `sequence` and `metric` for both `tl.tcr_neighbors`, `tl.define_clonotype_clusters` and `tl.clonotype_network`. This makes it more straightforward to have multiple, different versions of the clonotype network at the same time. The default parameters changed to `sequence="nt"` and `metric="identity" to comply with the traditional definition of clonotypes. The changes are also reflected in the glossary and the tutorial.
- Update the workflow figure (#154)
- Fix a bug that caused labels in the `repertoire_overlap` heatmap to be mixed up. (#157)
- Add a label to the heatmap annotation in `repertoire_overlap` (#158).

## v0.2

- Documentation overhaul. A lot of docstrings got corrected and improved and the formatting of the documentation now matches scanpy's.
- Experimental function to assess bias in clonotype abundance between conditions (#92)
- Scirpy now has a logo (#123)
- Update default parameters for `clonotype_network`:
    - Edges are now only automatically displayed if plotting < 1000 nodes
    - If plotting variables with many categories, the legend is hidden.
- Update default parameters for alignment-based `tcr_neighbors`
    - The gap extend penalty now equals the gap open penalty (`11`).

## v0.1.2

- Make 10x csv and json import consistent (#109)
- Fix version requirements (#112)
- Fix compatibility issues with pandas > 1 (#112)
- Updates to tutorial and README

## v0.1.1

- Update documentation about T-cell receptor model (#4, #10)
- Update README
- Fix curve plots (#31)
- Host datasets on GitHub (#104)

## v0.1

Initial release for pre-print
