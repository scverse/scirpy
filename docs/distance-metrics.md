(distance-metrics)=

# Choosing a sequence distance metric

Scirpy uses sequence distances to connect similar immune receptors when defining
{term}`clonotype clusters <Clonotype cluster>` or querying reference datasets.
{func}`scirpy.pp.ir_dist` computes distances between all unique VJ and VDJ junction sequences in a dataset, while
{func}`scirpy.ir_dist.sequence_dist` computes distances between arbitrary sequence arrays.

The choice of metric depends primarily on the biological question, receptor type, sequence type, and whether
insertions or deletions should be considered. Runtime and memory requirements are secondary considerations; see
{doc}`tutorials/large-datasets` for performance advice.

:::{important}
Distances and cutoffs are not directly comparable between metrics. For example, a cutoff of `10` has a different
meaning for `normalized_hamming`, `tcrdist`, and `needleman_wunsch`.
:::

## Quick reference

| Use case | Recommended metric | Sequence | Description |
| --- | --- | --- | --- |
| Exact receptor matching | `identity` | nt or aa | Matches identical sequences |
| BCR clonal-family inference | `normalized_hamming` | nt or aa | Percentage of mismatched positions |
| Position-wise mismatches between equal-length sequences | `hamming` | nt or aa | Number of mismatched positions |
| Large equal-length comparisons on a GPU | `gpu_hamming` | nt or aa | GPU-accelerated Hamming distance |
| General edit distance | `levenshtein` | nt or aa | Minimum number of single-character edits |
| TCR CDR3 similarity | `tcrdist` | aa | TCR-specific substitution and gap costs |
| Global amino-acid alignment with linear gaps | `needleman_wunsch` | aa | Global alignment with a linear gap penalty |
| Legacy global alignment | `alignment` | aa | Global alignment with affine gap penalties |
| Legacy heuristic alignment | `fastalignment` | aa | Heuristically prefiltered alignment distance |

## Exact sequence identity

Use `identity` when cells should be grouped only if their junction sequences are identical. The true distance between
identical sequences is zero, and the cutoff is therefore always set to zero.

```python
ir.pp.ir_dist(mdata, metric="identity", sequence="nt")
```

This is the default and the most conservative choice for defining clonotypes. It does not account for sequencing
errors, somatic hypermutation, or convergent receptors with similar amino-acid sequences.

## Hamming distances

The `hamming` metric counts positions at which two sequences differ. It only compares sequences of equal length.
`normalized_hamming` reports the percentage of different positions instead, providing a common scale for comparisons
of sequence pairs with different lengths.

For BCR data, we recommend `normalized_hamming` on nucleotide junction sequences because somatic hypermutation acts
at the nucleotide level {cite}`Yaari.2015`. A cutoff of `15`, corresponding to at least 85% sequence identity, can be
used as a starting point and should be adapted to the dataset.

```python
ir.pp.ir_dist(
    mdata,
    metric="normalized_hamming",
    sequence="nt",
    cutoff=15,
    histogram=True,
)
```

The `gpu_hamming` metric computes the non-normalized Hamming distance on a compatible GPU. It is useful for large
datasets but does not change the biological interpretation of the metric. See
{ref}`the GPU instructions <gpu-hamming-distance>` for installation and usage.

## Levenshtein distance

The `levenshtein` metric counts the minimum number of single-character substitutions, insertions, and deletions
needed to transform one sequence into another. All operations have unit cost. It is useful as a general-purpose edit
distance, but it does not distinguish conservative from unlikely amino-acid substitutions.

```python
ir.pp.ir_dist(mdata, metric="levenshtein", sequence="aa", cutoff=2)
```

## TCRdist

Use `tcrdist` for TCR CDR3 amino-acid similarity following the TCRdist scoring scheme {cite}`TCRdist`. The metric
combines substitution scores, length differences, terminal trimming, and configurable gap placement. Its defaults
match the original TCRdist parameters.

```python
ir.pp.ir_dist(
    mdata,
    metric="tcrdist",
    sequence="aa",
    cutoff=15,
)
```

BLOSUM62 is used by default. TCRBLOSUM matrices can be selected with `base_matrix="tcrblosum"`; when using
{func}`scirpy.pp.ir_dist`, Scirpy automatically selects the alpha-chain matrix for VJ sequences and the beta-chain
matrix for VDJ sequences.

```python
ir.pp.ir_dist(
    mdata,
    metric="tcrdist",
    sequence="aa",
    cutoff=15,
    base_matrix="tcrblosum",
)
```

## Needleman-Wunsch distance

Use `needleman_wunsch` when the complete amino-acid junction sequence should be aligned globally. Scirpy implements
Needleman-Wunsch alignment with a linear gap penalty. It converts the alignment score into a distance relative to the
best self-alignment score of the two sequences.

```python
ir.pp.ir_dist(
    mdata,
    metric="needleman_wunsch",
    sequence="aa",
    cutoff=10,
    gap_penalty=4,
)
```

The default substitution matrix is BLOSUM62. As with `tcrdist`, `base_matrix="tcrblosum"` enables chain-specific
TCRBLOSUM matrices.

(deprecated-alignment-metrics)=

## Deprecated alignment metrics

The `alignment` and `fastalignment` metrics are deprecated. Both use BLOSUM62 and affine-gap parameters through the
optional Parasail dependency. `alignment` applies lossless length-based prefiltering, while `fastalignment` adds a
heuristic mismatch filter that improves performance but can produce false negatives.

Use `needleman_wunsch` when `gap_open` and `gap_extend` are equal. To retain the same linear gap cost, set
`gap_penalty=gap_open`. Needleman-Wunsch is not an equivalent replacement when different gap-open and gap-extension
penalties are required.

## Choosing a cutoff

The cutoff determines which sequence pairs are retained in the sparse distance matrix. Distances greater than the
cutoff are discarded. Larger cutoffs produce denser matrices, require more memory, and connect more receptors into
the same clonotype clusters.

Scirpy offsets stored distances by one so that identical sequences can be represented in a sparse matrix:

- stored value `1` represents true distance `0`;
- stored value `d + 1` represents true distance `d`;
- stored value `0` means that the true distance is greater than the cutoff.

Choose and validate the cutoff separately for each metric. Depending on the application, useful strategies include
using thresholds established in the literature, inspecting nearest-neighbor distance distributions, and evaluating
cluster stability or agreement with known receptor annotations. Avoid selecting a cutoff solely to reproduce the
matrix density of another metric.

## Custom distance calculators

Advanced users can pass an instance of {class}`scirpy.ir_dist.metrics.DistanceCalculator` instead of a metric name.
This makes it possible to implement application-specific distances while retaining Scirpy's clonotype-clustering and
reference-query workflows.
