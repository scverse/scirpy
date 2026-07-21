(distance-metrics)=

# Choosing a sequence distance metric

Sequence distances quantify how dissimilar two immune-receptor sequences are. Smaller distances indicate greater
similarity, and identical sequences have distance zero. Scirpy uses a distance cutoff to connect similar receptors when
defining {term}`clonotype clusters <Clonotype cluster>` or querying reference datasets; sequence pairs above the cutoff
are not connected.

{func}`scirpy.pp.ir_dist` computes distances between all unique VJ and VDJ junction sequences in a dataset and stores
the resulting sparse distance matrices for downstream analyses. Only distances at or below the cutoff are stored.
{func}`scirpy.ir_dist.sequence_dist` computes a sparse distance matrix between arbitrary sequence arrays and likewise
retains only distances at or below the cutoff.

The choice of metric should first reflect the biological question, such as exact clonotype matching, BCR clonal-family
inference, or amino-acid similarity between TCRs. It also depends on the receptor and sequence type: nucleotide
distances retain information about synonymous mutations, whereas amino-acid distances focus on changes to the encoded
receptor. If insertions and deletions are relevant, an edit or alignment-based metric is preferable to a positional
metric such as Hamming distance.

Runtime and memory requirements are secondary considerations, although they can become limiting for large datasets.
The metric, cutoff, number and length of unique sequences, and density of the resulting matrix all affect computational
cost. See {doc}`tutorials/large-datasets` for performance advice.

:::{important}
Distances and cutoffs are not directly comparable between metrics because their scales and scoring rules differ. For
example, a `normalized_hamming` cutoff of `10` permits up to 10% mismatched positions, whereas cutoffs of `10` for
`tcrdist` and `needleman_wunsch` refer to substitution- and gap-based scoring schemes. Select and validate the cutoff
separately for each metric.
:::

## Quick reference

| Metric | Sequence | Description |
| --- | --- | --- |
| [`identity`](#exact-sequence-identity) | nt or aa | Exact matching that connects only identical receptor sequences |
| [`hamming`](#hamming-distances) | nt or aa | Number of mismatched positions between equal-length sequences |
| [`normalized_hamming`](#hamming-distances) | nt or aa | Percentage of mismatched positions in equal-length sequences; commonly used for BCR clonal-family inference |
| [`gpu_hamming`](#hamming-distances) | nt or aa | GPU-accelerated Hamming distance for large comparisons; requires CuPy |
| [`levenshtein`](#levenshtein-distance) | nt or aa | General edit distance counting substitutions, insertions, and deletions with equal cost |
| [`tcrdist`](#tcrdist) | aa | TCR CDR3 distance with amino-acid substitution scores, gap costs, and terminal trimming |
| [`needleman_wunsch`](#needleman-wunsch-distance) | aa | Global amino-acid alignment using a substitution matrix and linear gap penalty |
| [`alignment`](#deprecated-alignment-metrics) | aa | Deprecated global alignment with affine gap penalties; use `needleman_wunsch` when penalties for opening and extending gaps are equal |
| [`fastalignment`](#deprecated-alignment-metrics) | aa | Deprecated alignment with a heuristic mismatch prefilter; use `needleman_wunsch` when penalties for opening and extending gaps are equal |

## Exact sequence identity

Use `identity` when cells should be grouped only if their junction sequences are identical. The metric is fast, easy
to interpret, and requires no metric-specific parameters. The distance between identical sequences is zero, so the
cutoff is always zero.

```python
ir.pp.ir_dist(mdata, metric="identity", sequence="nt")
```

This is the default and the most conservative choice for defining clonotypes. A single sequence difference prevents
two receptors from being connected, so the metric does not account for sequencing errors, somatic hypermutation, or
convergent receptors with similar amino-acid sequences. It provides no graded measure of similarity and does not
account for insertions, deletions, or the biochemical similarity of amino acids. Results also depend on the selected
sequence type: different nucleotide sequences that encode the same amino-acid sequence match only with
`sequence="aa"`.

## Hamming distances

The `hamming` metric counts positions at which two sequences differ. It only compares sequences of equal length.
`normalized_hamming` reports the percentage of different positions instead, providing a common scale across
equal-length sequence pairs of varying lengths.

Hamming distances are fast, easy to interpret, and work well when substitutions are the main source of variation and
every mismatch should have the same weight. They are therefore useful for large datasets and, for example, for
comparing equal-length BCR nucleotide junctions that differ through somatic point mutations.

They are less suitable when insertions, deletions, or the biochemical similarity of amino-acid substitutions matter.
A single insertion shifts all subsequent positions and can make otherwise similar sequences appear highly dissimilar;
in these cases, use an edit or alignment-based metric instead.

For BCR data, we recommend `normalized_hamming` on nucleotide junction sequences because somatic hypermutation acts
at the nucleotide level {cite}`Yaari.2015`. A cutoff of `15`, corresponding to at least 85% sequence identity, can be
used as a starting point and should be adapted to the dataset. See the {doc}`BCR analysis tutorial
<tutorials/tutorial_5k_bcr>` for a detailed example.

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
needed to transform one sequence into another. It is easy to interpret, works with nucleotide and amino-acid
sequences of different lengths, and is useful when insertions or deletions are expected.

All operations have unit cost. Consequently, the metric does not distinguish conservative from unlikely amino-acid
substitutions and does not use a substitution matrix or receptor-specific scoring. It is also slower than Hamming
distance, particularly for large datasets.

```python
ir.pp.ir_dist(mdata, metric="levenshtein", sequence="aa", cutoff=2)
```

## TCRdist

Use `tcrdist` for TCR CDR3 amino-acid similarity following the TCRdist scoring scheme {cite}`TCRdist`. The metric
combines substitution scores, length differences, terminal trimming, and configurable gap placement. Its defaults
match the original TCRdist parameters.

Unlike Hamming or Levenshtein distance, `tcrdist` accounts for the biochemical similarity of amino-acid substitutions
and is faster than a full sequence alignment. However, its default gap placement and terminal trimming are heuristic,
differences at the trimmed sequence ends are ignored, and its distances and cutoffs are less intuitive.

From a technical perspective, `tcrdist` can also be applied to BCR amino-acid junction sequences. However, this use is
less well supported by the literature than nucleotide-based distances for BCR clonal-family inference
{cite}`Yaari.2015`, so its parameters and cutoff require application-specific validation.

```python
ir.pp.ir_dist(
    mdata,
    metric="tcrdist",
    sequence="aa",
    cutoff=15,
)
```

[BLOSUM62](https://doi.org/10.1073/pnas.89.22.10915) is the default amino-acid substitution matrix. The TCR-specific
[TCRBLOSUM](https://doi.org/10.1093/bib/bbae602) substitution matrices can be selected with
`base_matrix="tcrblosum"`; when using {func}`scirpy.pp.ir_dist`, Scirpy automatically selects the alpha-chain matrix
for VJ sequences and the beta-chain matrix for VDJ sequences.

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

The metric optimizes gap placement, supports sequences of different lengths, and uses a substitution matrix to account
for the biochemical similarity of amino acids. Unlike `tcrdist`, it compares the complete sequence without
TCR-specific trimming or gap-placement assumptions.

As a dynamic-programming alignment, it is more computationally expensive than Hamming distance or `tcrdist`. Global
alignment is also less suitable when only a local region is expected to be similar. Moreover, useful parameter values
and cutoffs are less intuitive than for simple edit distances.

```python
ir.pp.ir_dist(
    mdata,
    metric="needleman_wunsch",
    sequence="aa",
    cutoff=10,
    gap_penalty=4,
)
```

The default substitution matrix is [BLOSUM62](https://doi.org/10.1073/pnas.89.22.10915). As with `tcrdist`,
`base_matrix="tcrblosum"` enables chain-specific [TCRBLOSUM](https://doi.org/10.1093/bib/bbae602) matrices.

(deprecated-alignment-metrics)=

## Deprecated alignment metrics

The `alignment` and `fastalignment` metrics are deprecated. Both use
[BLOSUM62](https://doi.org/10.1073/pnas.89.22.10915) and affine-gap parameters through the optional Parasail
dependency. `alignment` applies lossless length-based prefiltering, while `fastalignment` adds a heuristic mismatch
filter that improves performance but can produce false negatives.

Both metrics allow separate penalties for opening and extending a gap. When their `gap_open` and `gap_extend`
parameters are equal, use `needleman_wunsch` for substantially faster execution. To retain the same linear gap cost,
set the `gap_penalty` parameter of `needleman_wunsch` to the value of `gap_open` (= `gap_extend`). Needleman-Wunsch is
not an equivalent replacement when different gap-open and gap-extension penalties are required.

## Choosing a cutoff

The cutoff determines which sequence pairs are retained in the sparse distance matrix. Distances greater than the
cutoff are discarded. Larger cutoffs produce denser matrices, require more memory, and connect more receptors into
the same clonotype clusters.

Scirpy offsets stored distances by one so that identical sequences can be represented in a sparse matrix:

- stored value `1` represents true distance `0`;
- stored value `d + 1` represents true distance `d`;
- stored value `0` means that the true distance is greater than the cutoff.

Choose and validate the cutoff separately for each metric. Scirpy's default cutoff for a metric can serve as a starting
point, but should not be assumed to be appropriate for every dataset. Depending on the application, useful strategies
include using thresholds established in the literature, inspecting nearest-neighbor distance distributions, and
evaluating cluster stability or agreement with known receptor annotations. Avoid selecting a cutoff solely to
reproduce the matrix density of another metric.

## Custom distance calculators

Advanced users can pass an instance of {class}`scirpy.ir_dist.metrics.DistanceCalculator` instead of a metric name.
This makes it possible to implement application-specific distances while retaining Scirpy's clonotype-clustering and
reference-query workflows.
