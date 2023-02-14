"""Compute distances between immune receptor sequences"""
from anndata import AnnData
from typing import Optional, Sequence, Union
from typing import Literal
import numpy as np
from scanpy import logging
from ..util import _is_na, deprecated
from scipy.sparse import csr_matrix
from ..util import _doc_params
from . import metrics
from ..io._util import _check_upgrade_schema
import itertools


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `sequence_dist`. The old version will be removed in a future release. "
)
def tcr_dist(*args, **kwargs):
    return sequence_dist(*args, **kwargs)


def TcrNeighbors(*args, **kwargs):
    raise RuntimeError("TcrNeighbors has been renamed in v0.5.0 and removed in v0.7.0.")


def IrNeighbors(*args, **kwargs):
    raise RuntimeError(
        "IrNeighbors has been removed in v0.7.0. Use either `ir_dist` "
        "or `sequence_dist` for that functionality. "
    )


MetricType = Union[
    Literal["alignment", "identity", "levenshtein", "hamming"],
    metrics.DistanceCalculator,
]

_doc_metrics = """\
metric
    You can choose one of the following metrics:
      * `identity` -- 1 for identical sequences, 0 otherwise.
        See :class:`~scirpy.ir_dist.metrics.IdentityDistanceCalculator`.
        This metric implies a cutoff of 0.
      * `levenshtein` -- Levenshtein edit distance.
        See :class:`~scirpy.ir_dist.metrics.LevenshteinDistanceCalculator`.
      * `hamming` -- Hamming distance for CDR3 sequences of equal length.
        See :class:`~scirpy.ir_dist.metrics.HammingDistanceCalculator`.
      * `alignment` -- Distance based on pairwise sequence alignments using the
        BLOSUM62 matrix. This option is incompatible with nucleotide sequences.
        See :class:`~scirpy.ir_dist.metrics.AlignmentDistanceCalculator`.
      * any instance of :class:`~scirpy.ir_dist.metrics.DistanceCalculator`.
"""

_doc_cutoff = """\
cutoff
    All distances `> cutoff` will be replaced by `0` and eliminated from the sparse
    matrix. A sensible cutoff depends on the distance metric, you can find
    information in the corresponding docs. If set to `None`, the cutoff
    will be `10` for the `alignment` metric, and `2` for `levenshtein` and `hamming`.
    For the identity metric, the cutoff is ignored and always set to `0`.
"""


def _get_metric_key(metric: MetricType) -> str:
    return "custom" if isinstance(metric, metrics.DistanceCalculator) else metric  # type: ignore


def _get_distance_calculator(
    metric: MetricType, cutoff: Union[int, None], *, n_jobs=None, **kwargs
):
    """Returns an instance of :class:`~scirpy.ir_dist.metrics.DistanceCalculator`
    given a metric.

    A cutoff of 0 will always use the identity metric.
    """

    if cutoff == 0 or metric == "identity":
        metric = "identity"
        cutoff = 0

    if isinstance(metric, metrics.DistanceCalculator):
        dist_calc = metric
    elif metric == "alignment":
        dist_calc = metrics.AlignmentDistanceCalculator(
            cutoff=cutoff, n_jobs=n_jobs, **kwargs
        )
    elif metric == "identity":
        dist_calc = metrics.IdentityDistanceCalculator(cutoff=cutoff, **kwargs)
    elif metric == "levenshtein":
        dist_calc = metrics.LevenshteinDistanceCalculator(
            cutoff=cutoff, n_jobs=n_jobs, **kwargs
        )
    elif metric == "hamming":
        dist_calc = metrics.HammingDistanceCalculator(
            cutoff=cutoff, n_jobs=n_jobs, **kwargs
        )
    else:
        raise ValueError("Invalid distance metric.")

    return dist_calc


@_check_upgrade_schema()
@_doc_params(metric=_doc_metrics, cutoff=_doc_cutoff, dist_mat=metrics._doc_dist_mat)
def _ir_dist(
    adata: AnnData,
    reference: AnnData = None,
    *,
    metric: MetricType = "identity",
    cutoff: Union[int, None] = None,
    sequence: Literal["aa", "nt"] = "nt",
    key_added: Union[str, None] = None,
    inplace: bool = True,
    n_jobs: Union[int, None] = None,
) -> Union[dict, None]:
    """
    Computes a sequence-distance metric between all unique :term:`VJ <Chain locus>`
    :term:`CDR3` sequences and between all unique :term:`VDJ <Chain locus>`
    :term:`CDR3` sequences.

    This is a required proprocessing step for clonotype definition and clonotype
    networks and for querying reference databases.

    {dist_mat}

    Parameters
    ----------
    adata
        annotated data matrix
    reference
        Another :class:`~anndata.AnnData` object, can be either a second dataset with
        :term:`IR` information or a epitope database.
        If specified, will compute distances between the sequences in `adata` and the
        sequences in `reference`. Otherwise computes pairwise distances
        of the sequences in `adata`.
    {metric}
    {cutoff}
    sequence
        Compute distances based on amino acid (`aa`) or nucleotide (`nt`) sequences.
    key_added
        Dictionary key under which the results will be stored in `adata.uns` if
        `inplace=True`. Defaults to `ir_dist_{{sequence}}_{{metric}}` or
        `ir_dist_{{name}}_{{sequence}}_{{metric}}` if `reference` is specified.
        If `metric` is an instance of :class:`scirpy.ir_dist.metrics.DistanceCalculator`,
        `{{metric}}` defaults to `custom`.
        `{{name}}` is taken from `reference.uns["DB"]["name"]`. If `reference` does not have a
        `"DB"` entry, `key_added` needs to be specified manually.
    inplace
        If true, store the result in `adata.uns`. Otherwise return a dictionary
        with the results.
    n_jobs
        Number of cores to use for distance calculation. Passed on to
        :class:`scirpy.ir_dist.metrics.DistanceCalculator`.

    Returns
    -------
    Depending on the value of `inplace` either returns nothing or a dictionary
    with sparse, pairwise distance matrices for all `VJ` and `VDJ`
    sequences.
    """
    key = "junction_aa" if sequence == "aa" else "junction"
    result = {
        "VJ": dict(),
        "VDJ": dict(),
        "params": {"metric": str(metric), "sequence": sequence, "cutoff": cutoff},
    }
    if inplace and key_added is None:
        if reference is not None:
            try:
                db_info = reference.uns["DB"]
                key_added = (
                    f"ir_dist_{db_info['name']}_{sequence}_{_get_metric_key(metric)}"
                )
            except KeyError:
                raise ValueError(
                    'If reference does not contain a `.uns["DB"]["name"]` entry, '
                    "you need to manually specify `key_added`."
                )
        else:
            key_added = f"ir_dist_{sequence}_{_get_metric_key(metric)}"
    if reference is not None:
        try:
            result["params"]["DB"] = reference.uns["DB"]
        except KeyError:
            result["params"]["DB"] = "AnnData without `.uns['DB'] metadata."

    # get all unique seqs for VJ and VDJ
    def _get_unique_seqs(tmp_adata, chain_type):
        """Get all unique sequences for a chain type"""
        obs_col = "IR_{chain_type}_{chain_id}_{key}"
        tmp_seqs = np.concatenate(
            [
                tmp_adata.obs[
                    obs_col.format(chain_type=chain_type, chain_id=chain_id, key=key)
                ].values
                for chain_id in ["1", "2"]
            ]
        )
        return np.unique([x.upper() for x in tmp_seqs[~_is_na(tmp_seqs)]])

    for i, tmp_adata in enumerate([adata, reference]):
        if tmp_adata is not None:
            for chain_type in ["VJ", "VDJ"]:
                tmp_key = "seqs2" if i == 1 else "seqs"
                unique_seqs = _get_unique_seqs(tmp_adata, chain_type)
                if tmp_key == "seqs2" and not len(unique_seqs):
                    logging.warning(
                        "No sequences found in reference anndata object. "
                        "Are you sure you chose the right sequence type (`aa` vs. `nt`)?"
                    )
                result[chain_type][tmp_key] = unique_seqs

    # compute distance matrices
    dist_calc = _get_distance_calculator(metric, cutoff, n_jobs=n_jobs)
    for chain_type in ["VJ", "VDJ"]:
        logging.info(
            f"Computing sequence x sequence distance matrix for {chain_type} sequences."
        )  # type: ignore
        result[chain_type]["distances"] = dist_calc.calc_dist_mat(
            result[chain_type]["seqs"], result[chain_type].get("seqs2", None)
        ).tocsr()

    # return or store results
    if inplace:
        adata.uns[key_added] = result
    else:
        return result


@_doc_params(metric=_doc_metrics, cutoff=_doc_cutoff, dist_mat=metrics._doc_dist_mat)
def sequence_dist(
    seqs: Sequence[str],
    seqs2: Optional[Sequence[str]] = None,
    *,
    metric: MetricType = "identity",
    cutoff: Union[None, int] = None,
    n_jobs: Union[None, int] = None,
    **kwargs,
) -> csr_matrix:
    """
    Calculate a sequence x sequence distance matrix.

    {dist_mat}

    When `seqs` or `seqs2` includes non-unique values, the function internally
    uses only unique sequences to calculate the distances. Note that, if the
    input arrays contain large numbers of duplicated values (i.e. hundreds each),
    this will lead to large "dense" blocks in the sparse matrix. This will result in
    slow processing and high memory usage.

    Parameters
    ----------
    seqs
        Numpy array of nucleotide or amino acid sequences.
        Note that not all distance metrics support nucleotide sequences.
    seqs2
        Second array sequences. When omitted, `sequence_dist` computes
        the square matrix of `unique_seqs`.
    {metric}
    {cutoff}
    n_jobs
        Number of CPU cores to use when running a DistanceCalculator that supports
        paralellization.

        A cutoff of 0 implies the `identity` metric.
    kwargs
        Additional parameters passed to the :class:`~scirpy.ir_dist.metrics.DistanceCalculator`.

    Returns
    -------
    Symmetrical, sparse pairwise distance matrix.
    """
    seqs = [x.upper() for x in seqs]
    seqs_unique, seqs_unique_inverse = np.unique(seqs, return_inverse=True)  # type: ignore
    if seqs2 is not None:
        seqs2 = [x.upper() for x in seqs2]
        seqs2_unique, seqs2_unique_inverse = np.unique(seqs2, return_inverse=True)  # type: ignore
    else:
        seqs2_unique, seqs2_unique_inverse = None, seqs_unique_inverse

    dist_calc = _get_distance_calculator(metric, cutoff, n_jobs=n_jobs, **kwargs)

    logging.info(f"Calculating distances with metric {metric}")

    dist_mat = dist_calc.calc_dist_mat(seqs_unique, seqs2_unique)

    # Slicing with CSR is faster than with DOK
    dist_mat = dist_mat.tocsr()

    logging.hint("Expanding non-unique sequences to sequence x sequence matrix")
    i, j = np.meshgrid(
        seqs_unique_inverse, seqs2_unique_inverse, sparse=True, indexing="ij"
    )
    dist_mat = dist_mat[i, j]

    return dist_mat
