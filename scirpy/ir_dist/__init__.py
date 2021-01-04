import itertools
from anndata import AnnData
from typing import Union, List, Tuple, Dict
from .._compat import Literal
import numpy as np
from scanpy import logging
from ..util import _is_na, deprecated
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse
from ..util import _doc_params
from tqdm import tqdm
from . import metrics


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `sequence_dist`. The old version will be removed in a future release. "
)
def tcr_dist(*args, **kwargs):
    return sequence_dist(*args, **kwargs)


# TODO: deprecated / removed exceptions for all changed public API items.
def TcrNeighbors(*args, **kwargs):
    raise NotImplementedError(
        "TcrNeighbors has been renamed in v0.5.0 and removed in v0.7.0"
    )


def IrNeighbors(*args, **kwargs):
    raise NotImplementedError("IrNeighbors has been removed in v0.7.0")


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


def _get_metric(metric: MetricType, cutoff: Union[int, None], *, n_jobs=None, **kwargs):
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


def ir_dist(
    adata: AnnData,
    *,
    metric: MetricType = "identity",
    cutoff: Union[int, None] = None,
    key_added: Union[str, None] = None,
    sequence: Literal["aa", "nt"] = "nt",
    inplace: bool = True,
    n_jobs: Union[int, None] = None,
) -> Union[dict, None]:
    COLUMN = "IR_{chain_type}_{chain_id}_{key}"
    key = "cdr3" if sequence == "aa" else "cdr3_nt"
    result = {
        "VJ": dict(),
        "VDJ": dict(),
        "params": {"metric": metric, "sequence": sequence, "cutoff": cutoff},
    }
    dist_calc = _get_metric(metric, cutoff, n_jobs=n_jobs)

    # get all unique seqs for VJ and VDJ
    for chain_type in ["VJ", "VDJ"]:
        tmp_seqs = np.concatenate(
            [
                np.unique(
                    adata.obs[
                        COLUMN.format(chain_type=chain_type, chain_id=chain_id, key=key)
                    ]
                )
                for chain_id in ["1", "2"]
            ]
        )
        result[chain_type]["seqs"] = tmp_seqs[~_is_na(tmp_seqs)]  # type: ignore

    # compute distance matrices
    for chain_type in ["VJ", "VDJ"]:
        logging.info(
            f"Computing sequence x sequence distance matrix for {chain_type} sequences."
        )
        tmp_seqs = result[chain_type]["seqs"]
        result[chain_type]["distances"] = dist_calc.calc_dist_mat(tmp_seqs).tocsr()

    # return or store results
    if inplace:
        if key_added is None:
            tmp_metric = (
                "custom" if isinstance(metric, metrics.DistanceCalculator) else metric
            )
            key_added = f"ir_dist_{sequence}_{tmp_metric}"
        adata.uns[key_added] = result
    else:
        return result


@_doc_params(metric=_doc_metrics, cutoff=_doc_cutoff, dist_mat=metrics._doc_dist_mat)
def sequence_dist(
    seqs: np.ndarray,
    seqs2: Union[None, np.ndarray] = None,
    *,
    metric: MetricType = "identity",
    cutoff: Union[None, int] = None,
    n_jobs: Union[None, int] = None,
    **kwargs,
) -> csr_matrix:
    """\
    Calculate a sequence x sequence distance matrix.

    {dist_mat}

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
    **kwargs
        Additional parameters passed to the :class:`~scirpy.ir_dist.metrics.DistanceCalculator`.

    When `seqs` or `seqs2` includes non-unique values, the function internally
    uses only unique sequences to calculate the distances. Note that, if the
    input arrays contain large numbers of duplicated values (i.e. hundreds each),
    this will lead to large "dense" blocks in the sparse matrix. This will result in
    slow processing and high memory usage.

    Returns
    -------
    Upper triangular distance matrix when only `seqs` is provided.
    A full rectangular distance matrix when both `seqs` and
    `seqs2` are provided.
    """
    seqs_unique, seqs_unique_inverse = np.unique(seqs, return_inverse=True)
    seqs2_unique, seqs2_unique_inverse = (
        np.unique(seqs2, return_inverse=True)
        if seqs2 is not None
        else (None, seqs_unique_inverse)
    )
    dist_calc = _get_metric(metric, cutoff, n_jobs=n_jobs, **kwargs)

    logging.info(f"Calculating distances with metric {metric}")

    dist_mat = dist_calc.calc_dist_mat(seqs_unique, seqs2_unique)

    # DOK format for slicing with meshgrid (COO does not support subscript)
    dist_mat = dist_mat.todok()

    if seqs2 is None:
        # mirror at diagonal, otherwise expansion from non-unique values
        # does not work properly.
        # The matrix is already upper diagonal. Use the transpose method, see
        # https://stackoverflow.com/a/58806735/2340703.
        logging.hint("Mirroring matrix at diagonal")
        dist_mat = dist_mat + dist_mat.T - scipy.sparse.diags(dist_mat.diagonal())

    logging.hint("Expanding non-unique sequences to sequence x sequence matrix")
    i, j = np.meshgrid(
        seqs_unique_inverse, seqs2_unique_inverse, sparse=True, indexing="ij"
    )
    dist_mat = dist_mat[i, j]

    if seqs2 is None:
        # If seq x seq, only return the upper triangle. Values in the lower
        # triangle may have been introduced by expanding the non-unique values.
        logging.hint("Converting to upper triangular CSR matrix")
        return scipy.sparse.triu(dist_mat, k=0, format="csr")  # type: ignore
    else:
        logging.hint("Converting to CSR matrix")
        return dist_mat.tocsr()  # type: ignore
