from typing import Union, Tuple, Sequence, Iterable
from anndata import AnnData
from scipy.sparse.csr import csr_matrix
from .._compat import Literal
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
from ..util.graph import igraph_from_sparse_matrix
import itertools


def define_clonotype_clusters(
    adata: AnnData,
    *,
    same_v_gene: Union[bool, Literal["primary_only", "all"]] = False,
    within_group: Union[str, None] = "receptor_type",
    partitions: Literal["connected", "leiden"] = "connected",
    receptor_arms=Literal["VJ", "VDJ", "all", "any"],
    dual_ir=Literal["primary_only", "all", "any"],
    resolution: float = 1,
    n_iterations: int = 5,
    neighbors_key: str = "ir_neighbors",
    key_added: str = "clonotype",
    inplace: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    seq_dist = adata.uns[f"ir_dist_{sequence}_{metric}"]


def _reduce_chains():
    pass


def _reduce_arms():
    pass


def _reduce_seqs(
    dist_mat1: spmatrix,
    dist_mat2: spmatrix,
    index1: Sequence,
    index2: Sequence,
    target_tuples: Sequence[Tuple],
    operation: Literal["and", "or"],
) -> csr_matrix:
    """Expands and merges two distance matrices

    Parameters
    ----------
    dist_mat1, dist_mat2
        square (upper triangular) sparse adjacency matrix
    index1, index2
        index for the dist matrices.
        `len(index) == dist_mat.shape[0] == dist_mat.shape[1]`
    target_tuples
        Contains tuples with entries from (index1, index2). The function
        computes the distances between the tuples based on the input distance
        matrix and the operation
    operation
        If `and`, both entries in the target tuple must be connected in their respective
        distance matrices. If `or`, at least one of the entries in the target tuple
        must be connected.

    Returns
    -------
    Square upper triangular sparse distance matrix with `ncol = nrow = len(target)`.
    """
    m1 = _expand_matrix_to_target(dist_mat1, index1, [x[0] for x in target_tuples])
    m2 = _expand_matrix_to_target(dist_mat2, index2, [x[1] for x in target_tuples])

    if operation == "and":
        res = result_dist_mat == 2
    else:
        res = result_dist_mat >= 1
    return res  # type: ignore


def _expand_matrix_to_target(
    dist_mat: spmatrix, index: Iterable, target: Sequence
) -> spmatrix:
    index_dict = {idx: i for i, idx in enumerate(index)}
    target_idx = np.fromiter(
        (index_dict[t] for t in target), dtype=int, count=len(target)
    )

    dist_mat = dist_mat.todok()
    dist_mat = dist_mat[target_idx, target_idx.T]
    return dist_mat
