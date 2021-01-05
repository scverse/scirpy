from typing import Union, Tuple, Sequence, Iterable, Dict
from anndata import AnnData
from .._compat import Literal
import numpy as np
import scipy.sparse as sp
import itertools
import pandas as pd
from ._util import SetDict, DoubleLookupNeighborFinder


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
    sequence,
    metric,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    pass


class ClonotypeNeighbors:
    def __init__(
        self,
        adata,
        *,
        same_v_gene,
        within_group,
        receptor_arms,
        dual_ir,
        sequence,
        metric,
    ):
        self.key = "cdr3" if sequence == "aa" else "cdr3_nt"

        # store sequence distances
        for chain_type in ["VJ", "VDJ"]:
            distance_dict = adata.uns[f"ir_dist_{sequence}_{metric}"]
            self.feature_distances[chain_type] = distance_dict["distances"]
            self.feature_indexes[chain_type] = {
                k: i for i, k in enumerate(distance_dict["seqs"])
            }

        # store v gene distances
        v_genes = np.unique(
            np.concatenate(
                [
                    adata.obs[c].values
                    for c in [
                        "IR_VJ_1_v_gene",
                        "IR_VJ_2_v_gene",
                        "IR_VDJ_1_v_gene",
                        "IR_VDJ_2_v_gene",
                    ]
                ]
            )
        )
        self.feature_distances["v_gene"] = sp.identity(
            len(v_genes), dtype=bool, format="csr"
        )
        self.feature_indexes["v_gene"] = {k: i for i, k in enumerate(v_genes)}

        # Define clonotypes. TODO v-genes, within_group
        receptor_arm_cols = (
            ["VJ", "VDJ"] if receptor_arms in ["all", "any"] else [receptor_arms]
        )
        dual_ir_cols = ["1"] if dual_ir == "primary_only" else ["1", "2"]
        clonotype_cols = [
            f"IR_{arm}_{i}_{self.key}"
            for arm, i in itertools.product(receptor_arm_cols, dual_ir_cols)
        ]

        self.clonotypes = (
            adata.obs.loc[clonotype_cols, :].drop_duplicates().reset_index()
        )

        self._build_lookup_table("IR_VJ_1_cdr3", "VJ", name="VJ_1")
        self._build_lookup_table("IR_VDJ_1_cdr3", "VDJ", name="VDJ_1")
        self._build_lookup_table("IR_VJ_2_cdr3", "VJ", name="VJ_2")
        self._build_lookup_table("IR_VDJ_2_cdr3", "VDJ", name="VDJ_2")
        self._build_lookup_table("IR_VJ_1_v_gene", "v_gene", name="VJ_v")
        self._build_lookup_table("IR_VDJ_1_v_gene", "v_gene", name="VDJ_v")

    def compute_distances(self):
        for i, ct in self.clonotypes.itertuples():
            neighbors_vj1 = self.reverse_lookups["VJ_1"][self.lookups["VJ_1"][i]]
            neighbors_vj1_2 = self.reverse_lookups["VJ_2"][self.lookups["VJ_1"][i]]


#
#
#
#
# def _expand_and_reduce_distances(adata, *receptor_arms, dual_ir, sequence, metric):
#     """
#     Expand the sequence distances to a matrix of distances between clonotypes.

#     # TODO include v_gene et al into the clonotype definition.
#     # I believe this could be handled as a matrix-vector product.

#     Returns a vector of unique clonotypes and a sparse distance matrix of the
#     same dimension.
#     """
#     if receptor_arms not in ["VJ", "VDJ", "all", "any"]:
#         raise ValueError("Invalid value for receptor_arms!")
#     if dual_ir not in ["primary_only", "all", "any"]:
#         raise ValueError("Invalid value for dual_ir!")

#     seq_dist = adata.uns[f"ir_dist_{sequence}_{metric}"]
#     key = "cdr3" if sequence == "aa" else "cdr3_nt"
#     use_receptor_arms = (
#         ["VJ", "VDJ"] if receptor_arms in ["all", "any"] else [receptor_arms]
#     )

#     receptor_arm_dists = {
#         arm: _reduce_chains(
#             adata, seq_dist[arm]["distances"], seq_dist[arm]["seqs"], dual_ir, arm, key
#         )
#         for arm in use_receptor_arms
#     }

#     return _reduce_arms(adata, receptor_arm_dists, receptor_arms, key)


# def _reduce_chains(adata, dist_mat, index, dual_ir, receptor_arm, key):
#     if dual_ir == "primary_only":
#         seqs = np.unique(adata.obs[f"IR_{receptor_arm}_1_{key}"])
#         seqs = seqs[~_is_na(seqs)]
#         return seqs, _expand_matrix_to_target(dist_mat, index, seqs)
#     else:
#         target_tuples = np.unique(
#             [
#                 (pri, sec)
#                 for pri, sec in zip(
#                     adata.obs[f"IR_{receptor_arm}_1_{key}"],
#                     adata.obs[f"IR_{receptor_arm}_2_{key}"],
#                 )
#                 if not _is_na(pri)
#             ]
#         )
#         m1 = _expand_matrix_to_target(dist_mat, index, [x[0] for x in target_tuples])
#         m2 = _expand_matrix_to_target(dist_mat, index, [x[1] for x in target_tuples])
#         if dual_ir == "all":
#             return target_tuples, m1.maximum(m2).multiply(m1 > 0).multiply(m2 > 0)
#         else:
#             return target_tuples, _reduce_nonzero(m1, m2)


# def _reduce_arms(adata, receptor_arm_dists, receptor_arms, key):
#     if receptor_arms in ["VJ", "VDJ"]:
#         return receptor_arm_dists[receptor_arms]
#     else:
#         target_tuples = np.unique(
#             [
#                 ((vj_pri, vj_sec), (vdj_pri, vdj_sec))
#                 for vj_pri, vj_sec, vdj_pri, vdj_sec in zip(
#                     adata.obs[f"IR_VJ_1_{key}"],
#                     adata.obs[f"IR_VJ_2_{key}"],
#                     adata.obs[f"IR_VDJ_1_{key}"],
#                     adata.obs[f"IR_VDJ_2_{key}"],
#                 )
#                 if not (_is_na(vj_pri) and _is_na(vdj_pri))
#             ]
#         )
#         index_vj, dist_vj = receptor_arm_dists["VJ"]
#         index_vdj, dist_vdj = receptor_arm_dists["VDJ"]
#         m1 = _expand_matrix_to_target(dist_vj, index_vj, [x[0] for x in target_tuples])
#         m2 = _expand_matrix_to_target(
#             dist_vdj, index_vdj, [x[1] for x in target_tuples]
#         )
#         if receptor_arms == "all":
#             return target_tuples, m1.maximum(m2).multiply(m1 > 0).multiply(m2 > 0)
#         else:
#             return target_tuples, _reduce_nonzero(m1, m2)


# # def _reduce_seqs(
# #     dist_mat1: spmatrix,
# #     dist_mat2: spmatrix,
# #     index1: Sequence,
# #     index2: Sequence,
# #     target_tuples: Sequence[Tuple],
# #     operation: Literal["and", "or"],
# # ) -> csr_matrix:
# #     """Expands and merges two distance matrices

# #     Parameters
# #     ----------
# #     dist_mat1, dist_mat2
# #         square (upper triangular) sparse adjacency matrix
# #     index1, index2
# #         index for the dist matrices.
# #         `len(index) == dist_mat.shape[0] == dist_mat.shape[1]`
# #     target_tuples
# #         Contains tuples with entries from (index1, index2). The function
# #         computes the distances between the tuples based on the input distance
# #         matrix and the operation
# #     operation
# #         If `and`, both entries in the target tuple must be connected in their respective
# #         distance matrices. If `or`, at least one of the entries in the target tuple
# #         must be connected.

# #     Returns
# #     -------
# #     Square upper triangular sparse distance matrix with `ncol = nrow = len(target)`.
# #     """
# #     m1 = _expand_matrix_to_target(dist_mat1, index1, [x[0] for x in target_tuples])
# #     m2 = _expand_matrix_to_target(dist_mat2, index2, [x[1] for x in target_tuples])

# #     if operation == "and":
# #         res = result_dist_mat == 2
# #     else:
# #         res = result_dist_mat >= 1
# #     return res  # type: ignore


# def _expand_matrix_to_target(
#     dist_mat: spmatrix, index: Iterable, target: Sequence
# ) -> spmatrix:
#     """Subset a square matrix with rows and columns `index` such that its new rows and
#     columns match `target`"""
#     index_dict = {idx: i for i, idx in enumerate(index)}
#     target_idx = np.fromiter(
#         (index_dict[t] for t in target), dtype=int, count=len(target)
#     )
#     i, j = np.meshgrid(target_idx, target_idx, sparse=True, indexing="ij")

#     dist_mat = dist_mat[i, j]
#     return dist_mat
