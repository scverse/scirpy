from typing import Mapping, Union, Sequence
from anndata import AnnData
from .._compat import Literal
import numpy as np
import scipy.sparse as sp
import itertools
from ._util import SetDict, DoubleLookupNeighborFinder
from ..util import _is_na, _is_true


class ClonotypeNeighbors:
    def __init__(
        self,
        adata: AnnData,
        *,
        receptor_arms=Literal["VJ", "VDJ", "all", "any"],
        dual_ir=Literal["primary_only", "all", "any"],
        same_v_gene: bool = False,
        within_group: Union[None, Sequence[str]] = None,
        distance_key: str,
        sequence_key: str,
    ):
        """Compute distances between clonotypes"""
        self.adata = adata
        self.same_v_gene = same_v_gene
        self.within_group = within_group
        self.receptor_arms = receptor_arms
        self.dual_ir = dual_ir
        self.distance_dict = adata.uns[distance_key]
        self.sequence_key = sequence_key

        self._receptor_arm_cols = (
            ["VJ", "VDJ"]
            if self.receptor_arms in ["all", "any"]
            else [self.receptor_arms]
        )
        self._dual_ir_cols = ["1"] if self.dual_ir == "primary_only" else ["1", "2"]

        self._prepare()

    def _prepare(self):
        """Initalize the DoubleLookupNeighborFinder and all required lookup tables"""
        self._make_clonotype_table()
        self.neighbor_finder = DoubleLookupNeighborFinder(self.clonotypes)
        self._add_distance_matrices()
        self._add_lookup_tables()

    def _make_clonotype_table(self):
        """Define clonotypes based identical IR features"""
        # Define clonotypes. TODO v-genes, within_group
        clonotype_cols = [
            f"IR_{arm}_{i}_{self.sequence_key}"
            for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols)
        ]
        clonotypes = (
            self.adata.obs.loc[_is_true(self.adata.obs["has_ir"]), clonotype_cols]
            .drop_duplicates()
            .reset_index()
        )

        # make sure all nans are consistent "nan"
        # This workaround will be made obsolete by #190.
        for col in clonotypes.columns:
            clonotypes.loc[_is_na(clonotypes[col]), col] = "nan"
        self.clonotypes = clonotypes

    def _add_distance_matrices(self):
        """Add all required distance matrices to the DLNF"""
        for chain_type in self._receptor_arm_cols:
            self.neighbor_finder.add_distance_matrix(
                name=chain_type,
                distance_matrix=self.distance_dict[chain_type]["distances"],
                labels=self.distance_dict[chain_type]["seqs"],
            )

        # # store v gene distances
        # v_genes = np.unique(
        #     np.concatenate(
        #         [
        #             self.adata.obs[c].values
        #             for c in [
        #                 "IR_VJ_1_v_gene",
        #                 "IR_VJ_2_v_gene",
        #                 "IR_VDJ_1_v_gene",
        #                 "IR_VDJ_2_v_gene",
        #             ]
        #         ]
        #     )
        # )
        # self.neighbor_finder.add_distance_matrix(
        #     "v_gene", sp.identity(len(v_genes), dtype=bool, format="csr"), v_genes  # type: ignore
        # )

    def _add_lookup_tables(self):
        """Add all required lookup tables to the DLNF"""
        for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols):
            self.neighbor_finder.add_lookup_table(
                f"{arm}_{i}", f"IR_{arm}_{i}_{self.sequence_key}", arm
            )

        # self.neighbor_finder.add_lookup_table("VJ_v", "IR_VJ_1_v_gene", "v_gene")
        # self.neighbor_finder.add_lookup_table("VDJ_v", "IR_VDJ_1_v_gene", "v_gene")

    def compute_distances(self) -> sp.csr_matrix:
        """Compute the distances between clonotypes. `prepare` must have
        been ran previously. Returns a clonotype x clonotype sparse
        distance matrix."""
        n_clonotypes = self.clonotypes.shape[0]
        dist_rows = map(self._dist_for_clonotype, range(n_clonotypes))
        return sp.vstack(dist_rows)  # type: ignore

    def _dist_for_clonotype(self, ct_id: int) -> sp.csr_matrix:
        """Compute neighboring clonotypes for a given clonotype"""
        if self.receptor_arms == "VJ":
            if self.dual_ir == "primary_only":
                res = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_1"))
            if self.dual_ir == "all":
                r1 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_1"))
                r2 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_2"))
                r1_2 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_1", "VJ_2"))
                r2_1 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_2", "VJ_1"))
                res = (r1 & r2) | (r1_2 & r2_1)
            if self.dual_ir == "any":
                r1 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_1"))
                r2 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_2"))
                r1_2 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_1", "VJ_2"))
                r2_1 = SetDict(self.neighbor_finder.lookup(ct_id, "VJ_2", "VJ_1"))
                res = r1 | r2 | r1_2 | r2_1
            row = self._dict_to_sparse_row(res, self.clonotypes.shape[0])
            return row

    @staticmethod
    def _dict_to_sparse_row(row_dict: Mapping, row_len: int) -> sp.csr_matrix:
        """Efficient way of converting a SetDict to a 1 x n sparse row in CSR format"""
        sparse_row = sp.csr_matrix((1, row_len))
        sparse_row.data = np.fromiter(row_dict.values(), int, len(row_dict))
        sparse_row.indices = np.fromiter(row_dict.keys(), int, len(row_dict))
        sparse_row.indptr = np.array([0, len(row_dict)])
        return sparse_row
