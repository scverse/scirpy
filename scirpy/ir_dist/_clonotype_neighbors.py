from typing import Mapping, Union, Sequence
from anndata import AnnData
from scanpy import logging
from scipy.sparse.csr import csr_matrix
from tqdm.std import tqdm
from .._compat import Literal
import numpy as np
import scipy.sparse as sp
import itertools
from ._util import DoubleLookupNeighborFinder, BoolSetMask, SetMask
from multiprocessing import Pool
from ..util import _is_na, _is_true
from functools import reduce
from operator import and_, or_
from tqdm.contrib import tmap
import pandas as pd


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
        self.same_v_gene = same_v_gene
        self.within_group = within_group
        self.receptor_arms = receptor_arms
        self.dual_ir = dual_ir
        self.distance_dict = adata.uns[distance_key]
        self.sequence_key = sequence_key

        # will be filled in self._prepare
        self.neighbor_finder = None  # instance of DoubleLookupNeighborFinder
        self.clonotypes = None  # pandas data frame with unique receptor configurations
        self.cell_indices = None  # a mapping row index from self.clonotypes -> obs name

        self._receptor_arm_cols = (
            ["VJ", "VDJ"]
            if self.receptor_arms in ["all", "any"]
            else [self.receptor_arms]
        )
        self._dual_ir_cols = ["1"] if self.dual_ir == "primary_only" else ["1", "2"]

        self._cdr3_cols, self._v_gene_cols = list(), list()
        for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols):
            self._cdr3_cols.append(f"IR_{arm}_{i}_{self.sequence_key}")
            if same_v_gene:
                self._v_gene_cols.append(f"IR_{arm}_{i}_v_gene")

        self._prepare(adata)

    def _prepare(self, adata: AnnData):
        """Initalize the DoubleLookupNeighborFinder and all required lookup tables"""
        start = logging.info("Initializing lookup tables. ")
        self._make_clonotype_table(adata)
        self.neighbor_finder = DoubleLookupNeighborFinder(self.clonotypes)
        self._add_distance_matrices(adata)
        self._add_lookup_tables()
        logging.hint("Done initializing lookup tables.", time=start)

    def _make_clonotype_table(self, adata):
        """Define 'preliminary' clonotypes based identical IR features. """
        if not adata.obs_names.is_unique:
            raise ValueError("Obs names need to be unique!")

        clonotype_cols = self._cdr3_cols + self._v_gene_cols
        if self.within_group is not None:
            clonotype_cols += list(self.within_group)

        obs_filtered = adata.obs.loc[lambda df: _is_true(df["has_ir"]), clonotype_cols]
        # make sure all nans are consistent "nan"
        # This workaround will be made obsolete by #190.
        for col in obs_filtered.columns:
            obs_filtered[col] = obs_filtered[col].astype(str)
            obs_filtered.loc[_is_na(obs_filtered[col]), col] = "nan"

        clonotype_groupby = obs_filtered.groupby(
            clonotype_cols, sort=False, observed=True
        )
        # This only gets the unique_values (the groupby index)
        clonotypes = clonotype_groupby.size().index.to_frame(index=False)

        if clonotypes.shape[0] == 0:
            raise ValueError(
                "Error computing clonotypes. "
                "No cells with IR information found (`adata.obs['has_ir'] == True`)"
            )

        # groupby.indices gets us a index -> array of row indices mapping.
        # It has the same order as `clonotypes` (derived from the same groupby) object.
        # Therefore, we can use `.values()` to obtain a clonotype_id -> cell_id mapping.
        self.cell_indices = [
            obs_filtered.index[idx].values for idx in clonotype_groupby.indices.values()
        ]

        # make 'within group' a single column of tuples (-> only one distance
        # matrix instead of one per column.)
        if self.within_group is not None:
            within_group_col = list(
                clonotypes.loc[:, self.within_group].itertuples(index=False, name=None)
            )
            for tmp_col in self.within_group:
                del clonotypes[tmp_col]
            clonotypes["within_group"] = within_group_col

        self.clonotypes = clonotypes

    def _add_distance_matrices(self, adata):
        """Add all required distance matrices to the DLNF"""
        # sequence distance matrices
        for chain_type in self._receptor_arm_cols:
            self.neighbor_finder.add_distance_matrix(
                name=chain_type,
                distance_matrix=self.distance_dict[chain_type]["distances"],
                labels=self.distance_dict[chain_type]["seqs"],
            )

        if self.same_v_gene:
            # V gene distance matrix (ID mat)
            v_genes = self._unique_values_in_multiple_columns(
                adata.obs, self._v_gene_cols
            )
            self.neighbor_finder.add_distance_matrix(
                "v_gene", sp.identity(len(v_genes), dtype=bool, format="csr"), v_genes  # type: ignore
            )

        if self.within_group is not None:
            within_group_values = np.unique(self.clonotypes["within_group"].values)
            self.neighbor_finder.add_distance_matrix(
                "within_group",
                sp.identity(len(within_group_values), dtype=bool, format="csr"),  # type: ignore
                within_group_values,
            )

    @staticmethod
    def _unique_values_in_multiple_columns(
        df: pd.DataFrame, columns: Sequence[str]
    ) -> np.ndarray:
        return np.unique(np.concatenate([df[c].values for c in columns]))  # type: ignore

    def _add_lookup_tables(self):
        """Add all required lookup tables to the DLNF"""
        for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols):
            self.neighbor_finder.add_lookup_table(
                f"{arm}_{i}", f"IR_{arm}_{i}_{self.sequence_key}", arm
            )
            if self.same_v_gene:
                self.neighbor_finder.add_lookup_table(
                    f"{arm}_{i}_v_gene", f"IR_{arm}_{i}_v_gene", "v_gene"
                )

        if self.within_group is not None:
            self.neighbor_finder.add_lookup_table(
                "within_group", "within_group", "within_group"
            )

    def compute_distances(self) -> sp.csr_matrix:
        """Compute the distances between clonotypes. `prepare` must have
        been ran previously. Returns a clonotype x clonotype sparse
        distance matrix."""
        start = logging.info("Computing clonotype x clonotype distances. ")
        n_clonotypes = self.clonotypes.shape[0]
        # TODO niceer progressbar
        # with Pool() as p:
        #     dist_rows = list(
        #         tqdm(
        #             p.imap(
        #                 self._dist_for_clonotype, range(n_clonotypes), chunksize=5000
        #             ),
        #             total=n_clonotypes,
        #         )
        #     )
        dist_rows = tmap(self._dist_for_clonotype, range(n_clonotypes))
        dist = sp.vstack(dist_rows)
        dist.eliminate_zeros()
        logging.hint("Done computing clonotype x clonotype distances. ", time=start)
        return dist  # type: ignore

    def _dist_for_clonotype(self, ct_id: int) -> sp.csr_matrix:
        """Compute neighboring clonotypes for a given clonotype.

        Or operations use the min dist of two matching entries.
        And operations use the max dist of two matchin entries.

        The motivation for using the max instead of sum/average is
        that our hypotheis is that a receptor recognizes the same antigen if it
        has a sequence dist < threshold. If we require both receptors to
        match ("and"), the higher one should count.
        """
        res = []
        for tmp_receptor_arm in self._receptor_arm_cols:

            def _lookup(tmp_chain1, tmp_chain2) -> SetMask:
                cdr3 = self.neighbor_finder.lookup(
                    ct_id,
                    f"{tmp_receptor_arm}_{tmp_chain1}",
                    f"{tmp_receptor_arm}_{tmp_chain2}",
                )
                if self.same_v_gene:
                    return cdr3 & self.neighbor_finder.lookup(
                        ct_id,
                        f"{tmp_receptor_arm}_{tmp_chain1}_v_gene",
                        f"{tmp_receptor_arm}_{tmp_chain2}_v_gene",
                    )
                else:
                    return cdr3

            if self.dual_ir == "primary_only":
                tmp_res = _lookup(1, 1)
            elif self.dual_ir == "all":
                tmp_res = (_lookup(1, 1) & _lookup(2, 2)) | (
                    _lookup(1, 2) & _lookup(2, 1)
                )
            else:  # "any"
                tmp_res = _lookup(1, 1) | _lookup(2, 2) | _lookup(1, 2) | _lookup(2, 1)

            res.append(tmp_res)

        operator = and_ if self.receptor_arms == "all" else or_
        res = reduce(operator, res)

        if self.within_group is not None:
            res = res & self.neighbor_finder.lookup(
                ct_id, "within_group", "within_group"
            )

        # if it's a bool set masks it corresponds to all nan
        if isinstance(res, BoolSetMask):
            return csr_matrix((1, len(res)), dtype=int)
        else:
            return res.data
