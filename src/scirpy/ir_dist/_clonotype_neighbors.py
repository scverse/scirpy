import itertools
from collections.abc import Mapping, Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scanpy import logging

from scirpy.get import _has_ir
from scirpy.get import airr as get_airr
from scirpy.util import DataHandler

from ._util import DoubleLookupNeighborFinder


class ClonotypeNeighbors:
    def __init__(
        self,
        params: DataHandler,
        params2: Optional[DataHandler] = None,
        *,
        receptor_arms: Literal["VJ", "VDJ", "all", "any"],
        dual_ir: Literal["primary_only", "all", "any"],
        same_v_gene: bool = False,
        same_j_gene: bool = False,
        match_columns: Union[None, Sequence[str]] = None,
        distance_key: str,
        sequence_key: str,
        n_jobs: int = -1,
        chunksize: int = 2000,
    ):
        """Computes pairwise distances between cells with identical
        receptor configuration and calls clonotypes from this distance matrix
        """
        self.same_v_gene = same_v_gene
        self.same_j_gene = same_j_gene
        self.match_columns = match_columns
        self.receptor_arms = receptor_arms
        self.dual_ir = dual_ir
        self.distance_dict = params.adata.uns[distance_key]
        self.sequence_key = sequence_key
        self.n_jobs = n_jobs
        self.chunksize = chunksize

        self._receptor_arm_cols = ["VJ", "VDJ"] if self.receptor_arms in ["all", "any"] else [self.receptor_arms]
        self._dual_ir_cols = ["1"] if self.dual_ir == "primary_only" else ["1", "2"]

        # Initialize the DoubleLookupNeighborFinder and all lookup tables
        start = logging.info("Initializing lookup tables. ")  # type: ignore

        self.cell_indices, self.clonotypes = self._make_clonotype_table(params)
        self._chain_count = self._make_chain_count(self.clonotypes)
        if params2 is not None:
            self.cell_indices2, self.clonotypes2 = self._make_clonotype_table(params2)
            self._chain_count2 = self._make_chain_count(self.clonotypes2)
        else:
            self.cell_indices2, self.clonotypes2, self._chain_count2 = None, None, None

        self.neighbor_finder = DoubleLookupNeighborFinder(self.clonotypes, self.clonotypes2)
        self._add_distance_matrices()
        self._add_lookup_tables()
        logging.hint("Done initializing lookup tables.", time=start)  # type: ignore

    def _make_clonotype_table(self, params: DataHandler) -> tuple[Mapping, pd.DataFrame]:
        """Define 'preliminary' clonotypes based identical IR features."""
        if not params.adata.obs_names.is_unique:
            raise ValueError("Obs names need to be unique!")

        airr_variables = [self.sequence_key]
        
        if self.same_v_gene:
            airr_variables.append("v_call")

        if self.same_j_gene:
            airr_variables.append("j_call")

        chains = [f"{arm}_{chain}" for arm, chain in itertools.product(self._receptor_arm_cols, self._dual_ir_cols)]

        obs = get_airr(params, airr_variables, chains)
        # remove entries without receptor (e.g. only non-productive chains) or no sequences
        obs = obs.loc[_has_ir(params) & np.any(~pd.isnull(obs), axis=1), :]
        if self.match_columns is not None:
            obs = obs.join(
                params.get_obs(self.match_columns),
                validate="one_to_one",
                how="inner",
            )

        # Converting nans to str("nan"), as we want string dtype
        for col in obs.columns:
            if obs[col].dtype == "category":
                obs[col] = obs[col].astype(str)
            obs.loc[pd.isnull(obs[col]), col] = "nan"  # type: ignore
            obs[col] = obs[col].astype(str)  # type: ignore

        # using groupby instead of drop_duplicates since we need the group indices below
        clonotype_groupby = obs.groupby(obs.columns.tolist(), sort=False, observed=True)
        # This only gets the unique_values (the groupby index)
        clonotypes = clonotype_groupby.size().index.to_frame(index=False)

        if clonotypes.shape[0] == 0:
            raise ValueError(
                "Error computing clonotypes. "
                "No cells with IR information found (adata.obsm['chain_indices'] is None for all cells)"
            )

        # groupby.indices gets us a (index -> array of row indices) mapping.
        # It doesn't necessarily have the same order as `clonotypes`.
        # This needs to be a dict of arrays, otherwiswe anndata
        # can't save it to h5ad.
        # Also the dict keys need to be of type `str`, or they'll get converted
        # implicitly.
        cell_indices = {
            str(i): obs.index[
                clonotype_groupby.indices.get(
                    # indices is not a tuple if it's just a single column.
                    ct_tuple[0] if len(ct_tuple) == 1 else ct_tuple,
                    [],
                )
            ].values
            for i, ct_tuple in enumerate(clonotypes.itertuples(index=False, name=None))
        }

        # make 'within group' a single column of tuples (-> only one distance
        # matrix instead of one per column.)
        if self.match_columns is not None:
            match_columns_col = list(clonotypes.loc[:, self.match_columns].itertuples(index=False, name=None))
            for tmp_col in self.match_columns:
                del clonotypes[tmp_col]
            clonotypes["match_columns"] = match_columns_col

        # consistency check: there must not be a secondary chain if there is no
        # primary one:
        if "2" in self._dual_ir_cols:
            for tmp_arm in self._receptor_arm_cols:
                primary_is_nan = clonotypes[f"{tmp_arm}_1_{self.sequence_key}"] == "nan"
                secondary_is_nan = clonotypes[f"{tmp_arm}_2_{self.sequence_key}"] == "nan"
                assert not np.sum(
                    ~secondary_is_nan[primary_is_nan]
                ), "There must not be a secondary chain if there is no primary one"

        return cell_indices, clonotypes

    def _add_distance_matrices(self) -> None:
        """Add all required distance matrices to the DoubleLookupNeighborFinder"""
        # sequence distance matrices
        for chain_type in self._receptor_arm_cols:
            self.neighbor_finder.add_distance_matrix(
                name=chain_type,
                distance_matrix=self.distance_dict[chain_type]["distances"],
                labels=self.distance_dict[chain_type]["seqs"],
                labels2=self.distance_dict[chain_type].get("seqs2", None),
            )

        if self.same_v_gene:
            # V gene distance matrix (identity mat)
            v_genes = self._unique_values_in_multiple_columns(
                self.clonotypes, [x for x in self.clonotypes.columns if "v_call" in x]
            )
            if self.clonotypes2 is not None:
                v_genes |= self._unique_values_in_multiple_columns(
                    self.clonotypes2,
                    [x for x in self.clonotypes.columns if "v_call" in x],
                )
            self.neighbor_finder.add_distance_matrix(
                "v_gene",
                sp.identity(len(v_genes), dtype=bool, format="csr"),
                v_genes,  # type: ignore
            )

        if self.same_j_gene:
            # J gene distance matrix (identity mat)
            j_genes = self._unique_values_in_multiple_columns(
                self.clonotypes, [x for x in self.clonotypes.columns if "j_call" in x]
            )
            if self.clonotypes2 is not None:
                j_genes |= self._unique_values_in_multiple_columns(
                    self.clonotypes2,
                    [x for x in self.clonotypes.columns if "j_call" in x],
                )
            self.neighbor_finder.add_distance_matrix(
                "j_gene",
                sp.identity(len(j_genes), dtype=bool, format="csr"),
                j_genes,  # type: ignore
            )

        if self.match_columns is not None:
            match_columns_values = set(self.clonotypes["match_columns"].values)
            if self.clonotypes2 is not None:
                match_columns_values |= set(self.clonotypes2["match_columns"].values)
            self.neighbor_finder.add_distance_matrix(
                "match_columns",
                sp.identity(len(match_columns_values), dtype=bool, format="csr"),  # type: ignore
                list(match_columns_values),
            )

    @staticmethod
    def _unique_values_in_multiple_columns(df: pd.DataFrame, columns: Sequence[str]) -> set:
        """Return the Union of unique values of multiple columns of a dataframe"""
        return set(np.concatenate([df[c].values.astype(str) for c in columns]))

    def _add_lookup_tables(self) -> None:
        """Add all required lookup tables to the DoubleLookupNeighborFinder"""
        for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols):
            self.neighbor_finder.add_lookup_table(f"{arm}_{i}", f"{arm}_{i}_{self.sequence_key}", arm)
            if self.same_v_gene:
                self.neighbor_finder.add_lookup_table(
                    f"{arm}_{i}_v_call",
                    f"{arm}_{i}_v_call",
                    "v_gene",
                    dist_type="boolean",
                )
            if self.same_j_gene:
                self.neighbor_finder.add_lookup_table(
                    f"{arm}_{i}_j_call",
                    f"{arm}_{i}_j_call",
                    "j_gene",
                    dist_type="boolean",
                )

        if self.match_columns is not None:
            self.neighbor_finder.add_lookup_table(
                "match_columns", "match_columns", "match_columns", dist_type="boolean"
            )

    def _make_chain_count(self, clonotype_table) -> dict[str, int]:
        """Compute how many chains there are of each type."""
        cols = {arm: [f"{arm}_{c}_{self.sequence_key}" for c in self._dual_ir_cols] for arm in self._receptor_arm_cols}
        cols["arms"] = [f"{arm}_1_{self.sequence_key}" for arm in self._receptor_arm_cols]
        return {step: np.sum(clonotype_table.loc[:, cols].values != "nan", axis=1) for step, cols in cols.items()}

    def compute_distances(self) -> sp.csr_matrix:
        start = logging.info("Computing clonotype x clonotype distances.")  # type: ignore
        n_clonotypes = self.clonotypes.shape[0]
        clonotype_ids = np.arange(n_clonotypes)

        dist = self._dist_for_clonotype(clonotype_ids)

        dist.eliminate_zeros()
        logging.hint("Done computing clonotype x clonotype distances. ", time=start)
        return dist  # type: ignore

    def _dist_for_clonotype(self, ct_ids: np.ndarray[int]) -> sp.csr_matrix:
        lookup = {}
        chain_ids = [(1, 1)] if self.dual_ir == "primary_only" else [(1, 1), (2, 2), (1, 2), (2, 1)]
        for tmp_arm in self._receptor_arm_cols:
            for c1, c2 in chain_ids:
                lookup[(tmp_arm, c1, c2)] = self.neighbor_finder.lookup(
                    ct_ids,
                    f"{tmp_arm}_{c1}",
                    f"{tmp_arm}_{c2}",
                )
        id_len = len(ct_ids)

        first_value = next(iter(lookup.values()))
        has_distance_table = sp.csr_matrix((id_len, first_value.shape[1]))
        for value in lookup.values():
            has_distance_table += value

        has_distance_mask = has_distance_table
        has_distance_mask.data = np.ones_like(has_distance_mask.data)

        def csr_min(a, b):
            max_value_a = np.max(a.data, initial=0)
            max_value_b = np.max(b.data, initial=0)
            max_value = np.int16(np.max([max_value_a, max_value_b]) + 1)
            min_mat_a = sp.csr_matrix((a.data.astype(np.int16), a.indices, a.indptr), shape=a.shape)
            min_mat_a.data -= max_value
            min_mat_b = sp.csr_matrix((b.data.astype(np.int16), b.indices, b.indptr), shape=b.shape)
            min_mat_b.data -= max_value
            a_smaller_b = min_mat_a < min_mat_b
            return b + (a - b).multiply(a_smaller_b)

        def csr_max(a, b):
            max_value_a = np.max(a.data, initial=0)
            max_value_b = np.max(b.data, initial=0)
            max_value = np.int16(np.max([max_value_a, max_value_b]) + 1)
            max_mat_a = sp.csr_matrix((a.data.astype(np.int16), a.indices, a.indptr), shape=a.shape)
            max_mat_a.data -= max_value
            max_mat_b = sp.csr_matrix((b.data.astype(np.int16), b.indices, b.indptr), shape=b.shape)
            max_mat_b.data -= max_value
            a_greater_b = max_mat_a > max_mat_b
            return b + (a - b).multiply(a_greater_b)

        if self.match_columns is not None:
            distance_matrix_name, forward, _ = self.neighbor_finder.lookups["match_columns"]
            distance_matrix_name_reverse, _, reverse = self.neighbor_finder.lookups["match_columns"]
            if distance_matrix_name != distance_matrix_name_reverse:
                raise ValueError("Forward and reverse lookup tablese must be defined " "on the same distance matrices.")
            reverse_lookup_values = np.vstack(list(reverse.lookup.values()))
            reverse_lookup_keys = np.zeros(reverse.size, dtype=np.int32)
            reverse_lookup_keys[list(reverse.lookup.keys())] = np.arange(len(list(reverse.lookup.keys())))
            match_column_mask = sp.csr_matrix(
                (np.empty(len(has_distance_mask.indices)), has_distance_mask.indices, has_distance_mask.indptr),
                shape=has_distance_mask.shape,
            )
            has_distance_mask_coo = match_column_mask.tocoo()
            indices_in_dist_mat = forward[has_distance_mask_coo.row]
            match_column_mask.data = reverse_lookup_values[
                reverse_lookup_keys[indices_in_dist_mat], has_distance_mask_coo.col
            ]

        tmp_arm_res = {}
        chain_res = {}

        def filter_chain_count_data(
            matrix_coo_data_chain_filtered,
            matrix_coo_data,
            matrix_coo_row,
            matrix_coo_col,
            chain_count_array1,
            chain_count_array2,
        ):
            data_indices = np.arange(len(matrix_coo_data))
            chain_counts1 = chain_count_array1[matrix_coo_row]
            chain_counts2 = chain_count_array2[matrix_coo_col]
            chain_counts_equal = chain_counts1 == chain_counts2
            matrix_coo_data_chain_filtered[chain_counts1[chain_counts_equal], data_indices[chain_counts_equal]] = (
                matrix_coo_data[chain_counts_equal]
            )
            return (
                matrix_coo_data_chain_filtered[0],
                matrix_coo_data_chain_filtered[1],
                matrix_coo_data_chain_filtered[2],
            )

        def filter_chain_count(matrix: sp.csr_matrix, col: str) -> sp.csr_matrix:
            chain_count_array1 = self._chain_count[col]

            if self._chain_count2 is None:
                chaint_count_array2 = chain_count_array1
            else:
                chaint_count_array2 = self._chain_count2[col]

            matrix_coo = matrix.tocoo()
            matrix_coo_data_chain_filtered = np.array(
                [np.zeros_like(matrix_coo.data), np.zeros_like(matrix_coo.data), np.zeros_like(matrix_coo.data)]
            )
            csr_filtered1, csr_filtered2, csr_filtered3 = matrix.copy(), matrix.copy(), matrix.copy()
            csr_filtered1.data, csr_filtered2.data, csr_filtered3.data = filter_chain_count_data(
                matrix_coo_data_chain_filtered,
                matrix_coo.data,
                matrix_coo.row,
                matrix_coo.col,
                chain_count_array1,
                chaint_count_array2,
            )
            return csr_filtered1, csr_filtered2, csr_filtered3

        for tmp_arm in self._receptor_arm_cols:
            for c1, c2 in chain_ids:
                tmp_arrays = lookup[(tmp_arm, c1, c2)][ct_ids]

                if not (self.same_v_gene or self.same_j_gene or self.match_columns):
                    tmp_arrays = tmp_arrays.multiply(has_distance_mask)

                def match_gene_segment(tmp_arrays: sp.csr_matrix, segment_suffix: str = "v_call") -> sp.csr_matrix:
                    distance_matrix_name, forward, _ = self.neighbor_finder.lookups[f"{tmp_arm}_{c1}_{segment_suffix}"]
                    distance_matrix_name_reverse, _, reverse = self.neighbor_finder.lookups[f"{tmp_arm}_{c2}_{segment_suffix}"]
                    if distance_matrix_name != distance_matrix_name_reverse:
                        raise ValueError(
                            "Forward and reverse lookup tablese must be defined " "on the same distance matrices."
                        )
                    empty_row = np.array([np.zeros(reverse.size, dtype=bool)])
                    reverse_lookup_values = np.vstack((*reverse.lookup.values(), empty_row))
                    reverse_lookup_keys = np.full(id_len, -1, dtype=np.int32)
                    keys_array = np.fromiter(reverse.lookup.keys(), dtype=int, count=len(reverse.lookup))
                    reverse_lookup_keys[keys_array] = np.arange(len(keys_array))
                    gene_segment_mask = sp.csr_matrix(
                        (np.empty(len(has_distance_mask.indices)), has_distance_mask.indices, has_distance_mask.indptr),
                        shape=has_distance_mask.shape,
                    )
                    has_distance_mask_coo = gene_segment_mask.tocoo()
                    indices_in_dist_mat = forward[has_distance_mask_coo.row]
                    gene_segment_mask.data = reverse_lookup_values[
                        reverse_lookup_keys[indices_in_dist_mat], has_distance_mask_coo.col
                    ]
                    return tmp_arrays.multiply(gene_segment_mask)

                if self.same_v_gene:
                    tmp_arrays = match_gene_segment(tmp_arrays, segment_suffix = "v_call")

                if self.same_j_gene:
                    tmp_arrays = match_gene_segment(tmp_arrays, segment_suffix = "j_call")

                if self.match_columns is not None:
                    tmp_arrays = tmp_arrays.multiply(match_column_mask)

                if self.dual_ir == "all":
                    filtered1, filtered2, filtered3 = filter_chain_count(tmp_arrays, tmp_arm)
                    chain_res[(tmp_arm, c1, c2, 0)] = filtered1
                    chain_res[(tmp_arm, c1, c2, 1)] = filtered2
                    chain_res[(tmp_arm, c1, c2, 2)] = filtered3
                else:
                    chain_res[(tmp_arm, c1, c2)] = tmp_arrays

            if self.dual_ir == "primary_only":
                tmp_arm_res[tmp_arm] = chain_res[(tmp_arm, 1, 1)]
            elif self.dual_ir == "any":
                tmp_arm_res[tmp_arm] = csr_min(
                    csr_min(chain_res[(tmp_arm, 1, 1)], chain_res[(tmp_arm, 1, 2)]),
                    csr_min(chain_res[(tmp_arm, 2, 1)], chain_res[(tmp_arm, 2, 2)]),
                )
            elif self.dual_ir == "all":
                tmp_arm_res[tmp_arm] = csr_min(
                    csr_max(chain_res[(tmp_arm, 1, 1, 2)], chain_res[(tmp_arm, 2, 2, 2)]),
                    csr_max(chain_res[(tmp_arm, 2, 1, 2)], chain_res[(tmp_arm, 1, 2, 2)]),
                )

                tmp_arm_res[tmp_arm] += chain_res[(tmp_arm, 1, 1, 1)] + chain_res[(tmp_arm, 1, 1, 0)]
            else:
                raise NotImplementedError(f"self.dual_ir method {self.dual_ir} is not implemented")

        if len(tmp_arm_res) == 1:
            final_res = tmp_arm_res[self._receptor_arm_cols[0]]
        else:
            if self.receptor_arms == "all":
                arm_res = {}
                arm_res[("VJ", 0)], arm_res[("VJ", 1)], arm_res[("VJ", 2)] = filter_chain_count(
                    tmp_arm_res["VJ"], "arms"
                )
                arm_res[("VDJ", 0)], arm_res[("VDJ", 1)], arm_res[("VDJ", 2)] = filter_chain_count(
                    tmp_arm_res["VDJ"], "arms"
                )
                final_res = csr_max(arm_res[("VJ", 2)], arm_res[("VDJ", 2)])
                final_res += arm_res[("VJ", 0)] + arm_res[("VJ", 1)] + arm_res[("VDJ", 0)] + arm_res[("VDJ", 1)]

            else:
                final_res = csr_min(tmp_arm_res["VJ"], tmp_arm_res["VDJ"])
        return final_res
