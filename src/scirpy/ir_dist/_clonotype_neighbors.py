import itertools
from collections.abc import Mapping, Sequence
from multiprocessing import cpu_count
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scanpy import logging
from tqdm.contrib.concurrent import process_map

from scirpy.get import _has_ir
from scirpy.get import airr as get_airr
from scirpy.util import DataHandler, tqdm

from ._util import DoubleLookupNeighborFinder, merge_coo_matrices, reduce_and, reduce_or


class ClonotypeNeighbors:
    def __init__(
        self,
        params: DataHandler,
        params2: Optional[DataHandler] = None,
        *,
        receptor_arms: Literal["VJ", "VDJ", "all", "any"],
        dual_ir: Literal["primary_only", "all", "any"],
        same_v_gene: bool = False,
        match_columns: Union[None, Sequence[str]] = None,
        distance_key: str,
        sequence_key: str,
        n_jobs: Union[int, None] = None,
        chunksize: int = 2000,
    ):
        """Computes pairwise distances between cells with identical
        receptor configuration and calls clonotypes from this distance matrix
        """
        self.same_v_gene = same_v_gene
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
                "v_gene", sp.identity(len(v_genes), dtype=bool, format="csr"), v_genes  # type: ignore
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
        """Compute the distances between clonotypes.
        Returns a clonotype x clonotype2 sparse distance matrix.
        """
        start = logging.info("Computing clonotype x clonotype distances.")  # type: ignore
        n_clonotypes = self.clonotypes.shape[0]

        # only use multiprocessing for sufficiently large datasets
        # for small datasets the overhead is too large for a benefit
        if self.n_jobs == 1 or n_clonotypes <= 2 * self.chunksize:
            dist_rows = tqdm(
                (self._dist_for_clonotype(i) for i in range(n_clonotypes)),
                total=n_clonotypes,
            )
        else:
            logging.info(
                "NB: Computation happens in chunks. The progressbar only advances " "when a chunk has finished. "
            )  # type: ignore

            dist_rows = process_map(
                self._dist_for_clonotype,
                range(n_clonotypes),
                max_workers=self.n_jobs if self.n_jobs is not None else cpu_count(),
                chunksize=2000,
                tqdm_class=tqdm,
            )

        dist = sp.vstack(list(dist_rows))
        dist.eliminate_zeros()
        logging.hint("Done computing clonotype x clonotype distances. ", time=start)
        return dist  # type: ignore

    def _dist_for_clonotype(self, ct_id: int) -> sp.csr_matrix:
        """Compute neighboring clonotypes for a given clonotype.

        Or operations use the min dist of two matching entries.
        And operations use the max dist of two matching entries.

        The motivation for using the max instead of sum/average is
        that our hypotheis is that a receptor recognizes the same antigen if it
        has a sequence dist < threshold. If we require both receptors to
        match ("and"), the higher one should count.
        """
        # Lookup distances for current row
        tmp_clonotypes = self.clonotypes2 if self.clonotypes2 is not None else self.clonotypes
        lookup = {}  # CDR3 distances
        lookup_v = {}  # V-gene distances
        for tmp_arm in self._receptor_arm_cols:
            chain_ids = [(1, 1)] if self.dual_ir == "primary_only" else [(1, 1), (2, 2), (1, 2), (2, 1)]
            for c1, c2 in chain_ids:
                lookup[(tmp_arm, c1, c2)] = self.neighbor_finder.lookup(
                    ct_id,
                    f"{tmp_arm}_{c1}",
                    f"{tmp_arm}_{c2}",
                )
                if self.same_v_gene:
                    lookup_v[(tmp_arm, c1, c2)] = self.neighbor_finder.lookup(
                        ct_id,
                        f"{tmp_arm}_{c1}_v_call",
                        f"{tmp_arm}_{c2}_v_call",
                    )

        # need to loop through all coordinates that have at least one distance.
        has_distance = merge_coo_matrices(lookup.values()).tocsr()  # type: ignore
        # convert to csr matrices to iterate over indices
        lookup = {k: v.tocsr() for k, v in lookup.items()}

        def _lookup_dist_for_chains(tmp_arm: Literal["VJ", "VDJ"], c1: Literal[1, 2], c2: Literal[1, 2]):
            """Lookup the distance between two chains of a given receptor
            arm. Only considers those columns in the current row that
            have an entry in `has_distance`. Returns a dense
            array with dimensions (1, n) where n equals the number
            of entries in `has_distance`.
            """
            ct_col2 = tmp_clonotypes[f"{tmp_arm}_{c2}_{self.sequence_key}"].values
            tmp_array = lookup[(tmp_arm, c1, c2)][0, has_distance.indices].todense().A1.astype(np.float16)
            tmp_array[ct_col2[has_distance.indices] == "nan"] = np.nan
            if self.same_v_gene:
                mask_v_gene = lookup_v[(tmp_arm, c1, c2)][0, has_distance.indices]
                tmp_array = np.multiply(tmp_array, mask_v_gene)
            return tmp_array

        # Merge the distances of chains
        res = []
        for tmp_arm in self._receptor_arm_cols:
            if self.dual_ir == "primary_only":
                tmp_res = _lookup_dist_for_chains(tmp_arm, 1, 1)
            elif self.dual_ir == "all":
                tmp_res = reduce_or(
                    reduce_and(
                        _lookup_dist_for_chains(tmp_arm, 1, 1),
                        _lookup_dist_for_chains(tmp_arm, 2, 2),
                        chain_count=self._chain_count[tmp_arm][ct_id],
                    ),
                    reduce_and(
                        _lookup_dist_for_chains(tmp_arm, 1, 2),
                        _lookup_dist_for_chains(tmp_arm, 2, 1),
                        chain_count=self._chain_count[tmp_arm][ct_id],
                    ),
                )
            else:  # "any"
                tmp_res = reduce_or(
                    _lookup_dist_for_chains(tmp_arm, 1, 1),
                    _lookup_dist_for_chains(tmp_arm, 1, 2),
                    _lookup_dist_for_chains(tmp_arm, 2, 2),
                    _lookup_dist_for_chains(tmp_arm, 2, 1),
                )

            res.append(tmp_res)

        # Merge the distances of arms.
        reduce_fun = reduce_and if self.receptor_arms == "all" else reduce_or
        # checking only the chain=1 columns here is enough, as there must not
        # be a secondary chain if there is no first one.
        res = reduce_fun(np.vstack(res), chain_count=self._chain_count["arms"][ct_id])

        if self.match_columns is not None:
            match_columns_mask = self.neighbor_finder.lookup(ct_id, "match_columns", "match_columns")
            res = np.multiply(res, match_columns_mask[0, has_distance.indices])

        final_res = has_distance.copy()
        final_res.data = res.astype(np.uint8)
        return final_res
