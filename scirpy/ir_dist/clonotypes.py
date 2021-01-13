from scirpy.ir_dist import MetricType
from typing import Mapping, Union, Tuple, Sequence, Iterable, Dict
from anndata import AnnData
from .._compat import Literal
import numpy as np
import scipy.sparse as sp
import itertools
import pandas as pd
from ._util import SetDict, DoubleLookupNeighborFinder


def define_clonotypes():
    """Alias for define_clonotype_clusters based on nt sequence identity"""
    pass


def define_clonotype_clusters(
    adata: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: Literal["alignment", "levenshtein", "hamming", "identity"] = "identity",
    receptor_arms=Literal["VJ", "VDJ", "all", "any"],
    dual_ir=Literal["primary_only", "all", "any"],
    same_v_gene: bool = False,
    within_group: Union[Sequence[str], str, None] = "receptor_type",
    key_added: str = "clonotype",
    partitions: Literal["connected", "leiden"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    distance_key: Union[str, None] = None,
    inplace: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Define :term:`clonotype clusters<Clonotype cluster>`.

    Parameters:
    -----------
    adata
        Annotated data matrix
    sequence
        The sequence parameter used when running :func:scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`
    receptor_arms
         * `"TRA"` - only consider TRA sequences
         * `"TRB"` - only consider TRB sequences
         * `"all"` - both TRA and TRB need to match
         * `"any"` - either TRA or TRB need to match
    dual_ir
         * `"primary_only"` - only consider most abundant pair of TRA/TRB chains
         * `"any"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for any of the chains.
         * `"all"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for all of the chains.

        See also :term:`Dual IR`.

    same_v_gene
        Enforces clonotypes to have the same :term:`V-genes<V(D)J>`. This is useful
        as the CDR1 and CDR2 regions are fully encoded in this gene.
        See :term:`CDR` for more details.

        v genes are matched based on the behaviour defined with `receptor_arms` and
        `dual_ir`.

    within_group
        Enforces clonotypes to have the same group defined by one or multiple grouping
        variables. Per default, this is set to :term:`receptor_type<Receptor type>`,
        i.e. clonotypes cannot comprise both B cells and T cells. Set this to
        :term:`receptor_subtype<Receptor subtype>` if you don't want clonotypes to
        be shared across e.g. gamma-delta and alpha-beta T-cells.
        You can also set this to any other column in `adata.obs` that contains
        a grouping, or to `None`, if you want no constraints.

    key_added
        TODO

    partitions
        How to find graph partitions that define a clonotype.
        Possible values are `leiden`, for using the "Leiden" algorithm and
        `connected` to find fully connected sub-graphs.

        The difference is that the Leiden algorithm further divides
        fully connected subgraphs into highly-connected modules.
    resolution
        `resolution` parameter for the leiden algorithm.
    n_iterations
        `n_iterations` parameter for the leiden algorithm.
    distance_key
        Key in `adata.uns` where the sequence distances are stored. This defaults
        to `ir_dist_{sequence}_{metric}`.
    inplace
        If `True`, adds the results to anndata, otherwise returns them.

    Returns
    -------
    clonotype
        an array containing the clonotype id for each cell
    clonotype_size
        an array containing the number of cells in the respective clonotype
        for each cell.
    """
    if receptor_arms not in ["VJ", "VDJ", "all", "any"]:
        raise ValueError(
            "Invalid value for `receptor_arms`. Note that starting with v0.5 "
            "`TRA` and `TRB` are not longer valid values."
        )

    if dual_ir not in ["primary_only", "all", "any"]:
        raise ValueError("Invalid value for `dual_ir")

    if within_group is not None:
        if isinstance(within_group, str):
            within_group = [within_group]
        for group_col in within_group:
            if group_col not in adata.obs.columns:
                msg = f"column `{within_group}` not found in `adata.obs`. "
                if group_col in ("receptor_type", "receptor_subtype"):
                    msg += "Did you run `tl.chain_qc`? "
                raise ValueError(msg)

    if distance_key is None:
        distance_key = f"ir_dist_{sequence}_{metric}"
    try:
        distance_dict = adata.uns[distance_key]
    except KeyError:
        raise ValueError(
            "Sequence distances were not found in `adata.uns`. Did you run `pp.ir_dist`?"
        )

    sequence_key = "cdr3" if sequence == "aa" else "cdr3_nt"

    ctn = ClonotypeNeighbors(
        adata,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
        within_group=within_group,
        distance_dict=distance_dict,
        sequence_key=sequence_key,
    )
    # TODO log progress and time
    ctn.prepare()
    ctn.compute_distances()
    pass

    pass


class ClonotypeNeighbors:
    def __init__(
        self,
        adata: AnnData,
        *,
        receptor_arms=Literal["VJ", "VDJ", "all", "any"],
        dual_ir=Literal["primary_only", "all", "any"],
        same_v_gene: bool,
        within_group: Union[None, Sequence[str]],
        distance_dict: Mapping[str, Mapping[str, object]],
        sequence_key: str,
    ):
        self.adata = adata
        self.same_v_gene = same_v_gene
        self.within_group = within_group
        self.receptor_arms = receptor_arms
        self.dual_ir = dual_ir
        self.distance_dict = distance_dict
        self.sequence_key = sequence_key

        self._receptor_arm_cols = (
            ["VJ", "VDJ"]
            if self.receptor_arms in ["all", "any"]
            else [self.receptor_arms]
        )
        self._dual_ir_cols = ["1"] if self.dual_ir == "primary_only" else ["1", "2"]

    def prepare(self):
        self._make_clonotype_table()
        self._add_distance_matrices()
        self._add_lookup_tables()
        self.neighbor_finder = DoubleLookupNeighborFinder(self.clonotypes)

    def _make_clonotype_table(self):
        """Define clonotypes based identical IR features"""
        # Define clonotypes. TODO v-genes, within_group
        clonotype_cols = [
            f"IR_{arm}_{i}_{self.sequence_key}"
            for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols)
        ]
        self.clonotypes = (
            self.adata.obs.loc[clonotype_cols, :].drop_duplicates().reset_index()
        )

    def _add_distance_matrices(self):
        for chain_type in self._receptor_arm_cols:
            distance_dict = self.adata.uns[self.distance_key]
            self.neighbor_finder.add_distance_matrix(
                name=chain_type,
                distance_matrix=distance_dict["distances"],
                labels=distance_dict["seqs"],
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
        for arm, i in itertools.product(self._receptor_arm_cols, self._dual_ir_cols):
            self.neighbor_finder.add_lookup_table(
                f"{arm}_{i}", f"IR_{arm}_{i}_{self.sequence_key}", arm
            )

        # self.neighbor_finder.add_lookup_table("VJ_v", "IR_VJ_1_v_gene", "v_gene")
        # self.neighbor_finder.add_lookup_table("VDJ_v", "IR_VDJ_1_v_gene", "v_gene")

    def compute_distances(self):
        for i, ct in self.clonotypes.itertuples():
            self.neighbor_finder.lookup(i, "VJ_1")
