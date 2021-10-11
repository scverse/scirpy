from typing import Optional, Union, Sequence, Tuple

from scirpy.ir_dist import MetricType
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from ._clonotypes import _common_doc, _common_doc_parallelism, _validate_parameters
from ..util import _doc_params
from ..ir_dist._clonotype_neighbors import ClonotypeNeighbors


@_doc_params(common_doc=_common_doc, paralellism=_common_doc_parallelism)
def query_reference(
    adata: AnnData,
    reference: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    receptor_arms: Literal["VJ", "VDJ", "all", "any"] = "all",
    dual_ir: Literal["any", "primary_only", "all"] = "any",
    same_v_gene: bool = False,
    match_columns: Union[Sequence[str], str, None] = None,
    key_added: Optional[str] = None,
    distance_key: Optional[str] = None,
    inplace: bool = True,
    n_jobs: Optional[int] = None,
    chunksize: int = 2000,
) -> Optional[Tuple[pd.Series, pd.Series, dict]]:
    """
    Annotate epitopes based on a reference database.

    Matches

    Parameters
    ----------
    adata
        annotated data matrix
    reference
        Another adata object, can be either a second dataset with :term:`IR` information
        or a epitope database. Must be the same object used when running
        :func:`scirpy.pp.ir_dist`
    sequence
        The sequence parameter used when running :func:`scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`
    {common_doc}
    match_columns
        One or multiple columns in `adata.obs` that must match between
        query and reference.
    key_added
    distance_key
    inplace
    {paralellism}

    Returns
    -------
    TODO


    """
    match_columns, distance_key, key_added = _validate_parameters(
        adata,
        receptor_arms,
        dual_ir,
        match_columns,
        distance_key,
        sequence,
        metric,
        key_added,
    )

    ctn = ClonotypeNeighbors(
        adata,
        reference,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
        # TODO would this work, maybe rename argument
        within_group=match_columns,
        distance_key=distance_key,
        sequence_key="junction_aa" if sequence == "aa" else "junction",
        n_jobs=n_jobs,
        chunksize=chunksize,
    )
    clonotype_dist = ctn.compute_distances()
    # TODO this looks as I could re-use basically all the code to call clonotypes here.

    # Return or store results
    clonotype_distance_res = {
        "distances": clonotype_dist,
        "cell_indices": ctn.cell_indices,
    }
    if inplace:
        adata.obs[key_added] = clonotype_cluster_series
        adata.obs[key_added + "_size"] = clonotype_cluster_size_series
        adata.uns[key_added] = clonotype_distance_res
        logging.info(f'Stored clonal assignments in `adata.obs["{key_added}"]`.')
    else:
        return (
            clonotype_cluster_series,
            clonotype_cluster_size_series,
            clonotype_distance_res,
        )


def annotate_with_reference(
    adata: AnnData, reference: AnnData, *, include_cols: Optional[Sequence[str]] = None
):
    """
    Once you added a distance matrix between query and reference dataset, you
    likely want to annotated the dataset based on the reference.
    The issue is that multiple entries from the reference may match
    a single cell in the query.

    Therefore, you'll need to choose a strategy to deal with duplicates

     * one-to-many-table: Don't modify obs, but return a table that maps
       each cell from the query to all entries from the reference. If
       inplace=True, will write json to obs. Use this to get the data and manually deal with it
     * unique-only: Only annotate those cells that have a unique result. Cells
       with multiple matches will receive the predicate "ambiguous"
     * most-frequent: if therer are multiple matches, assign the match that is
       most frequent. If there are ties, it will use the first entry.

    Parameters
    ----------
    adata
        query dataset
    reference
        reference dataset in anndata format. Must be the same used to run
        `query_reference`.
    include_cols
        List of columns from the reference database to add to obs of the query dataset.
        If set to None, will include all columns from the reference.
    """
    pass
