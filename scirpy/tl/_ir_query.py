from typing import Optional, Union, Sequence, Tuple

from scirpy.ir_dist import MetricType
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from ._clonotypes import (
    _common_doc,
    _common_doc_parallelism,
    _validate_parameters,
    _doc_clonotype_definition,
)
from ..util import _doc_params
from ..ir_dist._clonotype_neighbors import ClonotypeNeighbors
from scanpy import logging


@_doc_params(
    common_doc=_common_doc,
    paralellism=_common_doc_parallelism,
    clonotype_definition=_doc_clonotype_definition.split("3.")[0].strip(),
)
def ir_query(
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
) -> Optional[dict]:
    """
    Query a referece database for matching immune cell receptors.

    Requires funning :func:`~scirpy.pp.ir_dist` with the same values for `reference`,
    `sequence` and `metric` first.

    This function is essentially an extension of :func:`~scirpy.tl.define_clonotype_clusters`
    to two :class:`~anndata.AnnData` objects and follows the same logic:

    {clonotype_definition}

    Parameters
    ----------
    adata
        annotated data matrix
    reference
        Another :class:`~anndata.AnnData` object, can be either a second dataset with
        :term:`IR` information or a epitope database. Must be the same object used for
        running :func:`scirpy.pp.ir_dist`.
    sequence
        The sequence parameter used when running :func:`scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`

    {common_doc}

    match_columns
        One or multiple columns in `adata.obs` that must match between
        query and reference. Use this to e.g. enforce matching cell-types or HLA-types.

    key_added
        Dictionary key under which the resulting distance matrix will be stored in
        `adata.uns` if `inplace=True`. Defaults to `ir_query_{{name}}_{{sequence}}_{{metric}}`.
        If `metric` is an instance of :class:`scirpy.ir_dist.metrics.DistanceCalculator`,
        `{{metric}}` defaults to `custom`.
        `{{name}}` is taken from `reference.uns["DB"]["name"]`. If `reference` does not have a
        `"DB"` entry, `key_added` needs to be specified manually.
    distance_key
        Key in `adata.uns` where the results of :func:`~scirpy.pp.ir_dist` are stored.
        Defaults to `ir_dist_{{name}}_{{sequence}}_{{metric}}`.
        If `metric` is an instance of :class:`scirpy.ir_dist.metrics.DistanceCalculator`,
        `{{metric}}` defaults to `custom`.
        `{{name}}` is taken from `reference.uns["DB"]["name"]`. If `reference` does not have a
        `"DB"` entry, `distance_key` needs to be specified manually.
    inplace
        If True, store the result in `adata.uns`. Otherwise return a dictionary
        with the results.
    {paralellism}

    Returns
    -------
    A dictionary containing
     * `distances`: A sparse distance matrix between unique receptor configurations
       in `adata` aund unique receptor configurations in `reference`.
     * `cell_indices`: A dict of arrays, containing the the `adata.obs_names`
       (cell indices) for each row in the distance matrix.
     * `cell_indices_reference`: A dict of arrays, containing the `reference.obs_names`
       for each column in the distance matrix.

    If `inplace` is `True`, this is added to `adata.uns[key_added]`.
    """
    match_columns, distance_key, key_added = _validate_parameters(
        adata,
        reference,
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
        match_columns=match_columns,
        distance_key=distance_key,
        sequence_key="junction_aa" if sequence == "aa" else "junction",
        n_jobs=n_jobs,
        chunksize=chunksize,
    )
    clonotype_dist = ctn.compute_distances()

    # Return or store results
    clonotype_distance_res = {
        "distances": clonotype_dist,
        "cell_indices": ctn.cell_indices,
        "cell_indices_reference": ctn.cell_indices2,
    }
    if inplace:
        adata.uns[key_added] = clonotype_distance_res
        logging.info(f'Stored IR distance matrix in `adata.uns["{key_added}"]`.')  # type: ignore
    else:
        return clonotype_distance_res


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
