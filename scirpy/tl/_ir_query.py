from typing import Optional, Union, Sequence

from .._compat import Literal
from anndata import AnnData
import pandas as pd
import itertools
from ._clonotypes import (
    _common_doc,
    _common_doc_parallelism,
    _validate_parameters,
    _doc_clonotype_definition,
)
from ..util import _doc_params
from ..ir_dist._clonotype_neighbors import ClonotypeNeighbors
from ..ir_dist import _get_metric_key, MetricType
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

    The reference database can either be a imune cell receptor database, or
    simply another scRNA-seq dataset with some annotations in `.obs`. This function
    maps all cells to all matching entries from the reference.

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


def _validate_ir_query_annotate_params(reference, sequence, metric, query_key):
    """Validate and sanitize parameters for `ir_query_annotate`"""

    def _get_db_name():
        try:
            return reference.uns["DB"]["name"]
        except KeyError:
            raise ValueError(
                'If reference does not contain a `.uns["DB"]["name"]` entry, '
                "you need to manually specify `distance_key` and `key_added`."
            )

    if query_key is None:
        query_key = f"ir_query_{_get_db_name()}_{sequence}_{_get_metric_key(metric)}"
    return query_key


# TODO add test
def ir_query_annotate_df(
    adata: AnnData,
    reference: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    include_cols: Sequence[str] = None,
    query_key=None,
    suffix="_ref",
    **kwargs,
) -> pd.DataFrame:
    """
    Returns the inner join of `adata.obs` with matching entries from `reference.obs`.

    The function first creates a two-column dataframe mapping cell indices of `adata`
    to cell indices of `reference`. It then performs an inner join with `reference.obs`,
    and finally performs another join with `query.obs`.

    This function requires that `~scirpy.tl.ir_query` has been executed on `adata`
    with the same reference and the same parameters for `sequence` and `metric`.

    This function returns all matching entries in the reference database, which can
    be none for some cells, but many for others. If you want to add a single column
    to `adata.obs` for plotting, please refer to `~scirpy.tl.ir_query_annotate`.

    Parameters
    ----------
    adata
        query dataset
    reference
        reference dataset
    sequence
        The sequence parameter used when running :func:`scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`
    include_cols
        Subset the reference database to these columns
    query_key
        Use the distance matric stored under this key in `adata.uns`. If set to None,
        the key is automatically inferred based on `reference`, `sequence`, and `metric`.
        Additional arguments are passed to the last join.
    suffix
        Suffix appended to columns from `reference.obs` in case their names
        are conflicting with those in `adata.obs`.

    Returns
    -------
    DataFrame with matching entries from `reference.obs`.
    """
    query_key = _validate_ir_query_annotate_params(
        reference, sequence, metric, query_key
    )
    res = adata.uns[query_key]
    dist = res["distances"]

    def get_pairs():
        for i, query_cells in res["cell_indices"].items():
            reference_cells = itertools.chain.from_iterable(
                res["cell_indices_reference"][str(k)] for k in dist[int(i), :].indices
            )
            yield from itertools.product(query_cells, reference_cells)

    reference_obs = (
        reference.obs if include_cols is None else reference.obs.loc[:, include_cols]
    )
    df = pd.DataFrame.from_records(get_pairs(), columns=["query_idx", "reference_idx"])
    assert "query_idx" not in reference_obs.columns
    reference_obs = reference_obs.join(df.set_index("reference_idx"))

    return adata.obs.join(reference_obs.set_index("query_idx"), rsuffix=suffix)


def ir_query_annotate(
    adata: AnnData,
    reference: AnnData,
    *,
    include_cols: Optional[Sequence[str]] = None,
    strategy: Literal["one-to-many", "unique-only", "most-frequent"] = "unique-only",
    sequence,
    metric,
    query_key,
):
    """
    Annotate cells based on matching :term:`IR`s in a reference dataset.

    Once you added a distance matrix between query and reference dataset, you
    likely want to annotated the dataset based on the reference.
    The issue is that multiple entries from the reference may match
    a single cell in the query.

    Therefore, you'll need to choose a strategy to deal with duplicates

     * one-to-many:  Don't modify obs, but return a table that maps
       each cell from the query to all entries from the reference. If
       inplace=True, will write json to obs. Use this to get the data and manually deal with it
     * unique-only: Only annotate those cells that have a unique result. Cells
       with multiple inconsistent matches will receive the predicate "ambiguous"
     * most-frequent: if therer are multiple matches, assign the match that is
       most frequent. If there are ties, it will receive the predicate "ambiguous"

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
