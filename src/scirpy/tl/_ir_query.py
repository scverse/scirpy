import itertools
import json
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scanpy import logging

from scirpy.ir_dist import MetricType, _get_metric_key
from scirpy.ir_dist._clonotype_neighbors import ClonotypeNeighbors
from scirpy.util import DataHandler, _is_na, read_cell_indices, tqdm

from ._clonotypes import _common_doc, _common_doc_parallelism, _doc_clonotype_definition, _validate_parameters


def _validate_ir_query_annotate_params(params_ref: DataHandler, sequence, metric, query_key):
    """Validate and sanitize parameters for `ir_query_annotate`"""

    def _get_db_name():
        try:
            return params_ref.adata.uns["DB"]["name"]
        except KeyError:
            raise ValueError(
                'If reference does not contain a `.uns["DB"]["name"]` entry, '
                "you need to manually specify `distance_key` and `key_added`."
            ) from None

    if query_key is None:
        query_key = f"ir_query_{_get_db_name()}_{sequence}_{_get_metric_key(metric)}"
    return query_key


def _reduce_unique_only(values: np.ndarray):
    """If values only contains a single unique value, return that value.
    If values contains multiple different values, return 'ambiguous'.
    If values is empty, return `None`
    """
    values = values[~pd.isnull(values)]
    if values.size == 0:
        return np.nan
    elif np.unique(values).size == 1:
        return values[0]
    else:
        return "ambiguous"


def _reduce_most_frequent(values: np.ndarray):
    """Return the most frequent value in values. If two values are equally frequent,
    return "ambiguous". If values is empty, return `None`.
    """
    values = values[~pd.isnull(values)]
    if values.size == 0:
        return np.nan
    else:
        c = Counter(values)
        if (len(c)) == 1:
            return values[0]
        else:
            # 2 or more elements, get most common and 2nd most common
            (mc, mc_cnt), (mc2, mc2_cnt) = c.most_common()[:2]
            if mc_cnt > mc2_cnt:
                return mc
            else:
                return "ambiguous"


def _reduce_json(values: np.ndarray):
    """Returns a dict value -> count for each value in values as a json string."""
    values = values[~pd.isnull(values)]
    return json.dumps(Counter(values))


@DataHandler.inject_param_docs(
    common_doc=_common_doc,
    paralellism=_common_doc_parallelism,
    clonotype_definition=_doc_clonotype_definition.split("3.")[0].strip(),
)
def ir_query(
    adata: DataHandler.TYPE,
    reference: DataHandler.TYPE,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    receptor_arms: Literal["VJ", "VDJ", "all", "any"] = "all",
    dual_ir: Literal["any", "primary_only", "all"] = "any",
    same_v_gene: bool = False,
    match_columns: Sequence[str] | str | None = None,
    key_added: str | None = None,
    distance_key: str | None = None,
    inplace: bool = True,
    n_jobs: int = -1,
    chunksize: int = 2000,
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
    airr_mod_ref: str = "airr",
    airr_key_ref: str = "airr",
    chain_idx_key_ref: str = "chain_indices",
) -> dict | None:
    """\
    Query a referece database for matching immune cell receptors.

    .. warning::
        This is an experimental function that may change in the future.

    The reference database can either be an immune cell receptor database, or
    simply another scRNA-seq dataset with some annotations in `.obs`. This function
    maps all cells to all matching entries from the reference.

    Requires running :func:`~scirpy.pp.ir_dist` with the same values for `reference`,
    `sequence` and `metric` first.

    This function is essentially an extension of :func:`~scirpy.tl.define_clonotype_clusters`
    to two :class:`~anndata.AnnData` objects and follows the same logic:

    {clonotype_definition}

    Parameters
    ----------
    {adata}
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
    {airr_mod}
    {airr_key}
    {chain_idx_key}
    airr_mod_ref
        Like `airr_mod`, but for `reference`.
    airr_key_ref
        Like `airr_key`, but for `reference`.
    chain_idx_key_ref
        Like `chain_idx_key`, but for `reference`.

    Returns
    -------
    A dictionary containing
     * `distances`: A sparse distance matrix between unique receptor configurations
       in `adata` aund unique receptor configurations in `reference`.
     * `cell_indices`: A dict of lists, containing the the `adata.obs_names`
       (cell indices) for each row in the distance matrix.
     * `cell_indices_reference`: A dict of lists, containing the `reference.obs_names`
       for each column in the distance matrix.

    If `inplace` is `True`, this is added to `adata.uns[key_added]`.
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    params_ref = (
        DataHandler(reference, airr_mod_ref, airr_key_ref, chain_idx_key_ref) if reference is not None else None
    )
    match_columns, distance_key, key_added = _validate_parameters(
        params,
        params_ref if params_ref is not None else None,
        receptor_arms,
        dual_ir,
        match_columns,
        distance_key,
        sequence,
        metric,
        key_added,
    )

    ctn = ClonotypeNeighbors(
        params,
        params_ref,
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
        "cell_indices": json.dumps(ctn.cell_indices),
        "cell_indices_reference": json.dumps(ctn.cell_indices2),
    }
    if inplace:
        params.adata.uns[key_added] = clonotype_distance_res
        logging.info(f'Stored IR distance matrix in `adata.uns["{key_added}"]`.')  # type: ignore
    else:
        return clonotype_distance_res


@DataHandler.inject_param_docs()
def ir_query_annotate_df(
    adata: DataHandler.TYPE,
    reference: DataHandler.TYPE,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    include_ref_cols: Sequence[str] | None = None,
    include_query_cols: Sequence[str] = (),
    query_key: str | None = None,
    suffix: str = "",
    airr_mod: str = "airr",
    airr_mod_ref: str = "airr",
) -> pd.DataFrame:
    """
    Returns the inner join of `adata.obs` with matching entries from `reference.obs`
    based on the result of :func:`~scirpy.tl.ir_query`.

    .. warning::
        This is an experimental function that may change in the future.

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
    include_ref_cols
        Subset the reference database to these columns. Default: include all.
    include_query_cols
        Subset `adata.obs` to these columns. Default: include all.
    query_key
        Use the distance matric stored under this key in `adata.uns`. If set to None,
        the key is automatically inferred based on `reference`, `sequence`, and `metric`.
        Additional arguments are passed to the last join.
    suffix
        Suffix appended to columns from `reference.obs` in case their names
        are conflicting with those in `adata.obs`.
    {airr_mod}
    airr_mod_ref
        Like `airr_mod`, but for `reference`.

    Returns
    -------
    DataFrame with matching entries from `reference.obs`.
    """
    params = DataHandler(adata, airr_mod)
    params_ref = DataHandler(reference, airr_mod_ref)

    query_key = _validate_ir_query_annotate_params(params_ref, sequence, metric, query_key)
    res = params.adata.uns[query_key]
    dist = res["distances"]

    cell_indices = read_cell_indices(res["cell_indices"])
    cell_indices_reference = read_cell_indices(res["cell_indices_reference"])

    def get_pairs():
        for i, query_cells in cell_indices.items():
            reference_cells = itertools.chain.from_iterable(
                cell_indices_reference[str(k)] for k in dist[int(i), :].indices
            )
            yield from itertools.product(query_cells, reference_cells)

    query_obs, reference_obs = params.data.obs, params_ref.data.obs
    if include_query_cols is not None:
        query_obs = params.get_obs(include_query_cols)
    if include_ref_cols is not None:
        reference_obs = params_ref.get_obs(include_ref_cols)
    df = pd.DataFrame.from_records(get_pairs(), columns=["query_idx", "reference_idx"])
    assert "query_idx" not in reference_obs.columns
    reference_obs = reference_obs.join(df.set_index("reference_idx"), how="inner")

    return query_obs.join(reference_obs.set_index("query_idx"), rsuffix=suffix, how="inner")


@DataHandler.inject_param_docs()
def ir_query_annotate(
    adata: DataHandler.TYPE,
    reference: DataHandler.TYPE,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    strategy: Literal["json", "unique-only", "most-frequent"] = "unique-only",
    include_ref_cols: Sequence[str] | None = None,
    query_key: str | None = None,
    suffix: str = "",
    inplace=True,
    airr_mod: str = "airr",
    airr_mod_ref: str = "airr",
) -> pd.DataFrame | None:
    """
    Annotate cells based on the result of :func:`~scirpy.tl.ir_query`.

    .. warning::
        This is an experimental function that may change in the future.

    Multiple entries from the reference can match a single cell in the query dataset.
    In order to reduce the matching entries to a single value that can be added
    to `adata.obs` and used for plotting and other downstream analyses, you'll
    need to choose a strategy to deal with duplicates:

     * unique-only: Only annotate those cells that have a unique result. Cells
       with multiple inconsistent matches will receive the predicate "ambiguous"
     * most-frequent: if there are multiple matches, assign the match that is
       most frequent. If there are ties, it will receive the predicate "ambiguous"
     * json: store multiple values and their counts as json string

    NA values are ignored in all strategies (e.g. if an entry matches `"foo"` and `nan`,
    `"foo"` is considered unique)

    Alternatively, you can use :func:`scirpy.tl.ir_query_annotate_df` to obtain
    a data frame mapping all cells to their matching entries from `reference.obs`.

    Parameters
    ----------
    adata
        query dataset
    reference
        reference dataset in anndata format. Must be the same used to run
        `query_reference`.
    sequence
        The sequence parameter used when running :func:`scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`
    strategy
        Strategy to deal with non-unique values (see above).
    include_ref_cols
        Subset the reference database to these columns. Default: include all.
    query_key
        Use the distance matric stored under this key in `adata.uns`. If set to None,
        the key is automatically inferred based on `reference`, `sequence`, and `metric`.
        Additional arguments are passed to the last join.
    suffix
        Removed in v0.13. Has no effect.
    {inplace}
    {airr_mod}
    airr_mod_ref
        Like `airr_mod`, but for `reference`.

    Returns
    -------
    If inplace is True, modifies `adata.obs` inplace. Otherwise returns a data-frame
    with one column for each column in `reference.obs`, aligned to `adata.obs_names`.
    """
    params = DataHandler(adata, airr_mod)
    params_ref = DataHandler(reference, airr_mod_ref)
    df = ir_query_annotate_df(
        params,
        params_ref,
        include_ref_cols=include_ref_cols,
        include_query_cols=[],
        sequence=sequence,
        metric=metric,
        query_key=query_key,
    )
    df.index.name = "_query_cell_index"

    with tqdm(total=df.index.nunique() * df.shape[1]) as pbar:
        try:

            def reduce_fun(x):
                pbar.update(1)
                return {
                    "unique-only": _reduce_unique_only,
                    "most-frequent": _reduce_most_frequent,
                    "json": _reduce_json,
                }[strategy](  # type: ignore
                    x
                )

        except KeyError:
            raise ValueError("Invalid value for `strategy`.") from None
        df_res = df.groupby("_query_cell_index").aggregate(reduce_fun).reindex(params.adata.obs_names)

    # convert nan-equivalents to real nan values.
    for col in df_res:
        df_res.loc[_is_na(df_res[col]), col] = np.nan

    if inplace:
        for col in df_res:
            params.set_obs(str(col), df_res[col])
    else:
        return df_res
