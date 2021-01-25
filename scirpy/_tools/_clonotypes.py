import itertools

from scipy.sparse.csr import csr_matrix
from scirpy import ir_dist
from scirpy.ir_dist import MetricType, _get_metric_key
from anndata import AnnData
import igraph as ig
from .._compat import Literal
from typing import Dict, Union, Tuple, Sequence
from ..util import _doc_params
from ..util.graph import (
    _get_igraph_from_adjacency,
    igraph_from_sparse_matrix,
    layout_components,
)
from ..ir_dist._clonotype_neighbors import ClonotypeNeighbors
import numpy as np
import pandas as pd
import random
from scanpy import logging

_common_doc = """\
    receptor_arms
         * `"TRA"` - only consider TRA sequences
         * `"TRB"` - only consider TRB sequences
         * `"all"` - both TRA and TRB need to match
         * `"any"` - either TRA or TRB need to match

        If `"any"`, two distances are combined by taking their minimum. If `"all"`,
        two distances are combined by taking their maximum. This is motivated
        by the hypothesis that a receptor recognizes the same antigen if it
        has a distance smaller than a certain cutoff. If we require only one
        of the receptors to match (`"any"`) the smaller distance is relevant.
        If we require both receptors to match (`"all"`), the larger distance is
        relevant.

    dual_ir
         * `"primary_only"` - only consider most abundant pair of TRA/TRB chains
         * `"any"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for any of the chains.
         * `"all"` - consider both pairs of TRA/TRB sequences. Distance must be below
           cutoff for all of the chains.

        Distances are combined as for `receptor_arms`.

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
"""


def _validate_parameters(
    adata,
    receptor_arms,
    dual_ir,
    within_group,
    distance_key,
    sequence,
    metric,
    key_added,
):
    """Validate an sanitze parameters for `define_clonotypes`"""
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
        distance_key = f"ir_dist_{sequence}_{_get_metric_key(metric)}"
    if distance_key not in adata.uns:
        raise ValueError(
            "Sequence distances were not found in `adata.uns`. Did you run `pp.ir_dist`?"
        )

    if key_added is None:
        key_added = f"cc_{sequence}_{_get_metric_key(metric)}"

    return within_group, distance_key, key_added


@_doc_params(common_doc=_common_doc)
def define_clonotype_clusters(
    adata: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    receptor_arms=Literal["VJ", "VDJ", "all", "any"],
    dual_ir=Literal["primary_only", "all", "any"],
    same_v_gene: bool = False,
    within_group: Union[Sequence[str], str, None] = "receptor_type",
    key_added: str = None,
    partitions: Literal["connected", "leiden"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    distance_key: Union[str, None] = None,
    inplace: bool = True,
) -> Union[Tuple[pd.Series, pd.Series, Dict], None]:
    """
    Define :term:`clonotype clusters<Clonotype cluster>`.

    As opposed to :func:`~scirpy.tl.define_clonotypes()` which employs a more stringent
    definition of :term:`clonotypes <Clonotype>`, this function flexibly defines
    clonotype clusters based on amino acid or nucleic acid sequence identity or
    similarity.

    Requires running :func:`~scirpy.pp.ir_dist` with the same `sequence` and
    `metric` values first.

    Parameters:
    -----------
    adata
        Annotated data matrix
    sequence
        The sequence parameter used when running :func:scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`

    {common_doc}

    key_added
        The column name under which the clonotype clusters and cluster sizes
        will be stored in `adata.obs` and under which the clonotype network will be
        stored in `adata.uns`.
            * Defaults to `cc_{{sequence}}_{{metric}}`, e.g. `cc_aa_levenshtein`,
              where `cc` stands for "clonotype cluster".
            * The clonotype sizes will be stored in `{{key_added}}_size`,
              e.g. `cc_aa_levenshtein_size`.
            * The clonotype x clonotype network will be stored in `{{key_added}}_dist`,
              e.g. `cc_aa_levenshtein_dist`.

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
        to `ir_dist_{{sequence}}_{{metric}}`.
    inplace
        If `True`, adds the results to anndata, otherwise returns them.

    Returns
    -------
    clonotype
        an array containing the clonotype id for each cell
    clonotype_size
        an array containing the number of cells in the respective clonotype
        for each cell.
    distances
        a pairwise distance matrix between unique rows of adata.obs
    """
    within_group, distance_key, key_added = _validate_parameters(
        adata,
        receptor_arms,
        dual_ir,
        within_group,
        distance_key,
        sequence,
        metric,
        key_added,
    )

    ctn = ClonotypeNeighbors(
        adata,
        receptor_arms=receptor_arms,
        dual_ir=dual_ir,
        same_v_gene=same_v_gene,
        within_group=within_group,
        distance_key=distance_key,
        sequence_key="cdr3" if sequence == "aa" else "cdr3_nt",
    )
    clonotype_dist = ctn.compute_distances()
    g = igraph_from_sparse_matrix(clonotype_dist, matrix_type="distance")

    if partitions == "leiden":
        part = g.community_leiden(
            objective_function="modularity",
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )
    else:
        part = g.clusters(mode="weak")

    clonotype_cluster_series = pd.Series(data=None, index=adata.obs_names, dtype=str)
    clonotype_cluster_size_series = pd.Series(
        data=None, index=adata.obs_names, dtype=int
    )

    # clonotype cluster = graph partition
    for ct_id, clonotype_cluster in enumerate(part.membership):
        clonotype_cluster_series[ctn.cell_indices[ct_id]] = str(clonotype_cluster)

    clonotype_cluster_size_series = clonotype_cluster_series.groupby(
        clonotype_cluster_series
    ).transform("count")

    # Return or store results
    clonotype_distance_res = {
        "distances": clonotype_dist,
        "distance_keys": ctn.cell_indices,
    }
    if inplace:
        adata.obs[key_added] = clonotype_cluster_series
        adata.obs[key_added + "_size"] = clonotype_cluster_size_series
        adata.uns[key_added] = clonotype_distance_res
    else:
        return (
            clonotype_cluster_series,
            clonotype_cluster_size_series,
            clonotype_distance_res,
        )


@_doc_params(common_doc=_common_doc)
def define_clonotypes(
    adata: AnnData,
    *,
    key_added: str = "clonotype",
    distance_key: Union[str, None] = None,
    **kwargs,
) -> Union[Tuple[pd.Series, pd.Series, Dict], None]:
    """
    Define :term:`clonotypes <Clonotype>` based on :term:`CDR3` nucleic acid
    sequence identity.

    As opposed to :func:`~scirpy.tl.define_clonotype_clusters` which employs
    a more flexible definition of :term:`clonotype clusters <Clonotype cluster>`,
    this function stringently defines clonotypes based on nucleic acid sequence
    identity. Technically, this function is an alias to :func:`~scirpy.tl.define_clonotype_clusters`
    with different default parameters.

    Parameters
    ----------
    adata
        Annotated data matrix
    {common_doc}
    key_added
        The column name under which the clonotype clusters and cluster sizes
        will be stored in `adata.obs` and under which the clonotype network will be
        stored in `adata.uns`
    inplace
        If `True`, adds the results to anndata, otherwise return them.
    """
    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = "ir_neighbors_nt_identity"

    if distance_key is None and "ir_dist_nt_identity" not in adata.uns:
        # For the case of "clonotypes" we want to compute the distance automatically
        # if it doesn't exist yet. Since it's just a sparse ID matrix, this
        # should be instant.
        logging.info(
            "ir_dist for sequence='nt' and metric='identity' not found. "
            "Computing with default parameters."
        )  # type: ignore
        ir_dist(adata, metric="identity", sequence="nt", key_added=distance_key)

    return define_clonotype_clusters(
        adata,
        key_added=key_added,
        sequence="nt",
        metric="identity",
        partitions="connected",
        **kwargs,
    )


def clonotype_network(
    adata: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "nt",
    metric: Literal[
        "identity", "alignment", "levenshtein", "hamming", "custom"
    ] = "identity",
    min_size: int = 1,
    layout: str = "components",
    layout_kwargs: Union[dict, None] = None,
    neighbors_key: Union[str, None] = None,
    key_clonotype_size: Union[str, None] = None,
    key_added: str = "clonotype_network",
    inplace: bool = True,
    random_state=42,
) -> Union[None, np.ndarray]:
    """Layouts the clonotype network for plotting.

    Other than with transcriptomics data, this network usually consists
    of many disconnected components, each of them representing cells
    of the same clonotype.

    Singleton clonotypes can be filtered out with the `min_size` parameter.

    Requires running :func:`scirpy.pp.ir_neighbors` first.

    Stores coordinates of the clonotype network in `adata.obsm`.

    Parameters
    ----------
    sequence
        The `sequence` parameter :func:`scirpy.pp.ir_neighbors` was ran with.
    metric
        The `metric` parameter :func:`scirpy.pp.ir_neighbors` was ran with.
    min_size
        Only show clonotypes with at least `min_size` cells.
    layout
        The layout algorithm to use. Can be anything supported by
        `igraph.Graph.layout`  or "components" to layout all connected components
        individually. See :func:`scirpy.util.graph.layout_components` for more details.
    layout_kwargs
        Will be passed to the layout function
    neighbors_key
        Key under which the neighborhood graph is stored in `adata.uns`.
        Defaults to `ir_neighbors_{sequence}_{metric}`.
    key_clonotype_size
        Key under which the clonotype size information is stored in `adata.obs`
        Defaults to `ct_cluster_{sequence}_{metric}_size`.
    key_added
        Key under which the layout coordinates will be stored in `adata.obsm` and
        parameters will be stored in `adata.uns`.
    inplace
        If `True`, store the coordinates in `adata.obsm`, otherwise return them.
    random_state
        Random seed set before computing the layout.

    Returns
    -------
    Depending on the value of `inplace` returns either nothing or the computed
    coordinates.
    """
    if neighbors_key is None:
        neighbors_key = f"ir_neighbors_{sequence}_{metric}"
    if key_clonotype_size is None:
        if sequence == "nt" and metric == "identity":
            key_clonotype_size = "clonotype_size"
        else:
            key_clonotype_size = f"ct_cluster_{sequence}_{metric}_size"
    random.seed(random_state)
    try:
        conn = adata.uns[neighbors_key]["connectivities"]
    except KeyError:
        raise ValueError("Connectivity data not found. Did you run `pp.ir_neighbors`?")

    try:
        clonotype_size = adata.obs[key_clonotype_size].values
    except KeyError:
        raise ValueError(
            "Clonotype size information not found. Did you run `tl.define_clonotypes`?"
        )

    if not adata.n_obs == conn.shape[0] == conn.shape[0]:
        raise ValueError(
            "Dimensions of connectivity matrix and AnnData do not match. Maybe you "
            "need to re-run `pp.ir_neighbors?"
        )

    graph = _get_igraph_from_adjacency(conn)

    # remove singletons/small subgraphs
    subgraph_idx = np.where(clonotype_size >= min_size)[0]
    if len(subgraph_idx) == 0:
        raise ValueError("No subgraphs with size >= {} found.".format(min_size))
    graph = graph.subgraph(subgraph_idx)

    default_layout_kwargs = {"weights": "weight"} if layout == "fr" else dict()
    layout_kwargs = default_layout_kwargs if layout_kwargs is None else layout_kwargs
    if layout == "components":
        coords = layout_components(graph, **layout_kwargs)
    else:
        coords = graph.layout(layout, **layout_kwargs).coords

    coordinates = np.full((adata.n_obs, 2), fill_value=np.nan)
    coordinates[subgraph_idx, :] = coords

    if inplace:
        adata.obsm[f"X_{key_added}"] = coordinates
        adata.uns[key_added] = {"neighbors_key": neighbors_key}
    else:
        return coordinates


def clonotype_network_igraph(
    adata: AnnData, basis="clonotype_network"
) -> Tuple[ig.Graph, ig.Layout]:
    """
    Get an `igraph` object representing the clonotype network.

    Requires running :func:`scirpy.tl.clonotype_network` before, to
    compute the layout.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Key in `adata.obsm` where the network layout is stored.

    Returns
    -------
    graph
        igraph object
    layout
        corresponding igraph Layout object.
    """
    from ..util.graph import _get_igraph_from_adjacency

    try:
        neighbors_key = adata.uns[basis]["neighbors_key"]
    except KeyError:
        raise KeyError(
            f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?"
        )

    conn = adata.uns[neighbors_key]["connectivities"]
    idx = np.where(~np.any(np.isnan(adata.obsm["X_" + basis]), axis=1))[0]
    g = _get_igraph_from_adjacency(conn).subgraph(idx)
    layout = ig.Layout(coords=adata.obsm["X_" + basis][idx, :].tolist())
    return g, layout
