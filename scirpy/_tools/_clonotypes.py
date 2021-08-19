import itertools
import random
from typing import List, Optional, Sequence, Tuple, Union

import igraph as ig
import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging
import scipy.sparse as sp

from .._compat import Literal
from .._preprocessing import ir_dist
from ..ir_dist import MetricType, _get_metric_key
from ..ir_dist._clonotype_neighbors import ClonotypeNeighbors
from ..util import _doc_params
from ..util.graph import igraph_from_sparse_matrix, layout_components
from ..io._util import _check_upgrade_schema

_common_doc = """\
receptor_arms
    One of the following options:
      * `"VJ"` - only consider :term:`VJ<V(D)J>` sequences
      * `"VDJ"` - only consider :term:`VDJ<V(D)J>` sequences
      * `"all"` - both VJ and VDJ need to match
      * `"any"` - either VJ or VDJ need to match

    If `"any"`, two distances are combined by taking their minimum. If `"all"`,
    two distances are combined by taking their maximum. This is motivated
    by the hypothesis that a receptor recognizes the same antigen if it
    has a distance smaller than a certain cutoff. If we require only one
    of the receptors to match (`"any"`) the smaller distance is relevant.
    If we require both receptors to match (`"all"`), the larger distance is
    relevant.

dual_ir
    One of the following options:
      * `"primary_only"` - only consider most abundant pair of :term:`VJ/VDJ<V(D)J>` chains
      * `"any"` - consider both pairs of :term:`VJ/VDJ<V(D)J>` sequences. Distance must be below
        cutoff for any of the chains.
      * `"all"` - consider both pairs of :term:`VJ/VDJ<V(D)J>` sequences. Distance must be below
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

_common_doc_parallelism = """\
n_jobs
    Number of CPUs to use for clonotype cluster calculation. Default: use all cores.
    If the number of cells is smaller than `2 * chunksize` a single
    worker thread will be used to avoid overhead.
chunksize
    Number of objects to process per chunk. Each worker thread receives
    data in chunks. Smaller chunks result in a more meaningful progressbar,
    but more overhead.
"""

_common_doc_return_values = """\
Returns
-------
clonotype
    A Series containing the clonotype id for each cell. Will be stored in
    `adata.obs[key_added]` if `inplace` is `True`
clonotype_size
    A Series containing the number of cells in the respective clonotype
    for each cell. Will be stored in `adata.obs[f"{key_added}_size"]` if `inplace`
    is `True`.
distance_result
    A dictionary containing
     * `distances`: A sparse, pairwise distance matrix between unique
       receptor configurations
     * `cell_indices`: A dict of arrays, containing the adata.obs_names
       (cell indices) for each row in the distance matrix.

    If `inplace` is `True`, this is added to `adata.uns[key_added]`.
"""

_doc_clonotype_definition = """\
Definition of clonotype(-clusters) follows roughly the following procedure:
  1. Create a list of unique receptor configurations. This is useful to
     collapse heavily expanded clonotypes, leading to many cells with identical
     CDR3 sequences, to a single entry.
  2. Compute a pairwise distance matrix of unique receptor configurations.
     Unique receptor configurations are matched based on the pre-computed
     VJ and VDJ distance matrices and the parameters of `receptor_arms`,
     `dual_ir`, `same_v_gene` and `within_group`.
  3. Find connected modules in the graph defined by this distance matrix. Each
     connected module is considered a clonotype-cluster.
"""

_doc_clonotype_network = """\
The clonotype network usually consists of many disconnected components,
each of them representing a clonotype. Each node represents cells with an identical
receptor configuration (i.e. identical CDR3 sequences, and identical
v genes if `same_v_gene` was specified during clonotype definition).
The size of each dot refers to the number of cells with the same receptor
configurations.

For more details on the clonotype definition, see
:func:`scirpy.tl.define_clonotype_clusters` and the respective section
in the tutorial.
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
) -> Tuple[Optional[List[str]], str, str]:
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


@_check_upgrade_schema()
@_doc_params(
    common_doc=_common_doc,
    clonotype_definition=_doc_clonotype_definition,
    return_values=_common_doc_return_values,
    paralellism=_common_doc_parallelism,
)
def define_clonotype_clusters(
    adata: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    receptor_arms: Literal["VJ", "VDJ", "all", "any"] = "all",
    dual_ir: Literal["primary_only", "all", "any"] = "any",
    same_v_gene: bool = False,
    within_group: Union[Sequence[str], str, None] = "receptor_type",
    key_added: str = None,
    partitions: Literal["connected", "leiden"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    distance_key: Union[str, None] = None,
    inplace: bool = True,
    n_jobs: Union[int, None] = None,
    chunksize: int = 2000,
) -> Optional[Tuple[pd.Series, pd.Series, dict]]:
    """
    Define :term:`clonotype clusters<Clonotype cluster>`.

    As opposed to :func:`~scirpy.tl.define_clonotypes()` which employs a more stringent
    definition of :term:`clonotypes <Clonotype>`, this function flexibly defines
    clonotype clusters based on amino acid or nucleic acid sequence identity or
    similarity.

    Requires running :func:`~scirpy.pp.ir_dist` with the same `sequence` and
    `metric` values first.

    {clonotype_definition}

    Parameters
    ----------
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
    {paralellism}

    {return_values}
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
        sequence_key="junction_aa" if sequence == "aa" else "junction",
        n_jobs=n_jobs,
        chunksize=chunksize,
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
    idx, values = zip(
        *itertools.chain.from_iterable(
            zip(ctn.cell_indices[str(ct_id)], itertools.repeat(str(clonotype_cluster)))
            for ct_id, clonotype_cluster in enumerate(part.membership)
        )
    )
    clonotype_cluster_series = pd.Series(values, index=idx).reindex(adata.obs_names)
    clonotype_cluster_size_series = clonotype_cluster_series.groupby(
        clonotype_cluster_series
    ).transform("count")

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


@_check_upgrade_schema()
@_doc_params(
    common_doc=_common_doc,
    clonotype_definition=_doc_clonotype_definition,
    return_values=_common_doc_return_values,
    paralellism=_common_doc_parallelism,
)
def define_clonotypes(
    adata: AnnData,
    *,
    key_added: str = "clone_id",
    distance_key: Union[str, None] = None,
    **kwargs,
) -> Optional[Tuple[pd.Series, pd.Series, dict]]:
    """
    Define :term:`clonotypes <Clonotype>` based on :term:`CDR3` nucleic acid
    sequence identity.

    As opposed to :func:`~scirpy.tl.define_clonotype_clusters` which employs
    a more flexible definition of :term:`clonotype clusters <Clonotype cluster>`,
    this function stringently defines clonotypes based on nucleic acid sequence
    identity. Technically, this function is an alias to :func:`~scirpy.tl.define_clonotype_clusters`
    with different default parameters.

    {clonotype_definition}

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
    {paralellism}

    {return_values}

    """
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


@_check_upgrade_schema()
@_doc_params(clonotype_network=_doc_clonotype_network)
def clonotype_network(
    adata: AnnData,
    *,
    sequence: Literal["aa", "nt"] = "nt",
    metric: Literal[
        "identity", "alignment", "levenshtein", "hamming", "custom"
    ] = "identity",
    min_cells: int = 1,
    min_nodes: int = 1,
    layout: str = "components",
    size_aware: bool = True,
    base_size: Optional[float] = None,
    size_power: float = 1,
    layout_kwargs: Union[dict, None] = None,
    clonotype_key: Union[str, None] = None,
    key_added: str = "clonotype_network",
    inplace: bool = True,
    random_state=42,
) -> Union[None, pd.DataFrame]:
    """
    Computes the layout of the clonotype network.

    Requires running :func:`scirpy.tl.define_clonotypes` or
    :func:`scirpy.tl.define_clonotype_clusters` first.

    {clonotype_network}

    Singleton clonotypes can be filtered out with the `min_cells` and `min_nodes`
    parameters.

    The `components` layout algorithm takes node sizes into account, avoiding
    overlapping nodes. Therefore, we recommend specifying `base_size` and
    `size_power` already here instead of providing them to
    :func:`scirpy.pl.clonotype_network`.

    Stores coordinates of the clonotype network in `adata.obsm`.

    Parameters
    ----------
    adata
        annotated data matrix
    sequence
        The `sequence` parameter :func:`scirpy.tl.define_clonotypes` was ran with.
    metric
        The `metric` parameter :func:`scirpy.tl.define_clonotypes` was ran with.
    min_cells
        Only show clonotypes consisting of at least `min_cells` cells
    min_nodes
        Only show clonotypes consisting of at least `min_nodes` nodes (i.e.
        non-identical receptor configurations)
    layout
        The layout algorithm to use. Can be anything supported by
        `igraph.Graph.layout`, or "components" to layout all connected
        components individually.
        :func:`scirpy.util.graph.layout_components` for more details.
    size_aware
        If `True`, use a node-size aware layouting algorithm. This option is
        only compatible with `layout = 'components'`.
    base_size
        Size of a point respresenting 1 cell. Per default, this value is a
        automatically determined based on the number of nodes in the plot.
    size_power
        Sizes are raised to the power of this value. Set this to, e.g. 0.5 to
        dampen point size.
    layout_kwargs
        Will be passed to the layout function
    clonotype_key
        Key under which the result of :func:`scirpy.tl.define_clonotypes` or
        :func:`scirpy.tl.define_clonotype_clusters` is stored in `adata.uns`.
        Defaults to `clone_id` if `sequence == 'nt' and distance == 'identity'` or
        `cc_{{sequence}}_{{metric}}` otherwise.
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
    if size_aware and layout != "components":
        raise ValueError(
            "The `size_aware` option is only compatible with the `components` layout."
        )
    params_dict = dict()
    random.seed(random_state)
    np.random.seed(random_state)

    if clonotype_key is None:
        if metric == "identity" and sequence == "nt":
            clonotype_key = "clone_id"
        else:
            clonotype_key = f"cc_{sequence}_{metric}"

    try:
        clonotype_res = adata.uns[clonotype_key]
    except KeyError:
        raise ValueError(
            "Connectivity data not found. Did you run `tl.define_clonotypes` "
            "or `tl.define_clonotype_clusters`, respectively?"
        )

    graph = igraph_from_sparse_matrix(
        clonotype_res["distances"], matrix_type="distance"
    )

    if base_size is None:
        base_size = 240000 / len(graph.vs)

    # explicitly annotate node ids to keep them after subsetting
    graph.vs["node_id"] = np.arange(0, len(graph.vs))

    # store size in graph to be accessed by layout algorithms
    clonotype_size = np.array(
        [idx.size for idx in clonotype_res["cell_indices"].values()]
    )
    graph.vs["size"] = clonotype_size
    components = np.array(graph.decompose("weak"))
    component_node_count = np.array([len(component.vs) for component in components])
    component_sizes = np.array([sum(component.vs["size"]) for component in components])

    # Filter subgraph by `min_cells` and `min_nodes`
    subgraph_idx = list(
        itertools.chain.from_iterable(
            comp.vs["node_id"]
            for comp in components[
                (component_node_count >= min_nodes) & (component_sizes >= min_cells)
            ]
        )
    )
    if len(subgraph_idx) == 0:
        raise ValueError("No subgraphs with size >= {} found.".format(min_cells))
    graph = graph.subgraph(subgraph_idx)

    # Compute layout
    if layout_kwargs is None:
        layout_kwargs = dict()
    if layout == "components":
        tmp_layout_kwargs = dict()
        tmp_layout_kwargs["component_layout"] = "fr_size_aware" if size_aware else "fr"
        if size_aware:
            # layout kwargs for the fr_size_aware layout used for each component
            tmp_layout_kwargs["layout_kwargs"] = {
                "base_node_size": base_size / 2000,
                "size_power": size_power,
            }
            pad = 3
            tmp_layout_kwargs["pad_x"] = pad
            tmp_layout_kwargs["pad_y"] = pad

        tmp_layout_kwargs.update(layout_kwargs)
        coords = layout_components(graph, **tmp_layout_kwargs)
    else:
        tmp_layout_kwargs = {"weights": "weight"} if layout == "fr" else dict()
        tmp_layout_kwargs.update(layout_kwargs)
        coords = graph.layout(layout, **tmp_layout_kwargs).coords

    # Expand to cell coordinates to store in adata.obsm
    idx, coords = zip(
        *itertools.chain.from_iterable(
            zip(clonotype_res["cell_indices"][str(node_id)], itertools.repeat(coord))
            for node_id, coord in zip(graph.vs["node_id"], coords)  # type: ignore
        )
    )
    coord_df = pd.DataFrame(data=coords, index=idx, columns=["x", "y"]).reindex(
        adata.obs_names
    )

    # Store results or return
    if inplace:
        adata.obsm[f"X_{key_added}"] = coord_df
        params_dict["clonotype_key"] = clonotype_key
        params_dict["base_size"] = base_size
        params_dict["size_power"] = size_power
        adata.uns[key_added] = params_dict
    else:
        return coord_df


def _graph_from_coordinates(
    adata: AnnData, clonotype_key: str
) -> [pd.DataFrame, sp.spmatrix]:
    """
    Given an AnnData object on which `tl.clonotype_network` was ran, and
    the corresponding `clonotype_key`, extract a data-frame
    with x and y coordinates for each node, and an aligned adjacency matrix.

    Combined, it can be used for plotting the layouted graph with igraph or networkx.
    """
    clonotype_res = adata.uns[clonotype_key]
    # map the cell-id to the corresponding row/col in the clonotype distance matrix
    dist_idx, obs_names = zip(
        *itertools.chain.from_iterable(
            zip(itertools.repeat(i), obs_names)
            for i, obs_names in clonotype_res["cell_indices"].items()
        )
    )
    dist_idx_lookup = pd.DataFrame(index=obs_names, data=dist_idx, columns=["dist_idx"])
    clonotype_label_lookup = adata.obs.loc[:, [clonotype_key]].rename(
        columns={clonotype_key: "label"}
    )

    # Retrieve coordinates and reduce them to one coordinate per node
    coords = (
        adata.obsm["X_clonotype_network"]
        .dropna(axis=0, how="any")
        .join(dist_idx_lookup)
        .join(clonotype_label_lookup)
        .groupby(by=["label", "dist_idx", "x", "y"], observed=True)
        .size()
        .reset_index(name="size")
    )

    # Networkx graph object for plotting edges
    adj_mat = clonotype_res["distances"][coords["dist_idx"].values.astype(int), :][
        :, coords["dist_idx"].values.astype(int)
    ]

    return coords, adj_mat


@_check_upgrade_schema()
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
    from ..util.graph import igraph_from_sparse_matrix

    try:
        clonotype_key = adata.uns[basis]["clonotype_key"]
    except KeyError:
        raise KeyError(
            f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?"
        )
    if f"X_{basis}" not in adata.obsm_keys():
        raise KeyError(
            f"X_{basis} not found in `adata.obsm`. Did you run `tl.clonotype_network`?"
        )
    coords, adj_mat = _graph_from_coordinates(adata, clonotype_key)

    graph = igraph_from_sparse_matrix(adj_mat, matrix_type="distance")
    # flip y axis to be consistent with networkx
    coords["y"] = np.max(coords["y"]) - coords["y"]
    layout = ig.Layout(coords=coords.loc[:, ["x", "y"]].values.tolist())
    return graph, layout
