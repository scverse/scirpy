import itertools
import json
import random
from collections.abc import Sequence
from typing import Literal, cast

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scanpy import logging

from scirpy.ir_dist import MetricType, _get_metric_key
from scirpy.ir_dist._clonotype_neighbors import ClonotypeNeighbors
from scirpy.pp import ir_dist
from scirpy.util import DataHandler, read_cell_indices
from scirpy.util.graph import igraph_from_sparse_matrix, layout_components

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
"""

_common_doc_within_group = """\
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
     * `cell_indices`: A dict of lists, containing the `adata.obs_names`
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
    params: DataHandler,
    reference: DataHandler,
    receptor_arms,
    dual_ir,
    within_group,
    distance_key,
    sequence,
    metric,
    key_added,
) -> tuple[list[str] | None, str, str]:
    """Validate and sanitze parameters for `define_clonotypes`"""

    def _get_db_name():
        try:
            return reference.adata.uns["DB"]["name"]
        except KeyError:
            raise ValueError(
                'If reference does not contain a `.uns["DB"]["name"]` entry, '
                "you need to manually specify `distance_key` and `key_added`."
            ) from None

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
            try:
                params.get_obs(group_col)
            except KeyError:
                msg = f"column `{group_col}` not found in `obs`. "
                if group_col in ("receptor_type", "receptor_subtype"):
                    msg += "Did you run `tl.chain_qc`? "
                raise ValueError(msg) from None

    if distance_key is None:
        if reference is not None:
            distance_key = f"ir_dist_{_get_db_name()}_{sequence}_{_get_metric_key(metric)}"
        else:
            distance_key = f"ir_dist_{sequence}_{_get_metric_key(metric)}"
    if distance_key not in params.adata.uns:
        raise ValueError("Sequence distances were not found in `adata.uns`. Did you run `pp.ir_dist`?")

    if key_added is None:
        if reference is not None:
            key_added = f"ir_query_{_get_db_name()}_{sequence}_{_get_metric_key(metric)}"
        else:
            key_added = f"cc_{sequence}_{_get_metric_key(metric)}"

    return within_group, distance_key, key_added


@DataHandler.inject_param_docs(
    common_doc=_common_doc,
    within_group=_common_doc_within_group,
    clonotype_definition=_doc_clonotype_definition,
    return_values=_common_doc_return_values,
    paralellism=_common_doc_parallelism,
)
def define_clonotype_clusters(
    adata: DataHandler.TYPE,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: MetricType = "identity",
    receptor_arms: Literal["VJ", "VDJ", "all", "any"] = "all",
    dual_ir: Literal["primary_only", "all", "any"] = "any",
    same_v_gene: bool = False,
    same_j_gene: bool = False,
    within_group: Sequence[str] | str | None = "receptor_type",
    key_added: str | None = None,
    partitions: Literal["connected", "leiden", "fastgreedy"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    distance_key: str | None = None,
    inplace: bool = True,
    n_jobs: int = -1,
    chunksize: int = 2000,
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
) -> tuple[pd.Series, pd.Series, dict] | None:
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
    {adata}
    sequence
        The sequence parameter used when running :func:`scirpy.pp.ir_dist`
    metric
        The metric parameter used when running :func:`scirpy.pp.ir_dist`

    {common_doc}

    {within_group}

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
        Possible values are `leiden`, for using the "Leiden" algorithm,
        `fastgreedy` for using the "Fastgreedy" algorithm and
        `connected` to find fully connected sub-graphs.

        The difference is that the Leiden and Fastgreedy algorithms further divide
        fully connected subgraphs into highly-connected modules.

        "Leiden" finds the community structure of the graph using the
        Leiden algorithm of Traag, van Eck & Waltman.

        "Fastgreedy" finds the community structure of the graph according to the
        algorithm of Clauset et al based on the greedy optimization of modularity.

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
    {airr_mod}
    {airr_key}
    {chain_idx_key}

    {return_values}
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    within_group, distance_key, key_added = _validate_parameters(
        params,
        None,
        receptor_arms,
        dual_ir,
        within_group,
        distance_key,
        sequence,
        metric,
        key_added,
    )

    ctn = ClonotypeNeighbors(
        params,
        receptor_arms=receptor_arms,  # type: ignore
        dual_ir=dual_ir,  # type: ignore
        same_v_gene=same_v_gene,
        same_j_gene=same_j_gene,
        match_columns=within_group,
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
    elif partitions == "fastgreedy":
        part = g.community_fastgreedy().as_clustering()
    else:
        part = g.clusters(mode="weak")

    clonotype_cluster_series = pd.Series(data=None, index=params.adata.obs_names, dtype=str)
    clonotype_cluster_size_series = pd.Series(data=None, index=params.adata.obs_names, dtype=int)

    # clonotype cluster = graph partition
    idx, values = zip(
        *itertools.chain.from_iterable(
            zip(ctn.cell_indices[str(ct_id)], itertools.repeat(str(clonotype_cluster)))
            for ct_id, clonotype_cluster in enumerate(part.membership)
        ),
        strict=False,
    )
    clonotype_cluster_series = pd.Series(values, index=idx).reindex(params.adata.obs_names)
    clonotype_cluster_size_series = clonotype_cluster_series.groupby(clonotype_cluster_series).transform("count")

    # Return or store results
    clonotype_distance_res = {
        "distances": clonotype_dist,
        "cell_indices": json.dumps(ctn.cell_indices),
    }
    if inplace:
        params.set_obs(key_added, clonotype_cluster_series)
        params.set_obs(key_added + "_size", clonotype_cluster_size_series)
        params.adata.uns[key_added] = clonotype_distance_res
    else:
        return (
            clonotype_cluster_series,
            clonotype_cluster_size_series,
            clonotype_distance_res,
        )


@DataHandler.inject_param_docs(
    common_doc=_common_doc,
    within_group=_common_doc_within_group,
    clonotype_definition=_doc_clonotype_definition,
    return_values=_common_doc_return_values,
    paralellism=_common_doc_parallelism,
)
def define_clonotypes(
    adata: DataHandler.TYPE,
    *,
    key_added: str = "clone_id",
    distance_key: str | None = None,
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    **kwargs,
) -> tuple[pd.Series, pd.Series, dict] | None:
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
    {adata}
    {common_doc}
    {within_group}
    key_added
        The column name under which the clonotype clusters and cluster sizes
        will be stored in `adata.obs` and under which the clonotype network will be
        stored in `adata.uns`
    inplace
        If `True`, adds the results to anndata, otherwise return them.
    {paralellism}
    {airr_mod}
    {airr_key}
    {chain_idx_key}

    {return_values}

    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    if distance_key is None and "ir_dist_nt_identity" not in params.adata.uns:
        # For the case of "clonotypes" we want to compute the distance automatically
        # if it doesn't exist yet. Since it's just a sparse ID matrix, this
        # should be instant.
        logging.info("ir_dist for sequence='nt' and metric='identity' not found. Computing with default parameters.")  # type: ignore
        ir_dist(params, metric="identity", sequence="nt", key_added=distance_key)

    return define_clonotype_clusters(
        params,
        key_added=key_added,
        sequence="nt",
        metric="identity",
        partitions="connected",
        **kwargs,
    )


@DataHandler.inject_param_docs(clonotype_network=_doc_clonotype_network)
def clonotype_network(
    adata: DataHandler.TYPE,
    *,
    sequence: Literal["aa", "nt"] = "nt",
    metric: MetricType = "identity",
    min_cells: int = 1,
    min_nodes: int = 1,
    layout: str = "components",
    size_aware: bool = True,
    base_size: float | None = None,
    size_power: float = 1,
    layout_kwargs: dict | None = None,
    clonotype_key: str | None = None,
    key_added: str = "clonotype_network",
    inplace: bool = True,
    random_state=42,
    airr_mod="airr",
    mask_obs: np.ndarray[np.bool_] | str | None = None,
) -> None | pd.DataFrame:
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
    {adata}
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
    {airr_mod}
    mask_obs
        Boolean mask or the name of the column in anndata.obs that contains the boolean mask to select cells to filter the clonotype clusters that should be displayed
        in the graph. Only connected modules in the clonotype distance graph that contain at least one of these cells will be shown.
        Can be set to None to avoid filtering.

    Returns
    -------
    Depending on the value of `inplace` returns either nothing or the computed
    coordinates.
    """
    params = DataHandler(adata, airr_mod)
    if size_aware and layout != "components":
        raise ValueError("The `size_aware` option is only compatible with the `components` layout.")
    params_dict = {}
    random.seed(random_state)
    np.random.seed(random_state)

    if clonotype_key is None:
        if metric == "identity" and sequence == "nt":
            clonotype_key = "clone_id"
        else:
            clonotype_key = f"cc_{sequence}_{metric}"

    try:
        clonotype_res = params.adata.uns[clonotype_key]
    except KeyError:
        raise ValueError(
            "Connectivity data not found. Did you run `tl.define_clonotypes` "
            "or `tl.define_clonotype_clusters`, respectively?"
        ) from None

    graph = igraph_from_sparse_matrix(clonotype_res["distances"], matrix_type="distance")

    if base_size is None:
        base_size = 240000 / len(graph.vs)

    # explicitly annotate node ids to keep them after subsetting
    graph.vs["node_id"] = np.arange(0, len(graph.vs))

    cell_indices = read_cell_indices(clonotype_res["cell_indices"])

    # store size in graph to be accessed by layout algorithms
    clonotype_size = np.array([len(idx) for idx in cell_indices.values()])
    graph.vs["size"] = clonotype_size

    # create clonotype_mask for filtering according to mask_obs
    if mask_obs is not None:
        if isinstance(mask_obs, str):
            cell_mask = adata.obs[mask_obs]
        elif isinstance(mask_obs, np.ndarray) and mask_obs.dtype == np.bool_:
            cell_mask = mask_obs
        else:
            raise TypeError(f"mask_obs should be either a string or a boolean NumPy array, but got {type(mask_obs)}")

        cell_indices_reversed = {v: k for k, values in cell_indices.items() for v in values}
        clonotype_mask = np.zeros((len(cell_indices),), dtype=bool)
        cell_index_filter = adata.obs.loc[cell_mask].index
        for cell_index in cell_index_filter:
            if cell_index in cell_indices_reversed:
                clonotype_mask_index = int(cell_indices_reversed[cell_index])
                clonotype_mask[clonotype_mask_index] = True
        graph.vs["clonotype_mask"] = clonotype_mask

    # decompose graph
    components = np.array(graph.decompose("weak"))

    # create component_mask
    component_node_count = np.array([len(component.vs) for component in components])
    component_sizes = np.array([sum(component.vs["size"]) for component in components])
    component_mask = (component_node_count >= min_nodes) & (component_sizes >= min_cells)

    # adapt component_mask according to clonotype_mask
    if mask_obs is not None:
        component_filter = np.array([any(component.vs["clonotype_mask"]) for component in components])
        component_mask = component_mask & component_filter

    # Filter subgraph by `min_cells` and `min_nodes`
    subgraph_idx = list(itertools.chain.from_iterable(comp.vs["node_id"] for comp in components[component_mask]))

    if len(subgraph_idx) == 0:
        raise ValueError(f"No subgraphs with size >= {min_cells} found.")
    graph = graph.subgraph(subgraph_idx)

    # Compute layout
    if layout_kwargs is None:
        layout_kwargs = {}
    if layout == "components":
        tmp_layout_kwargs = {}
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
        tmp_layout_kwargs = {"weights": "weight"} if layout == "fr" else {}
        tmp_layout_kwargs.update(layout_kwargs)
        coords = graph.layout(layout, **tmp_layout_kwargs).coords

    # Expand to cell coordinates to store in adata.obsm
    idx, coords = zip(
        *itertools.chain.from_iterable(
            zip(cell_indices[str(node_id)], itertools.repeat(coord))
            for node_id, coord in zip(graph.vs["node_id"], coords, strict=False)  # type: ignore
        ),
        strict=False,
    )
    coord_df = pd.DataFrame(data=coords, index=idx, columns=["x", "y"]).reindex(params.adata.obs_names)

    # Store results or return
    if inplace:
        params.adata.obsm[f"X_{key_added}"] = coord_df
        params_dict["clonotype_key"] = clonotype_key
        params_dict["base_size"] = base_size
        params_dict["size_power"] = size_power
        params.adata.uns[key_added] = params_dict
    else:
        return coord_df


def _graph_from_coordinates(adata: AnnData, clonotype_key: str, basis: str) -> tuple[pd.DataFrame, sp.csr_matrix]:
    """
    Given an AnnData object on which `tl.clonotype_network` was ran, and
    the corresponding `clonotype_key`, extract a data-frame
    with x and y coordinates for each node, and an aligned adjacency matrix.

    Combined, it can be used for plotting the layouted graph with igraph or networkx.
    """
    clonotype_res = adata.uns[clonotype_key]
    # map the cell-id to the corresponding row/col in the clonotype distance matrix
    cell_indices = read_cell_indices(clonotype_res["cell_indices"])
    dist_idx, obs_names = zip(
        *itertools.chain.from_iterable(zip(itertools.repeat(i), obs_names) for i, obs_names in cell_indices.items()),
        strict=False,
    )
    dist_idx_lookup = pd.DataFrame(index=obs_names, data=dist_idx, columns=["dist_idx"])
    clonotype_label_lookup = adata.obs.loc[:, [clonotype_key]].rename(columns={clonotype_key: "label"})

    # Retrieve coordinates and reduce them to one coordinate per node
    coords = (
        cast(pd.DataFrame, adata.obsm[f"X_{basis}"])
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


@DataHandler.inject_param_docs()
def clonotype_network_igraph(
    adata: DataHandler.TYPE,
    basis="clonotype_network",
    airr_mod="airr",
) -> tuple[ig.Graph, ig.Layout]:
    """
    Get an `igraph` object representing the clonotype network.

    Requires running :func:`scirpy.tl.clonotype_network` before, to
    compute the layout.

    Parameters
    ----------
    {adata}
    basis
        Key in `adata.obsm` where the network layout is stored.
    {airr_mod}

    Returns
    -------
    graph
        igraph object
    layout
        corresponding igraph Layout object.
    """
    from scirpy.util.graph import igraph_from_sparse_matrix

    params = DataHandler(adata, airr_mod)

    try:
        clonotype_key = params.adata.uns[basis]["clonotype_key"]
    except KeyError:
        raise KeyError(f"{basis} not found in `adata.uns`. Did you run `tl.clonotype_network`?") from None
    if f"X_{basis}" not in params.adata.obsm_keys():
        raise KeyError(f"X_{basis} not found in `adata.obsm`. Did you run `tl.clonotype_network`?")
    coords, adj_mat = _graph_from_coordinates(params.adata, clonotype_key, basis)

    graph = igraph_from_sparse_matrix(adj_mat, matrix_type="distance")
    # flip y axis to be consistent with networkx
    coords["y"] = np.max(coords["y"]) - coords["y"]
    layout = ig.Layout(coords=coords.loc[:, ["x", "y"]].values.tolist())
    return graph, layout
