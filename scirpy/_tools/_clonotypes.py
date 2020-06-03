from anndata import AnnData
import igraph as ig
from .._compat import Literal
from typing import Union, Tuple
from ..util import _is_na, _doc_params
from ..util.graph import _get_igraph_from_adjacency, layout_components
import numpy as np
import pandas as pd
import random

_define_clonotypes_doc = """\
same_v_gene
    Enforces clonotypes to have the same :term:`V-genes<V(D)J>`. This is useful
    as the CDR1 and CDR2 regions are fully encoded in this gene. 
    See :term:`CDR` for more details.
    
    Possible values are 

        * `False` - Ignore V-gene during clonotype definition
        * `"primary_only"` - Only the V-genes of the primary pair of alpha
          and beta chains needs to match
        * `"all"` - All V-genes of all sequences need to match. 

    Chains with no detected V-gene will be treated like a separate "gene" with 
    the name "None".  
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
neighbors_key
    Key under which the neighboorhood graph is stored in `adata.uns`.
    By default, tries to read from `tcr_neighbors_{sequence}_{metric}`, 
    e.g. `tcr_neighbors_nt_identity`. 
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


@_doc_params(common_doc=_define_clonotypes_doc)
def define_clonotype_clusters(
    adata,
    *,
    sequence: Literal["aa", "nt"] = "aa",
    metric: Literal["alignment", "levenshtein", "identity"] = "identity",
    key_added: Union[str, None] = None,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """\
    Define :term:`clonotype clusters<Clonotype cluster>` based on :term:`CDR3` distance.

    As opposed to :func:`~scirpy.tl.define_clonotypes` which employs 
    a more stringent definition of :term:`clonotypes <Clonotype>`,
    this function flexibly defines clonotype clusters based on amino acid or nucleic
    acid sequence identity or similarity. 

    Requires running :func:`scirpy.pp.tcr_neighbors` first with the same 
    `sequence` and `metric` values first. 

    Parameters
    ----------
    adata
        Annotated data matrix
    sequence
        The sequence parameter used when running :func:scirpy.pp.tcr_neighbors`
    metric
        The metric parameter used when running :func:`scirpy.pp.tcr_neighbors`
    key_added
        Name of the columns which will be added to `adata.obs` if inplace is `True`. 
        Will create the columns `{{key_added}}` and `{{key_added}}_size`. 

        Defaults to `ct_cluster_{{sequence}}_{{metric}}` and `ct_cluster_{{sequence}}_{{metric}}_size`. 
    {common_doc}  
    """
    if key_added is None:
        key_added = f"ct_cluster_{sequence}_{metric}"
    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = f"tcr_neighbors_{sequence}_{metric}"
    return _define_clonotypes(adata, key_added=key_added, **kwargs)


@_doc_params(common_doc=_define_clonotypes_doc)
def define_clonotypes(
    adata, *, key_added="clonotype", **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """\
    Define :term:`clonotypes <Clonotype>` based on :term:`CDR3` nucleic acid
    sequence identity.

    As opposed to :func:`~scirpy.tl.define_clonotype_clusters` which employs 
    a more flexible definition of :term:`clonotype clusters <Clonotype cluster>`,
    this function stringently defines clonotypes based on nucleic acid sequence 
    identity. Technically, this function is an alias to :func:`~scirpy.tl.define_clonotype_clusters`
    with different default parameters.  

    Requires running :func:`scirpy.pp.tcr_neighbors` with `sequence='nt'` and 
    `metric='identity` first (which are the default parameters).  

    Parameters
    ----------
    adata
        Annotated data matrix
    key_added
        Name of the columns which will be added to `adata.obs` if inplace is `True`.
        Will create the columns `{{key_added}}` and `{{key_added}}_size`.  
    {common_doc}
    """
    if "neighbors_key" not in kwargs:
        kwargs["neighbors_key"] = "tcr_neighbors_nt_identity"
    return _define_clonotypes(adata, key_added=key_added, **kwargs)


def _define_clonotypes(
    adata,
    *,
    same_v_gene: Union[bool, Literal["primary_only", "all"]] = False,
    partitions: Literal["connected", "leiden"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    neighbors_key: str = "tcr_neighbors",
    key_added: str = "clonotype",
    inplace: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    try:
        conn = adata.uns[neighbors_key]["connectivities"]
    except KeyError:
        raise ValueError(
            "Connectivities were not found. Did you run `pp.tcr_neighbors`?"
        )
    g = _get_igraph_from_adjacency(conn)

    if partitions == "leiden":
        part = g.community_leiden(
            objective_function="modularity",
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )
    else:
        part = g.clusters(mode="weak")

    if same_v_gene is False:
        clonotype = np.array([str(x) for x in part.membership])
    elif same_v_gene == "primary_only":
        clonotype = np.array(
            [
                f"{x}_{tra1_v_gene}_{trb1_v_gene}"
                for x, tra1_v_gene, trb1_v_gene in zip(
                    part.membership,
                    adata.obs["TRA_1_v_gene"],
                    adata.obs["TRB_1_v_gene"],
                )
            ]
        )
    elif same_v_gene == "all":
        clonotype = np.array(
            [
                f"{x}_{tra1_v_gene}_{trb1_v_gene}_{tra2_v_gene}_{trb2_v_gene}"
                for x, tra1_v_gene, trb1_v_gene, tra2_v_gene, trb2_v_gene in zip(
                    part.membership,
                    adata.obs["TRA_1_v_gene"],
                    adata.obs["TRB_1_v_gene"],
                    adata.obs["TRA_2_v_gene"],
                    adata.obs["TRB_2_v_gene"],
                )
            ]
        )
    else:
        raise ValueError("Invalud value for `same_v_gene`.")

    clonotype_size = pd.Series(clonotype).groupby(clonotype).transform("count").values
    assert len(clonotype) == len(clonotype_size) == adata.obs.shape[0]

    if not inplace:
        return clonotype, clonotype_size
    else:
        adata.obs[key_added] = clonotype
        adata.obs[key_added + "_size"] = clonotype_size


def clonotype_network(
    adata,
    *,
    sequence: Literal["aa", "nt"] = "nt",
    metric: Literal["identity", "alignment", "levenshtein", "custom"] = "identity",
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

    Requires running :func:`scirpy.pp.tcr_neighbors` first. 

    Stores coordinates of the clonotype network in `adata.obsm`. 
    
    Parameters
    ----------
    sequence
        The `sequence` parameter :func:`scirpy.pp.tcr_neighbors` was ran with. 
    metric
        The `metric` parameter :func:`scirpy.pp.tcr_neighbors` was ran with. 
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
        Defaults to `tcr_neighbors_{sequence}_{metric}`. 
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
        neighbors_key = f"tcr_neighbors_{sequence}_{metric}"
    if key_clonotype_size is None:
        if sequence == "nt" and metric == "identity":
            key_clonotype_size = "clonotype_size"
        else:
            key_clonotype_size = f"ct_cluster_{sequence}_{metric}_size"
    random.seed(random_state)
    try:
        conn = adata.uns[neighbors_key]["connectivities"]
    except KeyError:
        raise ValueError("Connectivity data not found. Did you run `pp.tcr_neighbors`?")

    try:
        clonotype_size = adata.obs[key_clonotype_size].values
    except KeyError:
        raise ValueError(
            "Clonotype size information not found. Did you run `tl.define_clonotypes`?"
        )

    if not adata.n_obs == conn.shape[0] == conn.shape[0]:
        raise ValueError(
            "Dimensions of connectivity matrix and AnnData do not match. Maybe you "
            "need to re-run `pp.tcr_neighbors?"
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
    adata, basis="clonotype_network"
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


def _define_clonotypes_no_graph(
    adata: AnnData,
    *,
    flavor: Literal["all_chains", "primary_only"] = "all_chains",
    inplace: bool = True,
    key_added: str = "clonotype",
) -> Union[None, np.ndarray]:
    """Old version of clonotype definition that works without graphs.

    The current definition of a clonotype is
    same CDR3 sequence for both primary and secondary
    TRA and TRB chains. If all chains are `NaN`, the clonotype will
    be `NaN` as well. 

    Parameters
    ----------
    adata
        Annotated data matrix
    flavor
        Biological model to define clonotypes. 
        `all_chains`: All four chains of a cell in a clonotype need to be the same. 
        `primary_only`: Only primary alpha and beta chain need to be the same. 
    inplace
        If True, adds a column to adata.obs
    key_added
        Column name to add to 'obs'

    Returns
    -------
    Depending on the value of `inplace`, either
    returns a Series with a clonotype for each cell 
    or adds a `clonotype` column to `adata`. 
    
    """
    groupby_cols = {
        "all_chains": ["TRA_1_cdr3", "TRB_1_cdr3", "TRA_2_cdr3", "TRA_2_cdr3"],
        "primary_only": ["TRA_1_cdr3", "TRB_1_cdr3"],
    }
    clonotype_col = np.array(
        [
            "clonotype_{}".format(x)
            for x in adata.obs.groupby(groupby_cols[flavor], observed=True).ngroup()
        ]
    )
    clonotype_col[
        _is_na(adata.obs["TRA_1_cdr3"])
        & _is_na(adata.obs["TRA_2_cdr3"])
        & _is_na(adata.obs["TRB_1_cdr3"])
        & _is_na(adata.obs["TRB_2_cdr3"])
    ] = np.nan

    if inplace:
        adata.obs[key_added] = clonotype_col
    else:
        return clonotype_col
