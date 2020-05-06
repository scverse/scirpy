from anndata import AnnData
from .._compat import Literal
from typing import Union, Tuple
from ..util import _is_na
from ..util.graph import _get_igraph_from_adjacency, layout_components
import numpy as np
import pandas as pd
import random


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


def define_clonotypes(
    adata,
    *,
    partitions: Literal["connected", "leiden"] = "connected",
    resolution: float = 1,
    n_iterations: int = 5,
    neighbors_key: str = "tcr_neighbors",
    key_added: str = "clonotype",
    inplace: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Define :term:`clonotypes <Clonotype>` based on :term:`CDR3` distance.

    Requires running :func:`scirpy.pp.tcr_neighbors` first. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
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
    key_added
        Name of the columns that will be added to `adata.obs` if inplace is `True`. 
        Will create the columns `{key_added}` and `{key_added}_size`. 
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

    clonotype = np.array([str(x) for x in part.membership])
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
    min_size: int = 1,
    layout: str = "components",
    layout_kwargs: Union[dict, None] = None,
    neighbors_key: str = "tcr_neighbors",
    key_clonotype_size: str = "clonotype_size",
    key_added: str = "X_clonotype_network",
    inplace: bool = True,
    random_state=42,
) -> Union[None, np.ndarray]:
    """Layouts the clonotype network for plotting. 

    Other than with transcriptomics data, this network usually consists 
    of many disconnected components, each of them representing cells 
    of the same clonotype. 

    Singleton clonotypes can be filtered out with the `min_size` parameter. 

    Stores coordinates of the clonotype network in `adata.obsm`. 
    
    Parameters
    ----------
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
    key_clonotype_size
        Key under which the clonotype size information is stored in `adata.obs`
    key_added
        Key under which the layout coordinates will be stored in `adata.obsm`. 
    inplace
        If `True`, store the coordinates in `adata.obsm`, otherwise return them. 
    random_state
        Random seed set before computing the layout. 

    Returns
    -------
    Depending on the value of `inplace` returns either nothing or the computed
    coordinates. 
    """
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
        adata.obsm[key_added] = coordinates
    else:
        return coordinates
