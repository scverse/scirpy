from anndata import AnnData
from .._compat import Literal
from typing import Union, Tuple
from .._util import _is_na, get_igraph_from_adjacency
import numpy as np
import pandas as pd


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
            for x in adata.obs.groupby(groupby_cols[flavor]).ngroup()
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
    partitions: Literal["connected", "leiden"] = "leiden",
    resolution: float = 1,
    n_iterations: int = 5,
    neighbors_key: str = "neighbors",
    key_added: str = "clonotype",
    inplace: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Define clonotypes based on cdr3 distance.
    
    Parameters
    ----------
    adata
        annotated data matrix
    partitions
        How to find graph partitions that define a clonotype. 
        Possible values are 'leiden', for using the "Leiden" algorithm and 
        "connected" to find fully connected sub-graphs. 

        The difference is that the Leiden algorithm further divides 
        fully connected subgraphs into highly-connected modules. 
    resolution
        resolution parameter for the leiden algorithm. 
    n_iterations
        n_iterations parameter for the leiden algorithm. 
    neighbors_key
        key under which the neighboorhood graph is stored in adata
    key_added
        name of the columns that will be added to `adata.obs` if inplace is True. 
        Will create the columns `{key_added}` and `{key_added}_size`. 
    inplace
        If true, adds the results to anndata, otherwise returns them. 

    Returns
    -------
    clonotype
        an array containing the clonotype id for each cell
    clonotype_size
        an array containing the number of cells in the respective clonotype
        for each cell.    
    """
    try:
        conn = adata.uns["sctcrpy"][neighbors_key]["connectivities"]
    except KeyError:
        raise ValueError(
            "Connectivities were not found. Did you run `pp.tcr_neighbors`?"
        )
    g = get_igraph_from_adjacency(conn)

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
    layout: str = "fr",
    min_size: int = 1,
    neighbors_key: str = "neighbors",
    key_clonotype_size: str = "clonotype_size",
    key_added: str = "X_clonotype_network",
    inplace: bool = False,
):
    """Build the clonotype network for plotting
    
    Parameters
    ----------
    min_size
        Only show clonotypes with at least `min_size` cells.
    """
    try:
        conn = adata.uns["sctcrpy"][neighbors_key]["connectivities"]
    except KeyError:
        raise ValueError("Connectivity data not found. Did you run `pp.tcr_neighbors`?")

    try:
        clonotype_size = adata.obs[key_clonotype_size].values
    except KeyError:
        raise ValueError(
            "Clonotype size information not found. Did you run `tl.define_clonotypes`?"
        )

    graph = get_igraph_from_adjacency(conn)

    # remove singletons/small subgraphs
    subgraph_idx = np.where(clonotype_size >= min_size)[0]
    if len(subgraph_idx) == 0:
        raise ValueError("No subgraphs with size >= {} found.".format(min_size))

    graph = graph.subgraph(subgraph_idx)
    layout_ = graph.layout(layout)
    coordinates = np.full((adata.n_obs, 2), fill_value=np.nan)
    coordinates[subgraph_idx, :] = layout_.coords
    adata.obsm[key_added] = coordinates
