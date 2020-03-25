
from anndata import AnnData
from typing import Union, List
import pandas as pd


def vdj_usage(
    adata: AnnData,
    *,
    target_cols: Collection = (
        "TRA_1_j_gene",
        "TRA_1_v_gene",
        "TRB_1_v_gene",
        "TRB_1_d_gene",
        "TRB_1_j_gene",
    ),
    for_cells: Union[None, list, np.ndarray, pd.Series] = None,
    cell_weights: Union[None, str, list, np.ndarray, pd.Series] = None,
    fraction_base: Union[None, str] = None,
    as_dict: bool = False,
) -> Union[AnnData, dict]:
    """Gives a summary of the most abundant VDJ combinations in a given subset of cells. 

    Currently works with primary alpha and beta chains only.
    Does not add the result to `adata`!
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    target_cols
        Columns containing gene segment information. Overwrite default only if you know what you are doing!         
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified,
        all cells in  `adata` will be used that have at least a primary alpha or beta chain.
    cell_weights
        A size factor for each cell. By default, each cell count as 1, but due to normalization
        to different sample sizes for example, it is possible that one cell in a small sample
        is weighted more than a cell in a large sample.
    size_column
        The name of the column that will be used for storing cell weights. This value is used internally
        and should be matched with the column name used by the plotting function. Best left untouched.
    fraction_base
        As an alternative to supplying ready-made cell weights, this feature can also be calculated
        on the fly if a grouping column name is supplied. The parameter `cell_weights` takes piority
        over `fraction_base`. If both is `None`, each cell will have a weight of 1.
    as_dict
        If True, returns a dictionary instead of a dataframe. Useful for testing.

    Returns
    -------
    Depending on the value of `as_dict`, either returns a data frame  or a dictionary. 
    """

    # Preproces the data table (remove unnecessary rows and columns)
    size_column = 'cell_weights'
    if for_cells is None:
        for_cells = adata.obs.loc[
            ~_is_na(adata.obs.loc[:, target_cols]).all(axis="columns"), target_cols
        ].index.values
    observations = adata.obs.loc[for_cells, :]

    # Check how cells should be weighted
    makefractions = False
    if cell_weights is None:
        if fraction_base is None:
            observations[size_column] = 1
        else:
            makefractions = True
    else:
        if isinstance(cell_weights, str):
            makefractions = True
            fraction_base = cell_weights
        else:
            if len(cell_weights) == len(for_cells):
                observations[size_column] = cell_weights
            else:
                raise ValueError(
                    "Although `cell_weights` appears to be a list, its length is not identical to the number of cells specified by `for_cells`."
                )

    # Calculate fractions if necessary
    if makefractions:
        group_sizes = observations.loc[:, fraction_base].value_counts().to_dict()
        observations[size_column] = (
            observations[fraction_base].map(group_sizes).astype("int32")
        )
        observations[size_column] = 1 / observations[size_column]

    # Return the requested format
    if as_dict:
        observations = observations.to_dict(orient="index")

    return observations
