from .._util import _is_na
from anndata import AnnData
from typing import Collection, Union, List
import pandas as pd
import numpy as np


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
    fraction: Union[None, str, list, np.ndarray, pd.Series] = None,
    size_column: str = "cell_weights",
) -> pd.DataFrame:
    """Gives a summary of the most abundant VDJ combinations in a given subset of cells. 

    Currently works with primary alpha and beta chains only.
    Does not add the result to `adata`!
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    target_cols
        Columns containing gene segment information. Overwrite default only if you know what you are doing!         
    fraction
        Either the name of a categorical column that should be used as the base for computing fractions,
        or an iterable specifying a size factor for each cell. By default, each cell count as 1,
        but due to normalization to different sample sizes for example, it is possible that one cell
        in a small sample is weighted more than a cell in a large sample.
    size_column
        The name of the column that will be used for storing cell weights. This value is used internally
        and should be matched with the column name used by the plotting function. Best left untouched.

    Returns
    -------
    Depending on the value of `as_dict`, either returns a data frame  or a dictionary. 
    """

    # Check how cells should be weighted
    observations = adata.obs.copy()
    makefractions = False
    if cell_weights is None:
        if fraction is None:
            observations[size_column] = 1
        else:
            makefractions = True
    else:
        if isinstance(fraction, str):
            makefractions = True
            fraction_base = fraction
        else:
            if len(fraction) == observations.shape[0]:
                observations[size_column] = fraction
            else:
                raise ValueError(
                    "Although `fraction` appears to be an iterable, its length is not identical to the number of cells."
                )

    # Calculate fractions if necessary
    if makefractions:
        group_sizes = observations.loc[:, fraction_base].value_counts().to_dict()
        observations[size_column] = (
            observations[fraction_base].map(group_sizes).astype("int32")
        )
        observations[size_column] = 1 / observations[size_column]

    return observations
