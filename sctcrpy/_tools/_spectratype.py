from anndata import AnnData
from typing import Callable, Union, Collection
import numpy as np
import pandas as pd
from .._util import _which_fractions, _is_na


def spectratype(
    adata: AnnData,
    groupby: str,
    *,
    target_col: Collection = ("TRB_1_cdr3_len",),
    fun: Callable = np.sum,
    for_cells: Union[None, list, np.ndarray] = None,
    fraction: Union[None, str, bool] = None,
    fraction_base: Union[None, str] = None,
    inplace: bool = True,
    as_dict: bool = False,
) -> pd.DataFrame:
    """Show the distribution of CDR3 region lengths. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    fun
        A function definining how the target columns should be merged 
        (e.g. sum, mean, median, etc).  
    target_col
        Columns containing CDR3 lengths.        
    for_cells
        A whitelist of cells that should be included in the analysis. 
        If not specified, cells with NaN values in the group definition columns 
        will be ignored. When the tool is executed by the plotting function,
         the whitelist is not updated.         
    fraction
        If True, compute fractions of expanded clonotypes rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name 
        of a grouping (e.g. samples). 
    fraction_base
        Sets the column used as a bsis for fraction calculation explicitely.
        Overrides the column set by `fraction`, but gets
        ignored if `fraction` is `False`. 
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    as_dict
        If True, returns a dictionary instead of a dataframe. Useful for testing.

    Returns
    -------
    Depending on the value of `inplace`, either returns a data frame 
    or adds it to `adata.uns`. 
    """
    if len(np.intersect1d(adata.obs.columns, target_col)) < 1:
        raise ValueError(
            "`target_col` not found in obs. Where do you store CDR3 length information?"
        )

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, fraction_base, groupby)
    target_col = pd.unique(target_col).tolist()

    # Preproces the data table (remove unnecessary columns and rows)
    if (for_cells is None) or (len(for_cells) < 2):
        for_cells = np.intersect1d(
            adata.obs.loc[~_is_na(adata.obs[fraction_base])].index.values,
            adata.obs.loc[~_is_na(adata.obs[groupby])].index.values,
        )
    tcr_obs = adata.obs.loc[
        for_cells, pd.unique([groupby, fraction_base] + target_col).tolist()
    ]

    # Merge target columns into one single column applying the desired function rowwise
    tcr_obs["lengths"] = tcr_obs.loc[:, target_col].apply(fun, axis=1)

    # Compute group sizes as a basis of fractions)
    group_sizes = tcr_obs.loc[:, fraction_base].value_counts().to_dict()

    # Calculate distribution of lengths in each group
    cdr3_lengths = (
        tcr_obs.groupby(pd.unique([groupby, fraction_base, "lengths"]).tolist())
        .size()
        .reset_index(name="count")
    )
    cdr3_lengths["groupsize"] = (
        cdr3_lengths[fraction_base].map(group_sizes).astype("int32")
    )
    if fraction:
        cdr3_lengths["count"] /= cdr3_lengths["groupsize"]
    cdr3_lengths = cdr3_lengths.groupby([groupby, "lengths"]).sum().reset_index()
    cdr3_lengths = cdr3_lengths.pivot(index="lengths", columns=groupby, values="count")
    cdr3_lengths = cdr3_lengths.loc[range(int(tcr_obs["lengths"].max()) + 1), :].fillna(
        value=0.0
    )

    # By default, the most abundant group should be the first on the plot,
    # therefore we need their order
    cdr3_lengths[cdr3_lengths.apply(np.sum, axis=0).index.values]

    return cdr3_lengths
