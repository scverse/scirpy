from anndata import AnnData
from typing import Union
import numpy as np
import pandas as pd
from .._util import _is_na, _which_fractions


def group_abundance(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    for_cells: Union[None, list, np.ndarray] = None,
    fraction: Union[None, str, bool] = None,
    fraction_base: Union[None, str] = None,
    inplace: bool = True,
    as_dict: bool = False,
) -> pd.DataFrame:
    """Creates summary statsitics on how many
    cells belong to each clonotype within a certain sample. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.  
    target_col
        Column on which to compute the expansion.        
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified,
        cells with NaN values in the group definition columns will be ignored. 
        When the tool is executed by the plotting function, 
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
    if target_col not in adata.obs.columns:
        raise ValueError(
            "`target_col` not found in obs. Did you run `tl.define_clonotypes`?"
        )

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, fraction_base, groupby)

    # Preproces the data table (remove unnecessary rows and columns)
    if (for_cells is None) or (len(for_cells) < 2):
        for_cells = np.intersect1d(
            adata.obs.loc[~_is_na(adata.obs[fraction_base])].index.values,
            adata.obs.loc[~_is_na(adata.obs[groupby])].index.values,
        )
    tcr_obs = adata.obs.loc[
        for_cells, pd.unique([groupby, fraction_base, target_col]).tolist()
    ]
    tcr_obs.groupby(
        pd.unique([groupby, fraction_base, target_col]).tolist()
    ).size().reset_index(name="count")

    # Compute group sizes as a basis of fractions
    group_sizes = tcr_obs.loc[:, fraction_base].value_counts().to_dict()

    # Calculate clonotype abundance
    clonotype_counts = (
        tcr_obs.groupby(pd.unique([groupby, fraction_base, target_col]).tolist())
        .size()
        .reset_index(name="count")
    )
    clonotype_counts["groupsize"] = (
        clonotype_counts[fraction_base].map(group_sizes).astype("int32")
    )
    if fraction:
        clonotype_counts["count"] /= clonotype_counts["groupsize"]
    clonotype_counts = (
        clonotype_counts.groupby([groupby, target_col]).sum().reset_index()
    )

    # Calculate the frequency table already here and maybe save a little time
    #  for plotting by supplying wide format data
    result_df = clonotype_counts.pivot(
        index=target_col, columns=groupby, values="count"
    ).fillna(value=0.0)

    # By default, the most abundant clonotype should be the first on the plot,
    # therefore we need their order
    ranked_clonotypes = (
        clonotype_counts.groupby([target_col])
        .sum()
        .sort_values(by="count", ascending=False)
        .index.values
    )

    # Sort the groups as well
    ranked_groups = (
        result_df.apply(np.sum, axis=0).sort_values(ascending=False).index.values
    )
    result_df = result_df.loc[ranked_clonotypes, ranked_groups]

    return result_df
