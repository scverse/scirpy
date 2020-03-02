from anndata import AnnData
from typing import Union
import numpy as np
from .._util import _which_fractions, _is_na
from typing import Dict


def cdr_convergence(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "TRB_1_cdr3",
    key_added: Union[None, str] = None,
    for_cells: Union[None, list, np.ndarray] = None,
    fraction: Union[None, str, bool] = None,
    fraction_base: Union[None, str] = None,
    clip_at: int = 3,
    inplace: bool = True,
    as_dict: bool = False,
) -> Dict:
    """Creates summary statsitics on how many nucleotide versions
    a single CDR3 amino acid sequence typically has in a given group 
    of cells. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. E.g. samples or diagnosis.
    target_col
        Column on which to compute the expansion. Useful if we want to
        specify the chain.        
    key_added
        Manually specified name for the column where convergence information 
        should be added.
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified,
        cells with NaN values in the group definition columns will be ignored.
        When the tool is executed by the plotting function, the
        whitelist is not updated.          
    clip_at
        All clonotypes with more copies than `clip_at` will be summarized into 
        a single group. 
    fraction
        If True, compute fractions cells rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name 
        of a grouping (e.g. samples). 
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    as_dict
        If True, returns a dictionary instead of a dataframe. Useful for testing.

    Returns
    -------
    Depending on the value of `inplace`, either returns a data frame 
    or adds it to `adata.uns`. Also adds a column to `obs` with the name 
    convergence_`target_col`_`groupby` or the name specified by key_added.
    """
    if target_col not in adata.obs.columns:
        raise ValueError(
            "`target_col` not found in obs. Where do you store"
            " CDR3 amino acid sequence information?"
        )

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, None, groupby)

    # Preprocess data
    tcr_obs = adata.obs.loc[~_is_na(adata.obs[target_col]), :]
    result_df = (
        tcr_obs.groupby([groupby, target_col, target_col + "_nt"])
        .size()
        .reset_index(name="count")
    )
    result_df = (
        result_df.groupby([groupby, target_col]).size().reset_index(name="count")
    )
    result_df.loc[result_df["count"] >= clip_at, "count"] = clip_at
    map_df = result_df.loc[:, [groupby, target_col, "count"]]
    map_df.index = map_df.apply(
        lambda x: str(x[target_col]) + "_" + str(x[groupby]), axis=1
    )
    map_df = map_df["count"].to_dict()

    # Clip and make wide type table
    result_dict = dict()
    for group in result_df[groupby].unique():
        result_dict[group] = dict()
        for n in range(1, clip_at + 1):
            label = ">= {}".format(n) if n == clip_at else str(n)
            mask_group = result_df[groupby] == group
            mask_count = result_df["count"] == n
            tmp_count = np.sum(mask_group & mask_count)
            if fraction:
                tmp_count /= np.sum(mask_group)
            result_dict[group][label] = tmp_count

    # Add a column to `obs` that is basically the fraction of cells
    # having more than two nucleotide versions of the CDR3
    if key_added is None:
        key_added = "convergence_" + target_col + "_" + groupby
    adata.obs[key_added] = adata.obs.apply(
        lambda x: str(x[target_col]) + "_" + str(x[groupby]), axis=1
    )
    adata.obs[key_added] = adata.obs[key_added].map(map_df)

    return result_dict
