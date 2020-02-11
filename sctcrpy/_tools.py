import parasail
import numpy as np
from anndata import AnnData
import pandas as pd
from typing import Union, Callable
from ._util import _is_na, _add_to_uns, _which_fractions


def define_clonotypes(
    adata: AnnData, *, inplace: bool = True, key_added: str = "clonotype"
) -> Union[None, np.ndarray]:
    """Define clonotypes based on CDR3 region.

    The current definition of a clonotype is
    same CDR3 sequence for both primary and secondary
    TRA and TRB chains. If all chains are `NaN`, the clonotype will
    be `NaN` as well. 

    Parameters
    ----------
    adata
        Annotated data matrix
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
    clonotype_col = np.array(
        [
            "clonotype_{}".format(x)
            for x in adata.obs.groupby(
                ["TRA_1_cdr3", "TRB_1_cdr3", "TRA_2_cdr3", "TRA_2_cdr3"]
            ).ngroup()
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


def decide_chain_cat(x: pd.Series) -> str:
    """Helper function of `chain_pairing`. Associates categories to  a cell based on how many TRA and TRB chains they have.

    Parameters
    ----------
    x
        A Series contaning immune receptor chain information of a single cell.

    Returns
    -------
    The category the cell belongs to. 
    
    """
    s = "Uncategorized"
    if x["has_tcr"]:
        if x["multichain"] not in [False]:
            if x["TRA_1_cdr3"] not in [None, "None", "nan"]:
                if x["TRB_1_cdr3"] not in [None, "None", "nan"]:
                    if x["TRA_2_cdr3"] not in [None, "None", "nan"]:
                        if x["TRB_2_cdr3"] not in [None, "None", "nan"]:
                            s = "Two full chains"
                        else:
                            s = "Extra alpha"
                    else:
                        if x["TRB_2_cdr3"] not in [None, "None", "nan"]:
                            s = "Extra beta"
                        else:
                            s = "Single pair"
                else:
                    "Orphan alpha"
            else:
                if x["TRB_1_cdr3"] not in [None, "None", "nan"]:
                    "Orphan beta"
        else:
            s = "Multichain"
    else:
        s = "No TCR"
    return s


def chain_pairing(
    adata: AnnData, *, inplace: bool = True, key_added: str = "chain_pairing"
) -> Union[None, np.ndarray]:
    """Associate categories to cells based on how many TRA and TRB chains they have.

    Parameters
    ----------
    adata
        Annotated data matrix
    inplace
        If True, adds a column to adata.obs
    key_added
        Column name to add to 'obs'

    Returns
    -------
    Depending on the value of `inplace`, either
    returns a Series with a chain pairing category for each cell 
    or adds a `chain_pairing` column to `adata`. 
    
    """

    cp_col = adata.obs.loc[
        :,
        [
            "has_tcr",
            "multichain",
            "TRA_1_cdr3",
            "TRA_2_cdr3",
            "TRB_1_cdr3",
            "TRB_2_cdr3",
        ],
    ].apply(decide_chain_cat, axis=1)

    if inplace:
        adata.obs[key_added] = cp_col
    else:
        return cp_col


def tcr_dist(
    adata: AnnData,
    *,
    subst_mat=parasail.blosum62,
    gap_open: int = 8,
    gap_extend: int = 1,
    inplace: bool = True,
) -> Union[None, dict]:
    """Compute the TCRdist on CDR3 sequences. 

    Currently takes into account only dominant alpha and dominant beta. 

    High-performance sequence alignment through parasail library [Daily2016]_

    Parameters
    ----------
    adata
    subst_mat
    gap_open
    gap_extend
    """
    # TODO parallelize
    for chain in ["TRA", "TRB"]:
        col = "{}_1_cdr3".format(chain)
        unique_cdr3s = adata.obs.loc[_is_na(adata.obs[col]), col].unique()

        dist_mat = np.empty([len(unique_cdr3s)] * 2)

        for i, s1 in enumerate(unique_cdr3s):
            profile = parasail.profile_create_16(s1, subst_mat)
            for j, s2 in enumerate(unique_cdr3s[i + 1 :]):
                r = parasail.sw_striped_profile_16(profile, s2, gap_open, gap_extend)
                dist_mat[i, j] = r.score

        dist_df = pd.DataFrame(dist_mat)
        dist_df.index = dist_df.columns = unique_cdr3s

        # TODO work on each chain and implement inplace
        adata.uns["tcr_dist_alpha"] = dist_df


def alpha_diversity(
    adata: AnnData, groupby: str, *, target_col: str = "clonotype", inplace: bool = True
) -> Union[None, dict]:
    """Computes the alpha diversity of clonotypes within a group. 

    Ignores NaN values. 

    Parameters
    ----------
    adata
        Annotated data matrix 
    groupby 
        Column of `obs` by which the grouping will be performed. 
    target_col
        Column on which to compute the alpha diversity
    inplace
        If true, adds an entry to `adata.obs`. 
        Otherwise it returns a dictionary with the results. 

    Returns
    -------
    Depending on the value of `inplace`, either returns a dictionary 
    with the diversity for each group of adds the dict to 
    `adata.uns`. 
    """
    # Could rely on skbio.math if more variants are required.
    def _shannon_entropy(freq):
        np.testing.assert_almost_equal(np.sum(freq), 1)
        return -np.sum(freq * np.log2(freq))

    tcr_obs = adata.obs.loc[~_is_na(adata.obs[target_col]), :]
    clono_counts = (
        tcr_obs.groupby([groupby, target_col]).size().reset_index(name="count")
    )

    diversity = dict()
    for k in tcr_obs[groupby].unique():
        tmp_counts = clono_counts.loc[clono_counts[groupby] == k, "count"].values
        tmp_freqs = tmp_counts / np.sum(tmp_counts)
        diversity[k] = _shannon_entropy(tmp_freqs)

    if inplace:
        _add_to_uns(
            adata,
            "alpha_diversity",
            diversity,
            parameters={"groupby": groupby, "target_col": target_col},
        )
    else:
        return diversity


def clonal_expansion(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    clip_at: int = 3,
    inplace: bool = True,
    fraction: bool = True,
) -> Union[None, dict]:
    """Creates summary statsitics on how many
    clonotypes are expanded within a certain groups. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on
    groupby
        Group by this column from `obs`
    target_col
        Column on which to compute the expansion. 
    clip_at
        All clonotypes with more copies than `clip_at` will be summarized into 
        a single group.         
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    fraction
        If True, compute fractions of expanded clonotypes rather than reporting
        abosolute numbers

    Returns
    -------
    Depending on the value of `inplcae`, either returns a dictionary 
    or adds it to `adata.uns`. 
    """
    if target_col not in adata.obs.columns:
        raise ValueError(
            "`target_col` not found in obs. Did you run `tl.define_clonotypes`?"
        )
    # count abundance of each clonotype
    tcr_obs = adata.obs.loc[~_is_na(adata.obs[target_col]), :]
    clonotype_counts = (
        tcr_obs.groupby([groupby, target_col]).size().reset_index(name="count")
    )
    clonotype_counts.loc[clonotype_counts["count"] >= clip_at, "count"] = clip_at

    result_dict = dict()
    for group in clonotype_counts[groupby].unique():
        result_dict[group] = dict()
        for n in range(1, clip_at + 1):
            label = ">= {}".format(n) if n == clip_at else str(n)
            mask_group = clonotype_counts[groupby] == group
            mask_count = clonotype_counts["count"] == n
            tmp_count = np.sum(mask_group & mask_count)
            if fraction:
                tmp_count /= np.sum(mask_group)
            result_dict[group][label] = tmp_count

    if inplace:
        _add_to_uns(
            adata,
            "clonal_expansion",
            result_dict,
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
            },
        )
    else:
        return result_dict


def cdr_convergence(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "TRB_1_cdr3",
    for_cells: Union[None, list, np.ndarray] = None,
    fraction: Union[None, str, bool] = None,
    fraction_base: Union[None, str] = None,
    clip_at: int = 3,
    inplace: bool = True,
    as_dict: bool = False,
) -> Union[None, AnnData, dict]:
    """Creates summary statsitics on how many nucleotide versions a single CDR3 amino acid sequence typically has in a given group
    cells belong to each clonotype within a certain sample. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the expansion. Useful if we want to specify the chain.        
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified, cells with NaN values in the group definition columns will be ignored. When the tool is executed by the plotting function, the whitelist is not updated.          
    clip_at
        All clonotypes with more copies than `clip_at` will be summarized into 
        a single group. 
    fraction
        If True, compute fractions cells rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name of a grouping (e.g. samples). 
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
            "`target_col` not found in obs. Where do you store CDR3 amino acid sequence information??"
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
    result_df.loc[result_df["count"] >= clip_at, "count"] = clip_at

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

    if as_dict:
        result_df = result_dict
    else:
        result_df = pd.DataFrame.from_dict(result_dict, orient="index")

    # Pass on the resulting dataframe as requested
    if inplace:
        _add_to_uns(
            adata,
            "cdr_convergence",
            result_df,
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
                "fraction_base": fraction_base,
            },
        )
    else:
        return result_df


def spectratype(
    adata: AnnData,
    groupby: str,
    *,
    target_col: list = ["TRB_1_cdr3_len"],
    fun: Callable = np.sum,
    for_cells: Union[None, list, np.ndarray] = None,
    fraction: Union[None, str, bool] = None,
    fraction_base: Union[None, str] = None,
    inplace: bool = True,
    as_dict: bool = False,
) -> Union[None, AnnData, dict]:
    """Show the distribution of CDR3 region lengths. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    fun
        A function definining how the target columns should be merged (e.g. sum, mean, median, etc).  
    target_col
        Columns containing CDR3 lengths.        
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified, cells with NaN values in the group definition columns will be ignored. When the tool is executed by the plotting function, the whitelist is not updated.         
    fraction
        If True, compute fractions of expanded clonotypes rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name of a grouping (e.g. samples). 
    fraction_base
        Sets the column used as a bsis for fraction calculation explicitely. Overrides the column set by `fraction`, but gets ignored if `fraction` is `False`. 
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

    # By default, the most abundant group should be the first on the plot, therefore we need their order
    cdr3_lengths[cdr3_lengths.apply(np.sum, axis=0).index.values]

    if as_dict:
        cdr3_lengths = cdr3_lengths.to_dict(orient="index")

    # Pass on the resulting dataframe as requested
    if inplace:
        _add_to_uns(
            adata,
            "spectratype",
            cdr3_lengths,
            parameters={
                "groupby": groupby,
                "target_col": "|".join(target_col),
                "fraction": fraction,
                "fraction_base": fraction_base,
            },
        )
    else:
        return cdr3_lengths


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
) -> Union[None, AnnData, dict]:
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
        A whitelist of cells that should be included in the analysis. If not specified, cells with NaN values in the group definition columns will be ignored. When the tool is executed by the plotting function, the whitelist is not updated.         
    fraction
        If True, compute fractions of expanded clonotypes rather than reporting
        abosolute numbers. If a string is supplied, that should be the column name of a grouping (e.g. samples). 
    fraction_base
        Sets the column used as a bsis for fraction calculation explicitely. Overrides the column set by `fraction`, but gets ignored if `fraction` is `False`. 
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

    # Calculate the frequency table already here and maybe save a little time for plotting by supplying wide format data
    result_df = clonotype_counts.pivot(
        index=target_col, columns=groupby, values="count"
    ).fillna(value=0.0)

    # By default, the most abundant clonotype should be the first on the plot, therefore we need their order
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

    if as_dict:
        result_df = result_df.to_dict(orient="index")

    # Pass on the resulting dataframe as requested
    if inplace:
        _add_to_uns(
            adata,
            "group_abundance",
            result_df,
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
                "fraction_base": fraction_base,
            },
        )
    else:
        return result_df
