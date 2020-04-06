from anndata import AnnData
from typing import Union, Sequence
from scipy.stats import fisher_exact
import pandas as pd
import numpy as np
from .._util import _is_na, _normalize_counts


def clonotype_imbalance(
    adata: AnnData,
    replicate_col: str,
    groupby: str,
    case_label: str
    *,
    control_label: Union[None, str] = None,
    target_col: str = "clonotype",
    additional_hue: Union[None, str, bool] = None,
    fraction: Union[None, str, bool] = None,
    inplace: bool = True,
    overlap_key: Union[None, str] = None,
    added_key: str = "clonotype_imbalance",
) -> Union[None, Sequence[pd.DataFrame, pd.DataFrame]]:
    """Aims to find clonotypes that are the most enriched or depleted in a category.

    Uses Fischer's exact test to rank clonotypes.
    Depends on execution of clonotype_overlap.
    Adds two dataframes (abundance of clonotypes per sample; pval and logFC for clonotypes) to `uns`
    
    Parameters
    ----------
    adata
        AnnData object to work on.       
    replicate_col
        Column with batch or sample labels.
    groupby
        The column containing categories that we want to compare and find imbalance between
    case_label
        The label in `groupby` column that we want to compare. 
    control_label
        The label in `groupby` column that we use as a baseline for comparison. If not set
        (None by default), all labels that are not equal to `case_label` make up the baseline. 
    target_col
        The clusters (clonotypes by default) that are imbalanced. 
    additional_hue
        An additional grouping factor. If the `case_label` was tumor for example, this could
        help make a distinction between imbalance in lung and colorectal tumors.
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    inplace
        Whether results should be added to `uns` or returned directly.
    overlap_key
        Under what key should the repertoire overlap results be looked up in `uns`.
        By default it is None to ensure that the overlap tool is executed with the right parameters.
    added_key
        Results will be added to `uns` under this key.


    Returns
    -------
    Two dataframes: abundance of clonotypes per sample; pval and logFC for clonotypes. 
    """

    # Retrieve clonotype presence matrix
    if added_key is None:
        clonotype_presence, dM, lM = tl.repertoire_overlap(
            adata,
            groupby=replicate_col,
            target_col=target_col,
            fraction=fraction,
            inplace=True,
        )
    else:
        if added_key not in adata.uns:
            tl.repertoire_overlap(
                adata,
                groupby=replicate_col,
                target_col=target_col,
                fraction=fraction,
                added_key=added_key
            )
        clonotype_presence = adata.uns[added_key]["weighted"]

    # Create a series of case-control groups for comparison
    case_control_groups = []
    group_cols = [groupby replicate_col]
    if additional_hue is None:
        hues = [None]
    else:
        group_cols.append(additional_hue)
        hues = adata.obs[additional_hue].unique()
    df = adata.obs.groupby(group_cols).agg('size').reset_index()

    for hue in hues:
        if hue is None:
            tdf = df
        else:
            tdf = df.loc[df[additional_hue] == hue, :]
        cases = tdf.loc[df[groupby] == target_label,:]
        ncase = cases[0]
        cases = cases[replicate_col]
        if control_label is None:
            controls = tdf.loc[df[groupby] != case_label, :]
        else:
            controls = tdf.loc[df[groupby] == control_label, :]
        ncontrol = controls[0]
        controls = controls[replicate_col]
        case_control_groups.append([hue, cases, controls, ncase, ncontrol])

    #  Compare groups with Fischer's test
    clt_freq, clt_stats = [], []
    if control_label is None:
        control_label = 'Background'
    for hue, cases, controls, ncase, ncontrol in case_control_groups:
        if hue is None:
            hue = 'All'
        tdf1 = clonotype_presence.loc[cases, ]
        tdf2 = clonotype_presence.loc[controls, ]
        suspects = set(tdf1.loc[:, tdf1.sum() > 0].columns.values.tolist() + tdf2.loc[:, tdf2.sum() > 0].columns.values.tolist())
        for suspect in suspects:
            case_sizes = tdf1[suspect]
            control_sizes = tdf2[suspect]
            rel_case_sizes = case_sizes/np.array(ncase)
            rel_control_sizes = control_sizes/np.array(ncontrol)
            np.mean((case_sizes+0.0001)/np.array(ncase))
            case_mean_freq = np.mean((case_sizes+0.0001)/np.array(ncase))
            case_presence = case_sizes.sum()
            case_absence = ncase.sum() - case_presence
            control_mean_freq = np.mean((control_sizes+0.0001)/np.array(ncontrol))
            control_presence = control_sizes.sum()
            control_absence = ncontrol.sum() - control_presence
            oddsratio, p = fisher_exact([[case_presence, control_presence], [case_absence, control_absence]])
            logfoldchange = np.log2(case_mean_freq/control_mean_freq)
            clt_stats.append([suspect, p, -np.log10(p), logfoldchange])
            for e in rel_case_sizes.index.values:
                clt_freq.append((suspect, hue, case_label, e, rel_case_sizes[e]))
            for e in rel_control_sizes.index.values:
                clt_freq.append((suspect, hue, control_label, e, rel_control_sizes[e]))
    
    # Convert records to data frames
    clt_freq = pd.DataFrame.from_records(clt_freq, columns=[target_col, additional_hue, groupby, replicate_col, 'Normalized abundance'])
    clt_stats = pd.DataFrame.from_records(clt_stats, columns=[target_col, 'pValue', 'logpValue', 'logFC'])

    if inplace:

        # Store calculated data
        adata.uns[added_key] = {
            "abundance": clt_freq,
            "pvalues": clt_stats
        }

        return

    else:
        return clt_freq, clt_stats
