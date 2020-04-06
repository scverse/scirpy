import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union
from .._compat import Literal
from .. import tl
from ._base import volcano


def clonotype_imbalance(
    adata: AnnData,
    replicate_col: str,
    groupby: str,
    case_label: str,
    *,
    control_label: Union[None, str] = None,
    target_col: str = "clonotype",
    additional_hue: Union[None, str, bool] = None,
    top_n: int = 10,
    fraction: Union[None, str, bool] = None,
    inplace: bool = True,
    plot_type: Literal["volcano", "box", "bar", "strip"] = "box",
    added_key: str = "clonotype_imbalance",
    xlab: str = 'log2FoldChange',
    ylab: str = '-log10(p-value)',
    title: str = 'Volcano plot',
    **kwargs
) -> plt.Axes:
    """Aims to find clonotypes that are the most enriched or depleted in a category.

    Uses Fischer's exact test to rank clonotypes.
    Depends on execution of clonotype_overlap.
    Adds two dataframes (pval and logFC for clonotypes; abundance of clonotypes per sample) to `uns`
    
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
    top_n
        The number of top clonotypes to be visualized.
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    plot_type
        Whether a volcano plot of statistics or a box/bar/strip plot of frequencies should be shown.
    inplace
        Whether results should be added to `uns` or returned directly.
    added_key
        If the tools has already been run, the results are added to `uns` under this key.
    **kwargs
        Additional arguments passed to the base plotting function.  
    
    Returns
    -------
    Axes object
    """

    if added_key not in adata.uns:
        tl.clonotype_imbalance(
            adata,
            replicate_col=replicate_col,
            groupby=groupby,
            case_label=case_label,
            control_label=control_label,
            target_col=target_col,
            additional_hue=additional_hue,
            fraction=fraction,
            added_key=added_key,
        )
    
    df = adata.uns[added_key]["pvalues"]

    if plot_type == 'volcano':
        df = df.loc[:, ['logFC', 'logpValue']]
        default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
        if "style_kws" in kwargs:
            default_style_kws.update(kwargs["style_kws"])
        kwargs["style_kws"] = default_style_kws
        return volcano(df, **kwargs)
    
    else:
        df = df.sort_values(by='pValue')
        df = df.head(n=top_n)

        tclt_df = adata.uns[added_key]["abundance"]
        tclt_df = tclt_df.loc[tclt_df[target_col].isin(df[target_col]),:]
        
        if additional_hue is None:
            tclt_df = tclt_df.pivot_table(index=[groupby, replicate_col], columns=target_col, values='Normalized abundance', fill_value=0).reset_index()
            tclt_df = pd.melt(tclt_df, id_vars=[groupby, replicate_col], value_name='Normalized abundance')
            if plot_type == 'box':
                ax = sns.boxplot(x=target_col, y='Normalized abundance', hue=groupby, data=tclt_df)
            else:
                if plot_type == 'bar':
                    ax = sns.barplot(x=target_col, y='Normalized abundance', hue=groupby, data=tclt_df)
                else:
                    ax = sns.stripplot(x=target_col, y='Normalized abundance', hue=groupby, data=tclt_df, dodge=True, alpha=0.7)

        else:
            tclt_df = tclt_df.pivot_table(index=[additional_hue, groupby, replicate_col], columns=target_col, values='Normalized abundance', fill_value=0).reset_index()
            tclt_df = pd.melt(tclt_df, id_vars=[additional_hue, groupby, replicate_col], value_name='Normalized abundance')
            ax = sns.catplot(x=target_col, y='Normalized abundance', hue=groupby, kind=plot_type, col=additional_hue, data=tclt_df, dodge=True)
        return ax
