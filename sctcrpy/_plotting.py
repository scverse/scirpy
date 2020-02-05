import matplotlib.pyplot as plt
import numpy as np
from ._compat import Literal
from anndata import AnnData
from scanpy import logging
import pandas as pd
import seaborn as sns
from ._util import _get_from_uns
from . import tl


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar"
) -> None:
    """Plot the alpha diversity per group. 

    Parameters
    ----------
    adata
        Annotated data matrix
    groupby
        Column of `obs` by which the grouping will be performed
    target_col
        Column on which to compute the alpha diversity
    vistype
        Visualization type. Currently only 'bar' is supported. 
    """
    try:
        diversity = _get_from_uns(
            adata,
            "alpha_diversity",
            parameters={"groupby": groupby, "target_col": target_col},
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing alpha diversity now. "
        )
        tl.alpha_diversity(adata, groupby, target_col=target_col)
        diversity = _get_from_uns(
            adata,
            "alpha_diversity",
            parameters={"groupby": groupby, "target_col": target_col},
        )

    groups, diversity = zip(*diversity.items())

    # TODO: this should probably make use of a `barplot` "base" plotting function.
    fig, ax = plt.subplots()
    x = np.arange(len(groups))
    ax.bar(x, diversity)

    ax.set_ylabel("Shannon entropy")
    ax.set_title("Alpha diversity of {} by {}".format(target_col, groupby))
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def clonal_expansion(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    clip_at: int = 3,
    fraction: bool = True
):
    """Plot the fraction of cells in each group belonging to
    singleton, doublet or triplet clonotype. 
    """
    # Get pre-comuted results. If not available, compute them.
    try:
        expansion = _get_from_uns(
            adata,
            "clonal_expansion",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
            },
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing clonal expansion now. "
        )
        tl.clonal_expansion(
            adata, groupby, target_col=target_col, clip_at=clip_at, fraction=fraction
        )
        expansion = _get_from_uns(
            adata,
            "clonal_expansion",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "clip_at": clip_at,
                "fraction": fraction,
            },
        )

    pd.DataFrame.from_dict(expansion, orient="index").plot.bar(stacked=True)

def group_abundance(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    label_col: Union[str, None] = None,
    top_n: int = 10,
    group_order: list = [],
    inplace: bool = True,
    fraction: bool = True
):
    """Plots how many cells belong to each clonotype. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the abundance. 
    label_col
        Column containing names for clonotypes. 
    top_n
        Top clonotypes to plot.        
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype. 
    """
    # Get pre-comuted results. If not available, compute them.
    try:
        abundance = _get_from_uns(
            adata,
            "group_abundance",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing group abundance now. "
        )
        tl.group_abundance(
            adata, groupby, target_col=target_col, fraction=fraction
        )
        abundance = _get_from_uns(
            adata,
            "group_abundance",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )

    target_ranks_ = abundance.pop('_order')
    target_ranks = target_ranks_[:top_n]
    target_ranks = target_ranks[::-1]
    if len(group_order) < 1:
        group_order = list(abundance[target_ranks[0]].keys())
    abundance_toplot = dict()
    if target_col != label_col:
        relabels = dict()
        for d in adata.obs.loc[:, [label_col, target_col]].to_dict(orient="records"):
            relabels[d[target_col]] = d[label_col]
        for t in target_ranks:
            bd = dict()
            for g in group_order:
                bd[g] = abundance[t][g]
            abundance_toplot[relabels[t]] = bd
    else:
        for t in target_ranks:
            bd = dict()
            for g in group_order:
                bd[g] = abundance[t][g]
            abundance_toplot[t] = bd
    
    pd.DataFrame.from_dict(abundance_toplot, orient="index").plot.barh()
    abundance['_order'] = target_ranks_

def group_abundance_complicated(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    label_col: str = "clonotype",
    top_n: int = 10,
    group_order: list = [],
    inplace: bool = True,
    fraction: bool = True
):
    """Plots how many cells belong to each clonotype. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the abundance. 
    label_col
        Column containing names for clonotypes. 
    top_n
        Top clonotypes to plot.        
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype. 
    """
    # Get pre-comuted results. If not available, compute them.
    try:
        abundance = _get_from_uns(
            adata,
            "group_abundance_complicated",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing group abundance now. "
        )
        tl.group_abundance(
            adata, groupby, target_col=target_col, fraction=fraction
        )
        abundance = _get_from_uns(
            adata,
            "group_abundance_complicated",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )

    target_ranks_ = abundance.pop('_order')
    target_ranks = target_ranks_[:top_n]
    target_ranks = target_ranks[::-1]
    if len(group_order) < 1:
        group_order = list(abundance[target_ranks[0]].keys())
    abundance_toplot = dict()
    if target_col != label_col:
        relabels = dict()
        for d in adata.obs.loc[:, [label_col, target_col]].to_dict(orient="records"):
            relabels[d[target_col]] = d[label_col]
        for t in target_ranks:
            bd = dict()
            for g in group_order:
                bd[g] = abundance[t][g]
            abundance_toplot[relabels[t]] = bd
    else:
        for t in target_ranks:
            bd = dict()
            for g in group_order:
                bd[g] = abundance[t][g]
            abundance_toplot[t] = bd
    
    pd.DataFrame.from_dict(abundance_toplot, orient="index").plot.barh()
    abundance['_order'] = target_ranks_

def group_abundance_lazy(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    label_col: str = "clonotype",
    top_n: int = 10,
    group_order: list = [],
    inplace: bool = True,
    fraction: bool = True
):
    """Plots how many cells belong to each clonotype. 

    It is lazy because it uses Seaborn to do all the actual work.

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the abundance. 
    label_col
        Column containing names for clonotypes. 
    top_n
        Top clonotypes to plot.        
    inplace
        If True, the results are added to `adata.uns`. Otherwise it returns a dict
        with the computed values. 
    fraction
        If True, compute fractions of clonotypes rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype. 
    """
    # Get pre-comuted results. If not available, compute them.
    try:
        abundance = _get_from_uns(
            adata,
            "group_abundance_lazy",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )
    except KeyError:
        logging.warning(
            "No precomputed data found for current parameters. "
            "Computing group abundance now. "
        )
        tl.group_abundance_lazy(
            adata, groupby, target_col=target_col, fraction=fraction
        )
        abundance = _get_from_uns(
            adata,
            "group_abundance_lazy",
            parameters={
                "groupby": groupby,
                "target_col": target_col,
                "fraction": fraction,
            },
        )

    target_ranks = abundance['order']
    target_ranks = target_ranks[:top_n]
    if len(group_order) < 1:
        group_order = abundance['df'].loc[:, groupby].unique().tolist()
    ax = sns.countplot(y=target_col, hue=groupby, data=abundance['df'], order=target_ranks, hue_order=group_order)
    return ax
