import matplotlib.pyplot as plt
import numpy as np
from ._compat import Literal
from anndata import AnnData
from scanpy import logging
import pandas as pd
from ._util import _get_from_uns
from . import tl


def tcr_umap(adata: AnnData):
    """Plot the TCR umap"""
    pass


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar",
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
    fraction: bool = True,
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
