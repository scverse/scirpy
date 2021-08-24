from .styling import _init_ax, apply_style_to_axes
from .._compat import Literal
from typing import Union
import numpy as np


def _rand_jitter(arr):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


# TODO consider plotting size against connectivity, and color by pvalue (or circle
# those that are "significant". )


def clonotype_modularity(
    adata,
    ax=None,
    score_col="clonotype_modularity",
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
):
    """
    TODO docs
    """
    if ax is None:
        ax = _init_ax(fig_kws)

    apply_style_to_axes(ax, style, style_kws)

    if f"{score_col}_fdr" in adata.obs.columns:
        pvalue_col = f"{score_col}_fdr"
        pvalue_type = "FDR"
    else:
        pvalue_col = f"{score_col}_pvalue"
        pvalue_type = "p-value"

    score_df = (
        # TODO way to retrieve original clonotype annotation
        adata.obs.groupby(["clone_id", score_col, pvalue_col], observed=True)
        .size()
        .reset_index(name="clonotype_size")
    )

    ax.scatter(
        _rand_jitter(score_df[score_col].values),
        -np.log10(score_df[pvalue_col].values),
        marker=".",
        s=score_df["clonotype_size"].values,
    )
    ax.set_xlabel("connectivity scores")
    ax.set_ylabel(f"-log10({pvalue_type})")

    return ax
