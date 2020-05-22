from anndata import AnnData
from .._compat import Literal
from . import base
from .. import tl
import numpy as np
import matplotlib.pyplot as plt


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar",
    **kwargs
) -> plt.Axes:
    """Plot the alpha diversity per group. 

    Calls :func:`scirpy.tl.alpha_diversity`. 

    Parameters
    ----------
    adata
        Annotated data matrix. Will execute :func:`scirpy.tl.alpha_diversity` on-the-fly.
    groupby
        Column of `obs` by which the grouping will be performed
    target_col
        Column on which to compute the alpha diversity
    vistype
        Visualization type. Currently only 'bar' is supported. 
    **kwargs
        Additional parameters passed to :func:`scirpy.pl.base.bar`
    """
    diversity = tl.alpha_diversity(adata, groupby, target_col=target_col, inplace=False)
    default_style_kws = {
        "title": "Alpha diversity of {} by {}".format(target_col, groupby),
        "ylab": "norm. Shannon entropy",
    }
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    ax = base.bar(diversity, style_kws=default_style_kws, **kwargs)
    ax.set_ylim(np.min(diversity.values) - 0.05, 1.0)
    return ax
