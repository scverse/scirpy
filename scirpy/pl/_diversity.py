from anndata import AnnData
from .._compat import Literal
from . import base
from .. import tl
import numpy as np
import matplotlib.pyplot as plt
from ..io._util import _check_upgrade_schema
from typing import Union, Callable, Mapping


@_check_upgrade_schema()
def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: Union[
        str, Callable[[np.ndarray], Union[int, float]]
    ] = "normalized_shannon_entropy",
    metric_kwargs: Mapping = None,
    vistype: Literal["bar"] = "bar",
    **kwargs,
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
    metric
        A metric used for diversity estimation out of `normalized_shannon_entropy`,
        `D50`, `DXX`, any of scikit-bio’s alpha diversity metrics, or a custom function.
        For more details, see :func:`scirpy.tl.alpha_diversity`.
    metric_kwargs
        Dictionary of additional parameters passed to the metric function.
    vistype
        Visualization type. Currently only 'bar' is supported.
    **kwargs
        Additional parameters passed to :func:`scirpy.pl.base.bar`
    """
    diversity = tl.alpha_diversity(
        adata,
        groupby,
        target_col=target_col,
        metric=metric,
        inplace=False,
        **(dict() if metric_kwargs is None else metric_kwargs),
    )
    default_style_kws = {
        "title": "Alpha diversity of {} by {}".format(target_col, groupby),
        # convert snake case to title case
        "ylab": metric.replace("_", " ").title(),
    }
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    ax = base.bar(diversity, style_kws=default_style_kws, **kwargs)
    # commented out the line below to use default settings to
    # accommodate values from various metrics
    # ax.set_ylim(np.min(diversity.values) - 0.05, 1.0)
    return ax
