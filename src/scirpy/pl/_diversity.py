from collections.abc import Callable, Mapping
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from scirpy import tl
from scirpy.util import DataHandler

from . import base


@DataHandler.inject_param_docs()
def alpha_diversity(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: str | Callable[[np.ndarray], int | float] = "normalized_shannon_entropy",
    metric_kwargs: Mapping | None = None,
    vistype: Literal["bar"] = "bar",
    airr_mod: str = "airr",
    **kwargs,
) -> plt.Axes:
    """\
    Plot the alpha diversity per group.

    Will execute :func:`scirpy.tl.alpha_diversity` on-the-fly.

    Parameters
    ----------
    {adata}
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
    {airr_mod}
    **kwargs
        Additional parameters passed to :func:`scirpy.pl.base.bar`
    """
    params = DataHandler(adata, airr_mod)
    diversity = tl.alpha_diversity(
        params,
        groupby,
        target_col=target_col,
        metric=metric,
        inplace=False,
        **({} if metric_kwargs is None else metric_kwargs),
    )
    default_style_kws = {
        "title": f"Alpha diversity of {target_col} by {groupby}",
        # convert snake case to title case
        "ylab": metric.replace("_", " ").title(),
    }
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])

    assert diversity is not None
    ax = base.bar(diversity, style_kws=default_style_kws, **kwargs)
    ax.get_legend().remove()  # type: ignore
    return ax
