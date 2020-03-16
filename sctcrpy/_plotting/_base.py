"""Base plotting functions"""

from typing import Union, Tuple, List, Collection
from .._compat import Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ._styling import style_axes
from .._util import _doc_params
import matplotlib.ticker as ticker
from sklearn.neighbors import KernelDensity

DEFAULT_FIG_KWS = {"figsize": (3.44, 2.58), "figresolution": 300}

_common_doc = """
    style
        Style to apply to the axes. Currently supported are `None` (disable styling)
        and default (default style). 
    style_kws
        Parameters passed to :meth:`_plotting._styling._style_axes`
    fig_kws
        Parameters passed to the :meth:`matplotlib.pyplot.figure` call 
        if no `ax` is specified. Defaults to `{}` if None. 
""".format(
    str(DEFAULT_FIG_KWS)
)


def _init_ax(fig_kws: Union[dict, None] = None) -> plt.axes:
    fig_kws = DEFAULT_FIG_KWS if fig_kws is None else fig_kws
    _, ax = plt.subplots(fig_kw=fig_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def bar(
    data: pd.DataFrame,
    *,
    ax: Union[plt.axes, None] = None,
    stacked: bool = True,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
) -> plt.axes:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Plot into this axes object
    stacked
        Determines if the vars should be stacked.  
    {common_doc}
    
    Returns
    -------
    Axes object 
    """
    if ax is None:
        ax = _init_ax(fig_kws)
    ax = data.plot.bar(ax=ax, stacked=stacked)
    style_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def line(
    data: pd.DataFrame,
    *,
    ax: Union[plt.axes, None] = None,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
) -> plt.axes:
    """Basic plotting function built on top of line plot in Pandas.

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Plot into this axes object
    {common_doc}
    
    Returns
    -------
    Axes object
    """
    if ax is None:
        ax = _init_ax(fig_kws)
    ax = data.plot.line(ax=ax)
    style_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def curve(
    data: List[Collection],
    labels: Collection,
    *,
    ax: Union[plt.axes, None] = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    shade: bool = False,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
) -> plt.axes:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Counts or pseudo-counts for KDE.
    labels
        The label to display for each curve
    ax
        Custom axis if needed.
    curve_layout
        if the KDE-based curves should be stacked or shifted vetrically.  
    shade
        If True, draw a shade between curves
    {common_doc}
    
    Returns
    -------
    List of axes.
    """

    ax = _init_ax(fig_kws)

    # Check what would be the plotting range
    xmax = 0
    for d in data:
        try:
            m = np.nanmax(d)
            if m > xmax:
                xmax = m
        except:
            pass
    xmax += 1
    x = np.arange(0, xmax, 0.1)

    # Draw a curve for every series
    for i in range(len(data)):
        X = np.array([data[i]]).reshape(-1, 1)
        # kde = KernelDensity(kernel="epanechnikov", bandwidth=3).fit(X)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.6).fit(X)
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        if curve_layout == "shifted":
            y = y + i
            fy = i
        else:
            if curve_layout == "stacked":
                outline = False
                if i < 1:
                    _y = np.zeros(len(y))
                fy = _y[:]
                _y = _y + y
                y = fy + y
        if shade:
            if outline:
                ax.plot(x, y, label=labels[i])
                ax.fill_between(x, y, fy, alpha=0.6)
            else:
                ax.fill_between(x, y, fy, alpha=0.6, label=labels[i])
        else:
            ax.plot(x, y, label=labels[i])

    style_axes(ax, style, style_kws)

    return ax


def stripe(
    data: pd.DataFrame,
    *,
    ax: Union[plt.axes, None] = None,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
) -> plt.axes:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    {common_doc}

    Returns
    -------
    Axes object
    """
    if ax is None:
        ax = _init_ax(fig_kws)
    ax = data.plot.barh(ax=ax)
    style_axes(ax)
    return ax
