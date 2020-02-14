"""Base plotting functions"""

from typing import Union, Tuple, List
from .._compat import Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ._styling import _prettify, _prettify_doc
from .._util import _doc_params
import matplotlib.ticker as ticker

_figsize_doc = """
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
"""


def _to_df(data: Union[dict, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """Convert input data to pandas dataframe if it is not one already"""
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient="index")
        else:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            else:
                raise ValueError("`data` does not seem to be a valid input type")

    return data


def bar(
    data: Union[dict, np.ndarray, pd.DataFrame],
    *,
    ax: Union[plt.axes, None] = None,
    stacked: bool = True,
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    **kwargs
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.
    stacked
        Determines if the vars should be stacked.  
    {figsize_doc}
    {prettify_doc}
    
    Returns
    -------
    List of axes.
    """

    data = _to_df(data)

    # Create figure if not supplied already.
    prettify = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        prettify = True

    # Draw the plot with Pandas
    ax = data.plot.bar(ax=ax, stacked=stacked)

    if prettify:
        _prettify(ax, **kwargs)

    return [ax]


@_doc_params(prettify_doc=_prettify_doc, figsize_doc=_figsize_doc)
def line(
    data: Union[dict, np.ndarray, pd.DataFrame],
    *,
    ax: Union[plt.axes, list, None] = None,
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    **kwargs
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    {figsize_doc}
    {prettify_doc}
    
    Returns
    -------
    List of axes.
    """
    data = _to_df(data)

    prettify = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        prettify = True

    # Draw the plot with Pandas
    ax = data.plot.line(ax=ax)

    # Make plot a bit prettier
    if prettify:
        _prettify(ax, **kwargs)
        # ax.set_title(
        #     title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        # )
        # ax.set_xlabel(xlab, fontsize=label_fontsize)
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        # ax.set_xticklabels(
        #     [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
        # )
        # ax.set_ylabel(ylab, fontsize=label_fontsize)
        # ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        # if fraction:
        #     ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        #     ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        #     ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
        # else:
        #     ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        #     ax.set_yticklabels(
        #         [str(int(x)) for x in ax.get_yticks()], fontsize=tick_fontsize
        #     )
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.legend(
        #     title=legend_title,
        #     loc="upper left",
        #     bbox_to_anchor=(1.2, 1),
        #     title_fontsize=label_fontsize,
        #     fontsize=tick_fontsize,
        #     frameon=False,
        # )
        # ax.set_position([0.3, 0.2, 0.5, 0.75])
    return [ax]


def curve(
    data: List[Union[np.ndarray, pd.Series]],
    labels: Union[list, np.ndarray, pd.Series],
    *,
    ax: Union[plt.axes, list, None] = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 10,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 6,
    shade: bool = True,
    outline: bool = True,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
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
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    shade
        If shade area should be plotted. 
    outline
        If the outline should be drawn. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Create figure if not supplied already. If multiple axes are supplied, it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Check what would be the plotting range
    xmax = 0
    for d in data:
        try:
            m = max(d)
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

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.set_xticklabels(
            [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
        )
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        if fraction:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            ax.set_yticklabels(
                [str(int(x)) for x in ax.get_yticks()], fontsize=tick_fontsize
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if curve_layout == "shifted":
            for e in range(xmax):
                ax.axvline(e, color="whitesmoke", lw=0.1)
            ax.set_position([0.3, 0.2, 0.7, 0.75])
            ax.set_yticks(range(i + 1))
            ax.set_yticklabels(
                [labels[e] for e in range(i + 1)], fontsize=tick_fontsize
            )
            ax.legend().remove()
        else:
            ax.legend(
                title=legend_title,
                loc="upper left",
                bbox_to_anchor=(1.2, 1),
                title_fontsize=label_fontsize,
                fontsize=tick_fontsize,
                frameon=False,
            )
            ax.set_position([0.3, 0.2, 0.5, 0.75])
    return [ax]


def stripe(
    data: Union[dict, np.ndarray, pd.DataFrame],
    *,
    ax: Union[plt.axes, list, None] = None,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Tuple[float, float] = (3.44, 2.58),
    figresolution: int = 300,
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 10,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 6,
    stacked: bool = True,
    fraction: bool = True,
    **kwds
) -> List[plt.axes]:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to show (wide format).
    ax
        Custom axis if needed.  
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
    figsize
        Size of the resulting figure in inches.
    figresolution
        Resolution of the figure in dpi. 
    title_loc
        Position of the plot title (can be {'center', 'left', 'right'}). 
    title_pad
        Padding of the plot title.
    title_fontsize
        Font size of the plot title. 
    label_fontsize
        Font size of the axis labels.   
    tick_fontsize
        Font size of the axis tick labels. 
    stacked
        Determines if the vars should be stacked.   
    **kwds
        Arguments not used by the current plotting layout.
    
    Returns
    -------
    List of axes.
    """

    # Convert data to a Pandas dataframe if not already a dataframe.
    data = _to_df(data)

    # Create figure if not supplied already. If multiple axes are supplied,
    # it is assumed that the first one is relevant to the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=figresolution)
        needprettier = True
    else:
        needprettier = False
        if type(ax) is list:
            ax = ax[0]

    # Draw the plot with Pandas
    ax = data.plot.barh(ax=ax)

    # Make plot a bit prettier
    if needprettier:
        ax.set_title(
            title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
        )
        ax.set_xlabel(xlab, fontsize=label_fontsize)
        if fraction:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set_xticklabels(ax.get_xticks(), fontsize=tick_fontsize)
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
            # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: str(int(x))))
            ax.set_xticklabels(
                [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
            )
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.set_yticklabels(
            ax.get_yticklabels(), fontsize=tick_fontsize, horizontalalignment="left"
        )
        yax = ax.get_yaxis()
        yax.set_tick_params(length=0, pad=60)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            title_fontsize=label_fontsize,
            fontsize=tick_fontsize,
            frameon=False,
        )
        ax.set_position([0.3, 0.2, 0.4, 0.65])
    return [ax]
