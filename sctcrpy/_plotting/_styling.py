from .._compat import Literal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Union


def style_axes(
    ax: plt.axes, style: Union[Literal["default"], None], style_kws: Union[dict, None]
) -> None:
    """Apply a style to an axis object. 

    Parameters:
    -----------
    ax
        Axes object
    style
        Style to apply to the axes. Currently supported are `None` (disable styling)
        and default (default style). 
    style_kws
        Parameters passed to :meth:`_plotting._styling._style_axes`
    """
    if style is not None:
        style_kws = dict() if style_kws is None else style_kws
        if style == "default":
            return _style_axes(ax, **style_kws)
        else:
            raise ValueError("Unknown style: {}".format(style))


def _style_axes(
    ax: plt.axes,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = 1.5,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 8,
    fraction: bool = True,
) -> None:
    """Style an axes object. 
    
    Parameters
    ----------
    ax
        Axis object to style
    title
        Figure title.
    legend_title
        Figure legend title.
    xlab
        Label for the x axis.
    ylab
        Label for the y axis.
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
    fraction
        Style as though the plot shows fractions
    
    """
    ax.set_title(
        title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
    )
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
    ax.set_ylabel(ylab, fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)

    ax.set_title(
        title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
    )
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    ax.set_xticklabels(
        ax.get_xticklabels(), fontsize=tick_fontsize, rotation=30, ha="right"
    )
    xax = ax.get_xaxis()
    xax.set_tick_params(length=0)
    ax.set_ylabel(ylab, fontsize=label_fontsize)
    if fraction:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_yticklabels(
            [str(int(x)) for x in ax.get_xticks()], fontsize=tick_fontsize
        )
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
    ax.set_position([0.1, 0.3, 0.6, 0.55])
