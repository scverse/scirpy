from typing import Union
from .._compat import Literal
import matplotlib.pyplot as plt
from .._util import _doc_params, _get_from_uns, _add_to_uns
import matplotlib.ticker as ticker
from anndata import AnnData


def _reset_plotting_profile(adata: AnnData) -> None:
    """
    Reverts plotting profile to matplotlib defaults (rcParams).  
    """
    try:
        p = _get_from_uns(adata, "plotting_profile")
    except KeyError:
        p = dict()
    p["title_loc"] = plt.rcParams["axes.titleloc"]
    p["title_pad"] = plt.rcParams["axes.titlepad"]
    p["title_fontsize"] = plt.rcParams["axes.titlesize"]
    p["label_fontsize"] = plt.rcParams["axes.labelsize"]
    p["tick_fontsize"] = plt.rcParams["xtick.labelsize"]
    _add_to_uns(adata, "plotting_profile", p)
    return


def _check_for_plotting_profile(profile: Union[AnnData, str, None] = None) -> dict:
    """
    Passes a predefined set of plotting atributes to basic plotting fnctions.
    """
    profiles = {
        "vanilla": {},
        "small": {
            "figsize": (3.44, 2.58),
            "figresolution": 300,
            "title_loc": "center",
            "title_pad": 10,
            "title_fontsize": 10,
            "label_fontsize": 8,
            "tick_fontsize": 6,
        },
    }
    p = profiles["small"]
    if isinstance(profile, AnnData):
        try:
            p = _get_from_uns(profile, "plotting_profile")
        except KeyError:
            pass
    else:
        if isinstance(profile, str):
            if profile in profiles:
                p = profiles[profile]
    return p


_prettify_doc = """
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


@_doc_params(prettify_doc=_prettify_doc)
def _prettify(
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
    {prettify_doc}
    
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
