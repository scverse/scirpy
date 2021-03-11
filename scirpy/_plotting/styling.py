from .._compat import Literal
import matplotlib.pyplot as plt
from typing import Union, Sequence, Dict
from scanpy.plotting._utils import (
    _set_colors_for_categorical_obs,
    _set_default_colors_for_categorical_obs,
    _validate_palette,
)
from cycler import Cycler

DEFAULT_FIG_KWS = {"figsize": (3.44, 2.58), "dpi": 120}


def apply_style_to_axes(
    ax: plt.Axes, style: Union[Literal["default"], None], style_kws: Union[dict, None]
) -> None:
    """Apply a predefined style to an axis object.

    Parameters
    ----------
    ax
        Axes object
    style
        Style to apply to the axes. Currently supported are `None` (disable styling)
        and `'default'` (default style).
    style_kws
        Parameters passed to :func:`scirpy.pl.styling.style_axes` which
        override the defaults provided by the style.
    """
    if style is not None:
        style_kws = dict() if style_kws is None else style_kws
        if style == "default":
            return style_axes(ax, **style_kws)
        else:
            raise ValueError("Unknown style: {}".format(style))


def _init_ax(fig_kws: Union[dict, None] = None) -> plt.Axes:
    fig_kws = DEFAULT_FIG_KWS if fig_kws is None else fig_kws
    _, ax = plt.subplots(**fig_kws)
    return ax


def style_axes(
    ax: plt.Axes,
    title: str = "",
    legend_title: str = "",
    xlab: str = "",
    ylab: str = "",
    title_loc: Literal["center", "left", "right"] = "center",
    title_pad: float = None,
    title_fontsize: int = 10,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    change_xticks: bool = True,
    add_legend: bool = True,
) -> None:
    """Style an axes object.

    Parameters
    ----------
    ax
        Axis object to style.
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
    change_xticks
        REmoves ticks from x axis.
    add_legend
        Font size of the axis tick labels.
    """
    ax.set_title(
        title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
    )
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
    ax.set_ylabel(ylab, fontsize=label_fontsize)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)

    ax.set_title(
        title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc
    )
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    if change_xticks:
        ax.set_xticklabels(
            ax.get_xticklabels(), fontsize=tick_fontsize, rotation=30, ha="right"
        )
        xax = ax.get_xaxis()
        xax.set_tick_params(length=0)
    ax.set_ylabel(ylab, fontsize=label_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if add_legend:
        ax.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            title_fontsize=label_fontsize,
            fontsize=tick_fontsize,
            frameon=False,
        )
        ax.set_position([0.1, 0.3, 0.6, 0.55])


def _get_colors(
    adata, obs_key: str, palette: Union[str, Sequence[str], Cycler, None] = None
) -> Dict[str, str]:
    """Return colors for a category stored in AnnData.

    If colors are not stored, new ones are assigned.

    Since we currently don't plot expression values, only keys from `obs`
    are supported, while in scanpy `values_to_plot` (used instead of `obs_key`)
    can be a key from either `obs` or `var`.

    TODO: This makes use of private scanpy functions. This is Evil and
    should be changed in the future.
    """
    # required to turn into categoricals
    adata._sanitize()
    values = adata.obs[obs_key].values
    color_key = f"{obs_key}_colors"
    if palette is not None:
        _set_colors_for_categorical_obs(adata, obs_key, palette)
    elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
        values.categories
    ):
        #  set a default palette in case that no colors or few colors are found
        _set_default_colors_for_categorical_obs(adata, obs_key)
    else:
        _validate_palette(adata, obs_key)

    return {cat: col for cat, col in zip(values.categories, adata.uns[color_key])}
