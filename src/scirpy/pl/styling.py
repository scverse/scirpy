from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
from cycler import Cycler
from mudata import MuData
from scanpy.plotting._utils import (
    _set_colors_for_categorical_obs,
    _set_default_colors_for_categorical_obs,
    _validate_palette,
)

from scirpy.util import DataHandler

DEFAULT_FIG_KWS = {"figsize": (3.44, 2.58), "dpi": 120}


def apply_style_to_axes(ax: plt.Axes, style: Literal["default"] | None, style_kws: dict | None) -> None:
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
        style_kws = {} if style_kws is None else style_kws
        if style == "default":
            return style_axes(ax, **style_kws)
        else:
            raise ValueError(f"Unknown style: {style}")


def _init_ax(fig_kws: dict | None = None) -> plt.Axes:
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
    ax.set_title(title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc)
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize)
    ax.set_ylabel(ylab, fontsize=label_fontsize)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)

    ax.set_title(title, fontdict={"fontsize": title_fontsize}, pad=title_pad, loc=title_loc)
    ax.set_xlabel(xlab, fontsize=label_fontsize)
    if change_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize, rotation=30, ha="right")
        xax = ax.get_xaxis()
        xax.set_tick_params(length=0)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
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


def _get_colors(
    params: DataHandler,
    obs_key: str,
    palette: str | Sequence[str] | Cycler | None = None,
) -> dict[str, str] | None:
    """Return colors for a category stored in AnnData.

    If colors are not stored, new ones are assigned.

    Since we currently don't plot expression values, only keys from `obs`
    are supported, while in scanpy `values_to_plot` (used instead of `obs_key`)
    can be a key from either `obs` or `var`.

    TODO: This makes use of private scanpy functions. This is evil and
    should be changed in the future.
    """
    # required to turn into categoricals
    params.data.strings_to_categoricals()

    # we can only get a palette for columns that are now categorical. Boolean/int/... won't work
    if isinstance(params.data, MuData):
        if obs_key in params.data.obs.columns:
            uns_lookup = params.data
        else:
            uns_lookup = params.adata
    else:
        uns_lookup = params.adata
    if uns_lookup.obs[obs_key].dtype.name == "category":
        values = uns_lookup.obs[obs_key].values
        categories = values.categories  # type: ignore
        color_key = f"{obs_key}_colors"
        if palette is not None:
            _set_colors_for_categorical_obs(uns_lookup, obs_key, palette)
        elif color_key not in uns_lookup.uns or len(uns_lookup.uns[color_key]) < len(categories):
            #  set a default palette in case that no colors or few colors are found
            _set_default_colors_for_categorical_obs(uns_lookup, obs_key)
        else:
            _validate_palette(uns_lookup, obs_key)

        return dict(zip(categories, uns_lookup.uns[color_key], strict=False))
