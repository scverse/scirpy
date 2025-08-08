"""Base plotting functions"""

import itertools
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from cycler import Cycler
from matplotlib import cycler, rcParams
from sklearn.neighbors import KernelDensity

from scirpy.util import _doc_params

from .styling import DEFAULT_FIG_KWS, _init_ax, apply_style_to_axes

_common_doc = f"""\
style
    Style to apply to the axes. Currently supported are `None` (disable styling)
    and default (default style).
style_kws
    Parameters passed to :func:`scirpy.pl.styling.style_axes`
fig_kws
    Parameters passed to the :func:`matplotlib.pyplot.figure` call
    if no `ax` is specified. Defaults to `{str(DEFAULT_FIG_KWS)}` if None.
"""


@_doc_params(common_doc=_common_doc)
def bar(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    stacked: bool = True,
    style: Literal["default"] | None = "default",
    style_kws: dict | None = None,
    fig_kws: dict | None = None,
    **kwargs,
) -> plt.Axes:
    """\
    Basic plotting function built on top of bar plot in Pandas.

    Draws bars without stdev.

    Parameters
    ----------
    data
        Data to plot in wide-format (i.e. each row becomes a bar)
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
    if "grid" not in kwargs:
        kwargs["grid"] = False
    ax = data.plot.bar(ax=ax, stacked=stacked, **kwargs)

    # Remove excess x label
    if style_kws is not None:
        if "xlab" in style_kws:
            if "ylab" in style_kws:
                if style_kws["xlab"] == style_kws["ylab"]:
                    style_kws["xlab"] = ""

    apply_style_to_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def line(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    style: Literal["default"] | None = "default",
    style_kws: dict | None = None,
    fig_kws: dict | None = None,
    **kwargs,
) -> plt.Axes:
    """\
    Basic plotting function built on top of line plot in Pandas.

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
    if "grid" not in kwargs:
        kwargs["grid"] = False
    ax = data.plot.line(ax=ax, **kwargs)
    if style_kws is None:
        style_kws = {}
    style_kws["change_xticks"] = False
    apply_style_to_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def barh(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    style: Literal["default"] | None = "default",
    style_kws: dict | None = None,
    fig_kws: dict | None = None,
    **kwargs,
) -> plt.Axes:
    """\
    Basic plotting function built on top of bar plot in Pandas.

    Draws a horizontal bar plot.

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
    if "grid" not in kwargs:
        kwargs["grid"] = False
    ax = data.plot.barh(ax=ax, **kwargs)
    apply_style_to_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def curve(
    data: dict,
    *,
    ax: plt.Axes | None = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    shade: bool = True,
    kde_norm: bool = True,
    order: list | None = None,
    kernel_kws: dict | None = None,
    color: Sequence[str] | None = None,
    style: Literal["default"] | None = "default",
    style_kws: dict | None = None,
    fig_kws: dict | None = None,
    **kwargs,
) -> plt.Axes:
    """\
    Basic plotting function for drawing KDE-smoothed curves.

    Primarily designed for the :func:`scirpy.pl.spectratype` plotting
    function.

    Parameters
    ----------
    data
        Weighted counts for KDE.
    ax
        Custom axis if needed.
    curve_layout
        if the KDE-based curves should be stacked or shifted vetrically.
    kde_norm
        KDE curves are by default normalized to a sum of 1. Set to False in order to keep normalized cell weights.
    kernel_kws
        Parameters that should be passed to `KernelDensity` function of sklearn.
    order
        Specifies the order of groups.
    shade
        If True, draw a shade between curves
    color
        List of colors for each curve
    {common_doc}

    Returns
    -------
    Axes object.
    """
    if ax is None:
        ax = _init_ax(fig_kws)

    xmax = 0
    for _k, v in data.items():
        mx = np.amax(v)
        if mx > xmax:
            xmax = mx
    x = np.arange(0, xmax, 0.1)
    fy, _i = 0, 0
    yticks = []

    outline = curve_layout != "stacked"

    if order is None:
        order = list(data.keys())

    if kernel_kws is None:
        kernel_kws = {}
    if "kernel" not in kernel_kws:
        kernel_kws["kernel"] = "gaussian"
    if "bandwidth" not in kernel_kws:
        kernel_kws["bandwidth"] = 0.6

    # Draw a curve for every series
    for i in range(len(order)):
        tmp_color = None if color is None else color[i]
        label = order[i]
        col = data[label]
        sx = col.sum()
        X = col.reshape(-1, 1)
        kde = KernelDensity(**kernel_kws).fit(X)
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        if not kde_norm:
            y *= sx
        if curve_layout == "shifted":
            y = y + _i
            fy = _i + 0
            _i = y.max()
            yticks.append(fy)
        else:
            if curve_layout == "stacked":
                if i < 1:
                    _y = np.zeros(len(y))
                fy = _y[:]
                _y = _y + y
                y = fy + y
        if shade:
            if outline:
                ax.plot(x, y, label=label, color=tmp_color)
                ax.fill_between(x, y, fy, alpha=0.6, color=tmp_color)
            else:
                ax.fill_between(x, y, fy, alpha=0.6, label=label, color=tmp_color)
        else:
            ax.plot(x, y, label=label, color=tmp_color)

    if style_kws is None:
        style_kws = {}
    style_kws["change_xticks"] = False
    if kde_norm:
        style_kws["ylab"] = "Probability"
    if curve_layout == "shifted":
        style_kws["add_legend"] = False
        style_kws["ylab"] = ""
        ax.set_yticklabels(order)
        ax.set_yticks(yticks)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_yaxis().set_tick_params(length=0)
    apply_style_to_axes(ax, style, style_kws)
    ax.grid(False)
    return ax


@_doc_params(common_doc=_common_doc)
def ol_scatter(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    style_kws: dict | None = None,
    style: Literal["default"] | None = "default",
    fig_kws: dict | None = None,
) -> plt.Axes:
    """\
    Scatterplot where dot size is proportional to group size.

    Draws bars without stdev.

    Parameters
    ----------
    data
        Dataframe with three columns: x, y for position and z for dot size.
    ax
        Plot into this axes object
    {common_doc}

    Returns
    -------
    Axes object
    """
    if ax is None:
        ax = _init_ax(fig_kws)
    axlim = data["x"].max() + 1
    if data["y"].max() > axlim:
        axlim = data["y"].max() + 1
    ax.scatter(data["x"], data["y"], s=data["z"], alpha=0.3)
    ax.set_xlim(0, axlim)
    ax.set_ylim(0, axlim)
    if style_kws is None:
        style_kws = {}
    style_kws["change_xticks"] = False
    apply_style_to_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def volcano(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    style_kws: dict | None = None,
    style: Literal["default"] | None = "default",
    fig_kws: dict | None = None,
) -> plt.Axes:
    """\
    Volcano plot (special case of scatter plot)

    Draws bars without stdev.

    Parameters
    ----------
    data
        Dataframe with three columns: log-fold-change, log10 p-value and optionally colors.
    ax
        Plot into this axes object
    {common_doc}

    Returns
    -------
    Axes object
    """
    if ax is None:
        ax = _init_ax(fig_kws)

    if data.shape[1] > 3:
        data = data.iloc[:, [0, 1, 2]]
    if data.shape[1] == 2:
        data.columns = ["x", "y"]
        ax.scatter(data["x"], data["y"], s=5, alpha=0.3)
    else:
        data.columns = ["x", "y", "color"]
        ax.scatter(data["x"], data["y"], c=data["color"], s=5, alpha=0.3)
    axlim = 1.1 * max(data["x"].max(), abs(data["x"].min()))
    if np.isinf(axlim) or np.isnan(axlim):
        axlim = 5
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(0, 1.1 * (data["y"].max()))
    if style_kws is None:
        style_kws = {}
    style_kws["change_xticks"] = False
    apply_style_to_axes(ax, style, style_kws)
    return ax


def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: str | Sequence[str] | None = None,
    panel_size: tuple[float, float] = (4, 4),
    palette: str | Cycler | Sequence[str] | Sequence[Cycler] | None = None,
    legend_loc: str = "right margin",
    ax: plt.Axes | Sequence[plt.Axes] | None = None,
    ncols: int = 3,
    show: bool | None = False,
    hspace: float = 0.25,
    wspace: float = None,
    **kwargs,
) -> None | Sequence[plt.Axes]:
    """A customized wrapper to the :func:`scanpy.pl.embedding` function.

    The differences to the scanpy embedding function are:
        * allows to specify a `panel_size`
        * Allows to specify a different `basis`, `legend_loc` and `palette`
          for each panel. The number of panels is defined by the `color` parameter.
        * Use a patched version for adding "on data" labels. The original
          raises a flood of warnings when coords are `nan`.
        * For columns with many categories, cycles through colors
          instead of reverting to grey
        * allows to specify axes, even if multiple colors are set.

    Parameters
    ----------
    adata
        annotated data matrix
    basis
        embedding to plot.
        Get the coordinates from the "X_{basis}" key in `adata.obsm`.
        This can be a list of the same length as `color` to specify
        different bases for each panel.
    color
        Keys for annotations of observations/cells or variables/genes, e.g.,
        `'ann1'` or `['ann1', 'ann2']`.
    panel_size
        Size tuple (`width`, `height`) of a single panel in inches
    palette
        Colors to use for plotting categorical annotation groups.
        The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
        (`'Set2'`, `'tab20'`, …) or a :class:`~cycler.Cycler` object.
        It is possible to specify a list of the same size as `color` to choose
        a different color map for each panel.
    legend_loc
        Location of legend, either `'on data'`, `'right margin'` or a valid keyword
        for the `loc` parameter of :class:`~matplotlib.legend.Legend`.
    ax
        A matplotlib axes object or a list with the same length as `color` thereof.
    ncols
        Number of columns for multi-panel plots
    show
        If True, show the firgure. If false, return a list of Axes objects
    wspace
        Adjust the width of the space between multiple panels.
    hspace
        Adjust the height of the space between multiple panels.
    **kwargs
        Arguments to pass to :func:`scanpy.pl.embedding`.

    Returns
    -------
    axes
        A list of axes objects, containing one
        element for each `color`, or None if `show == True`.

    See Also
    --------
    :func:`scanpy.pl.embedding`
    """
    adata._sanitize()

    def _make_iterable(var, singleton_types=(str,)):
        return itertools.repeat(var) if isinstance(var, singleton_types) or var is None else list(var)

    color = [color] if isinstance(color, str) or color is None else list(color)
    basis = _make_iterable(basis)
    legend_loc = _make_iterable(legend_loc)
    palette = _make_iterable(palette, (str, Cycler))

    # set-up grid, if no axes are provided
    if ax is None:
        n_panels = len(color)
        nrows = int(np.ceil(float(n_panels) / ncols))
        ncols = np.min((n_panels, ncols))
        hspace = rcParams.get("figure.subplot.hspace", 0.0) if hspace is None else hspace
        wspace = rcParams.get("figure.subplot.wspace", 0.0) if wspace is None else wspace
        # Don't ask about +/- 1 but appears to be most faithful to the panel size
        fig_width = panel_size[0] * ncols + hspace * (ncols + 1)
        fig_height = panel_size[1] * nrows + wspace * (nrows - 1)
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fig_width, fig_height),
            gridspec_kw={"wspace": wspace, "hspace": hspace},
            squeeze=False,
        )
        axs = axs.flatten()
    else:
        axs = [ax] if not isinstance(ax, Sequence) else list(ax)
        fig = axs[0].get_figure()

    # use the scanpy plotting api to fill individual components
    for ax, tmp_color, tmp_basis, tmp_legend_loc, tmp_palette in zip(
        axs, color, basis, legend_loc, palette, strict=False
    ):
        # cycle colors for categories with many values instead of
        # coloring them in grey
        if tmp_palette is None and tmp_color is not None:
            if str(adata.obs[tmp_color].dtype) == "category":
                if adata.obs[tmp_color].unique().size > len(sc.pl.palettes.default_102):
                    tmp_palette = cycler(color=sc.pl.palettes.default_102)

        sc.pl.embedding(
            adata,
            tmp_basis,
            ax=ax,
            show=False,
            color=tmp_color,
            legend_loc=tmp_legend_loc,
            palette=tmp_palette,
            **kwargs,
        )

    # hide unused panels in grid
    for ax in axs[len(color) :]:
        ax.axis("off")

    if show:
        fig.show()
    else:
        # only return axes that actually contain a plot.
        return axs[: len(color)]
