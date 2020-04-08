"""Base plotting functions"""
from typing import Union, Sequence, Tuple, Optional
from .._compat import Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ._styling import style_axes, DEFAULT_FIG_KWS, _init_ax
from .._util import _doc_params
from sklearn.neighbors import KernelDensity
from cycler import Cycler
import itertools
import scanpy as sc
from matplotlib import rcParams, cycler, patheffects


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


@_doc_params(common_doc=_common_doc)
def bar(
    data: pd.DataFrame,
    *,
    ax: Union[plt.Axes, None] = None,
    stacked: bool = True,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
    **kwargs,
) -> plt.Axes:
    """Basic plotting function built on top of bar plot in Pandas.
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
    ax = data.plot.bar(ax=ax, stacked=stacked, **kwargs)

    # Remove excess x label
    if style_kws is not None:
        if "xlab" in style_kws:
            if "ylab" in style_kws:
                if style_kws["xlab"] == style_kws["ylab"]:
                    style_kws["xlab"] = ""

    style_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def line(
    data: pd.DataFrame,
    *,
    ax: Union[plt.Axes, None] = None,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
    **kwargs,
) -> plt.Axes:
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
    ax = data.plot.line(ax=ax, **kwargs)
    style_axes(ax, style, style_kws)
    return ax


def barh(
    data: pd.DataFrame,
    *,
    ax: Union[plt.Axes, None] = None,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
    **kwargs,
) -> plt.Axes:
    """Basic plotting function built on top of bar plot in Pandas.
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
    ax = data.plot.barh(ax=ax, **kwargs)
    style_axes(ax, style, style_kws)
    return ax


@_doc_params(common_doc=_common_doc)
def curve(
    data: pd.DataFrame,
    *,
    ax: Union[plt.Axes, None] = None,
    curve_layout: Literal["overlay", "stacked", "shifetd"] = "overlay",
    shade: bool = False,
    style: Union[Literal["default"], None] = "default",
    style_kws: Union[dict, None] = None,
    fig_kws: Union[dict, None] = None,
) -> plt.Axes:
    """Basic plotting function built on top of bar plot in Pandas.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Counts or pseudo-counts for KDE.
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

    xmax = np.nanmax(data.values)
    x = np.arange(0, xmax, 0.1)
    fy = 0

    outline = curve_layout != "stacked"

    # # We need to convert the contingency tables back for the KDE in seaborn,
    # # using pseudo-counts in case of fractions
    # if fraction:
    #     ftr = 1000 / np.max(data.values)
    # countable, counted = [], []
    # for cn in data.columns:
    #     counts = np.round(data[cn] * ftr)
    #     if counts.sum() > 0:
    #         countable.append(np.repeat(data.index.values, counts))
    #         counted.append(cn)
    # # countable, counted = countable[:top_n], counted[:top_n]

    # Draw a curve for every series
    for i, (label, col) in enumerate(data.iteritems()):
        X = col.values.reshape(-1, 1)
        # kde = KernelDensity(kernel="epanechnikov", bandwidth=3).fit(X)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.6).fit(X)
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        if curve_layout == "shifted":
            y = y + i
            fy = i
        else:
            if curve_layout == "stacked":
                if i < 1:
                    _y = np.zeros(len(y))
                fy = _y[:]
                _y = _y + y
                y = fy + y
        if shade:
            if outline:
                ax.plot(x, y, label=label)
                ax.fill_between(x, y, fy, alpha=0.6)
            else:
                ax.fill_between(x, y, fy, alpha=0.6, label=label)
        else:
            ax.plot(x, y, label=label)

    style_axes(ax, style, style_kws)

    return ax


@_doc_params(common_doc=_common_doc)
def ol_scatter(
    data: pd.DataFrame,
    *,
    ax: Union[plt.Axes, None] = None,
    style_kws: Union[dict, None] = None,
    style: Union[Literal["default"], None] = "default",
    fig_kws: Union[dict, None] = None,
) -> plt.Axes:
    """Scatterplot where dot size is proportional to group size.
    Draws bars without stdev. 

    Parameters
    ----------
    data
        Data to plot in wide-format (i.e. each row becomes a bar)
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
        style_kws = dict()
    style_kws["change_xticks"] = False
    style_axes(ax, style, style_kws)
    return ax


def _add_labels(
    ax: plt.Axes,
    coords: np.ndarray,
    label_data: np.ndarray,
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
):
    """Add legend labels on data at centroids position"""
    categories = np.unique(label_data)
    nan_mask = ~np.any(np.isnan(coords), axis=1)
    label_idx = {label: list() for label in categories}
    for i, label in enumerate(label_data):
        if nan_mask[i]:
            label_idx[label].append(i)

    for label, idx in label_idx.items():
        if len(idx):
            _scatter = coords[idx, :]
            x_pos, y_pos = np.median(_scatter, axis=0)

            if legend_fontoutline is not None:
                path_effect = [
                    patheffects.withStroke(
                        linewidth=legend_fontoutline, foreground="w",
                    )
                ]
            else:
                path_effect = None

            ax.text(
                x_pos,
                y_pos,
                label,
                weight=legend_fontweight,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=legend_fontsize,
                path_effects=path_effect,
            )


def embedding(
    adata,
    basis,
    *,
    color: Union[str, Sequence[str], None] = None,
    panel_size: Tuple[float] = (4, 4),
    palette: Union[str, Cycler, Sequence[str], Sequence[Cycler], None] = None,
    legend_loc: str = "right margin",
    ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
    ncols: int = 3,
    show: Optional[bool] = False,
    hspace: float = 0.25,
    wspace: float = None,
    **kwargs,
) -> Union[None, Sequence[plt.Axes]]:
    """A customized wrapper to the :meth:`sc.pl.embedding` function. 

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
    kwargs
        Arguments to pass to :func:`scanpy.pl.embedding`. 

    Returns
    -------
    axes
        A list of axes objects, containing one
        element for each `color`, or None if `show == True`. 
    
    See also
    --------
    :meth:`scanpy.pl.embedding`
    """
    adata._sanitize()

    def _make_iterable(var, singleton_types=(str,)):
        return (
            itertools.repeat(var)
            if isinstance(var, singleton_types) or var is None
            else list(var)
        )

    color = [color] if isinstance(color, str) or color is None else list(color)
    basis = _make_iterable(basis)
    legend_loc = _make_iterable(legend_loc)
    palette = _make_iterable(palette, (str, Cycler))

    # set-up grid, if no axes are provided
    if ax is None:
        n_panels = len(color)
        nrows = int(np.ceil(float(n_panels) / ncols))
        ncols = np.min((n_panels, ncols))
        hspace = (
            rcParams.get("figure.subplot.hspace", 0.0) if hspace is None else hspace
        )
        wspace = (
            rcParams.get("figure.subplot.wspace", 0.0) if wspace is None else wspace
        )
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
        axs, color, basis, legend_loc, palette
    ):
        # cycle colors for categories with many values instead of
        # coloring them in grey
        if tmp_palette is None and tmp_color is not None:
            if str(adata.obs[tmp_color].dtype) == "category":
                if adata.obs[tmp_color].unique().size > len(sc.pl.palettes.default_102):
                    tmp_palette = cycler(color=sc.pl.palettes.default_102)

        add_labels = tmp_legend_loc == "on data"
        tmp_legend_loc = None if add_labels else tmp_legend_loc

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

        # manually add labels for "on data", as missing entries in `obsm` will cause
        # a flood of matplotlib warnings.
        # TODO: this could eventually be fixed upstream in scanpy
        if add_labels:
            _add_labels(
                ax,
                adata.obsm["X_" + tmp_basis],
                adata.obs[tmp_color].values,
                legend_fontweight=kwargs.get("legend_fontweight", "bold"),
                legend_fontsize=kwargs.get("legend_fontsize", None),
                legend_fontoutline=kwargs.get("legend_fontoutline", None),
            ),

    # hide unused panels in grid
    for ax in axs[len(color) :]:
        ax.axis("off")

    if show:
        fig.show()
    else:
        # only return axes that actually contain a plot.
        return axs[: len(color)]
