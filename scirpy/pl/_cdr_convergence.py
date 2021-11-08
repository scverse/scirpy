import matplotlib.pyplot as plt
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from .. import tl
from . import _base as base
from typing import Union, List, Collection
from ..util import _doc_params
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
@_doc_params(common_doc=base._common_doc)
def cdr_convergence(
    adata: Union[dict, AnnData],
    groupby: str,
    *,
    target_col: str = "TRB_1_junction_aa",
    clip_at: int = 3,
    group_order: Union[Collection, None] = None,
    top_n: int = 10,
    vizarg: Union[dict, None] = None,
    ax: Union[plt.Axes, None] = None,
    sizeprofile: Union[Literal["small"], None] = None,
    no_singles: bool = False,
    fraction: Union[None, str, bool] = None,
    **kwds,
) -> Union[List[plt.Axes], AnnData]:
    """how many nucleotide versions a single CDR3 amino acid sequence typically has in a given group
    cells belong to each clonotype within a certain sample.

    Ignores NaN values.

    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the expansion. Useful if we want to specify the chain.
    for_cells
        A whitelist of cells that should be included in the analysis. If not specified, cells with NaN values in the group definition columns will be ignored. When the tool is executed by the plotting function, the whitelist is not updated.
    clip_at
        All clonotypes with more copies than `clip_at` will be summarized into
        a single group.
    grouporder
        Specifies the order of group (samples).
    top_n
        Number of groups to plot.
    viztype
        The user can later choose the layout of the plot. Currently supports `bar` and `stacked`.
    vizarg
        Custom values to be passed to the plot in a dictionary of arguments.
    ax
        Custom axis if needed.
    curve_layout
        if the KDE-based curves should be stacked or shifted vetrically.
    sizeprofile
        Figure size and font sizes to make everything legible. Currenty only `small` is supported.
    fraction
        If True, compute fractions of cells rather than reporting
        abosolute numbers. Always relative to the main grouping variable.leton, doublet or triplet clonotype.
    no_singles
        If non-convergent clonotypes should be shown explicitely.

    Returns
    -------
    List of axes or the dataFrame to plot.
    """

    # Check how fractions should be computed
    fraction, fraction_base = _which_fractions(fraction, None, groupby)
    plottable = tl.cdr_convergence(
        adata,
        groupby,
        target_col=target_col,
        fraction=fraction,
        fraction_base=fraction_base,
    )

    if type(plottable) == dict:
        plottable = pd.DataFrame.from_dict(adata, orient="index")
    if no_singles:
        plottable = plottable.drop("1", axis=1)

    if vizarg is None:
        vizarg = dict()

    if group_order is None:
        group_order = plottable.index.values
    plottable = plottable.loc[group_order, :]

    # Create text for default labels
    title = "Convergence of CDR3 regions in " + groupby + "s"
    if fraction:
        xlab = "Fraction of cells in " + fraction_base
        ylab = "Fraction of cells in " + fraction_base
    else:
        xlab = "Number of cells"
        ylab = "Number of cells"

    # Create a dictionary of plot layouts
    plot_router = {
        "bar": {
            "f": base.bar,
            "arg": {
                "data": plottable,
                "title": title,
                "legend_title": "Versions",
                "ylab": ylab,
                "xlab": target_col,
                "ax": ax,
                "fraction": fraction,
                "stacked": True,
            },
        }
    }

    # Check for settings in the profile and call the basic plotting function with merged arguments
    if viztype == "table":
        return plottable
    else:
        if sizeprofile is None:
            profile_args = _check_for_plotting_profile(adata)
        else:
            profile_args = _check_for_plotting_profile(sizeprofile)
        main_args = dict(
            dict(dict(profile_args, **kwds), **plot_router[viztype]["arg"]), **vizarg
        )
        axl = plot_router[viztype]["f"](**main_args)
        return axl
