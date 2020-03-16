import matplotlib.pyplot as plt
from .._compat import Literal
from anndata import AnnData
import pandas as pd
from .._util import _is_na
from .. import tl
from . import _base as base
from typing import Union, List
from . import base


def group_abundance(
    adata: Union[dict, AnnData],
    groupby: str,
    *,
    target_col: str = "clonotype",
    fraction: Union[None, str, bool] = None,
    **kwargs
) -> plt.Axes:
    """Plots how many cells belong to each clonotype. 

    Ignores NaN values. 
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Group by this column from `obs`. Samples or diagnosis for example.
    target_col
        Column on which to compute the abundance. 
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized.  
    **kwargs
        Additional arguments passed to the base plotting function.  
    
    Returns
    -------
    Axes object
    """

    abundance = tl.group_abundance(
        adata, groupby, target_col=target_col, fraction=fraction
    )

    # Create text for default labels
    if fraction:
        title = "Fraction of top " + target_col + "s in each " + groupby
        xlab = "Fraction of cells in " + fraction_base
        ylab = "Fraction of cells in " + fraction_base
    else:
        title = "Number of cells in top " + target_col + "s by " + groupby
        xlab = "Number of cells"
        ylab = "Number of cells"

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])

    return base.bar(abundance, style_kws=default_style_kws)
