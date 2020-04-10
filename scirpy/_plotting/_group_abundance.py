import matplotlib.pyplot as plt
from .._compat import Literal
from anndata import AnnData
from .. import tl
from . import _base as base
from typing import Union, List, Sequence


def group_abundance(
    adata: Union[dict, AnnData],
    groupby: str,
    target_col: str = "has_tcr",
    *,
    fraction: Union[None, str, bool] = None,
    max_cols: Union[None, int] = None,
    sort: Union[Literal["count", "alphabetical"], Sequence[str]] = "count",
    **kwargs,
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
        Defaults to `has_tcr` which computes the number of all cells
        that have a T-cell receptor. 
    fraction
        If True, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized. 
    max_cols: 
        Only plot the first `max_cols` columns. Will raise a 
        `ValueError` if attempting to plot more than 100 columsn. 
        Set to `0` to disable. 
    sort
        How to arrange the dataframe columns. 
        Default is by the category count ("count"). 
        Other options are "alphabetical" or to provide a list of column names.
        By providing an explicit list, the DataFrame can also be subsetted to
        specific categories. Sorting (and subsetting) occurs before `max_cols` 
        is applied. 
    **kwargs
        Additional arguments passed to the base plotting function.  
    
    Returns
    -------
    Axes object
    """
    abundance = tl.group_abundance(
        adata, groupby, target_col=target_col, fraction=fraction, sort=sort
    )
    if abundance.shape[0] > 100 and max_cols is None:
        raise ValueError(
            "Attempting to plot more than 100 columns. "
            "Set `max_cols` to a sensible value or to `0` to disable this message"
        )
    if max_cols is not None and max_cols > 0:
        abundance = abundance.iloc[:max_cols, :]

    color_key = f"{target_col}_colors"
    if color_key in adata.uns and "color" not in kwargs:
        cat_index = {
            cat: i for i, cat in enumerate(adata.obs[target_col].cat.categories)
        }
        kwargs["color"] = [
            adata.uns[color_key][cat_index[cat]] for cat in abundance.columns
        ]

    # Create text for default labels
    if fraction:
        fraction_base = target_col if fraction is True else fraction
        title = "Fraction of " + target_col + " in each " + groupby
        xlab = groupby
        ylab = "Fraction of cells in " + fraction_base
    else:
        title = "Number of cells in " + groupby + " by " + target_col
        xlab = groupby
        ylab = "Number of cells"

    default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    kwargs["style_kws"] = default_style_kws

    return base.bar(abundance, **kwargs)
