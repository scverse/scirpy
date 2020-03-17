from anndata import AnnData
from .._compat import Literal
from . import base
from .. import tl


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar",
    **kwargs
) -> None:
    """Plot the alpha diversity per group. 

    Calls :meth:`tl.alpha_diversity` on-the-fly. 

    Parameters
    ----------
    adata
        Annotated data matrix. Will execute `tl.alpha_diversity` on-the-fly.
    groupby
        Column of `obs` by which the grouping will be performed
    target_col
        Column on which to compute the alpha diversity
    vistype
        Visualization type. Currently only 'bar' is supported. 
    **kwargs
        Additional parameters passed to :meth:`pl.base.bar`
    """
    diversity = tl.alpha_diversity(adata, groupby, target_col=target_col)
    default_style_kws = {
        "title": "Alpha diversity of {} by {}".format(target_col, groupby),
        "ylab": "Shannon entropy",
    }
    if "style_kws" in kwargs:
        default_style_kws.update(kwargs["style_kws"])
    return base.bar(diversity, style_kws=default_style_kws, **kwargs)
