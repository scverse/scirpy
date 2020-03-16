from anndata import AnnData
from .._compat import Literal
from . import base
from .. import tl


def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clonotype",
    vistype: Literal["bar"] = "bar"
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
    """
    diversity = tl.alpha_diversity(adata, groupby, target_col=target_col)

    return base.bar(diversity)

    # ax.set_ylabel("Shannon entropy")
    # ax.set_title("Alpha diversity of {} by {}".format(target_col, groupby))
