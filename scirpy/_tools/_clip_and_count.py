from anndata import AnnData
from typing import Dict
from .._util import _is_na
import numpy as np
import pandas as pd
from typing import Union


def clip_and_count(
    adata: AnnData,
    target_col: str,
    *,
    groupby: Union[str, None] = None,
    clip_at: int = 3,
    inplace: bool = True,
    key_added: Union[str, None] = None,
    fraction: bool = True,
) -> Union[None, np.ndarray]:
    """Counts the number of identical entries in `target_col` 
    for each group in `group_by`. 

    Counts NaN values like any other value in `target_col`. 

    Parameters
    ----------
    adata
        AnnData object to work on
    target_col
        Column to count on.
    groupby
        Calculate counts within groups (e.g. sample or patient) 
    clip_at
        All entries in `target_col` with more copies than `clip_at`
        will be summarized into a single group.         
    inplace
        If True, adds a column to `obs`. Otherwise returns an array 
        with the clipped counts. 
    key_added
        Key under which the results will be stored in `obs`. Defaults
        to `{target_col}_clipped_count`

    Returns
    -------
    Depending on the value of inplace, adds a column to adata or returns
    an array with the clipped count per cell. 
    """
    if target_col not in adata.obs.columns:
        raise ValueError("`target_col` not found in obs.")

    groupby = [groupby] if isinstance(groupby, str) else groupby
    groupby_cols = [target_col] if groupby is None else groupby + [target_col]
    clonotype_counts = (
        adata.obs.groupby(groupby_cols)
        .size()
        .reset_index(name="tmp_count")
        .assign(
            tmp_count=lambda X: [
                ">= {}".format(min(n, clip_at)) if n >= clip_at else str(n)
                for n in X["tmp_count"].values
            ]
        )
    )
    clipped_count = adata.obs.merge(clonotype_counts, how="left", on=groupby_cols)[
        "tmp_count"
    ].values

    if inplace:
        key_added = (
            "{}_clipped_count".format(target_col) if key_added is None else key_added
        )
        adata.obs[key_added] = clipped_count
    else:
        return clipped_count


def clonal_expansion(
    adata: AnnData,
    target_col: str = "clonotype",
    *,
    groupby: Union[str, None] = None,
    clip_at: int = 3,
    key_added: str = "clonal_expansion",
    **kwargs,
) -> Union[None, np.ndarray]:
    """Adds a column to obs which clonotypes are expanded. 

    Parameters
    ----------
    adata
        Annoated data matrix
    target_col
        Column containing the clontype annoataion
    groupby
        Calculate clonal expansion within groups. Usually makes sense to set
        this to the column containing sample annotation. 
    clip_at:
        All clonotypes with more than `clip_at` clones will be summarized into 
        a single category
    key_added
        Key under which the results will be added to `obs`. 
    kwargs
        Additional arguments passed to :func:`scirpy.tl.clip_and_count`. 
    """
    return clip_and_count(
        adata,
        target_col,
        groupby=groupby,
        clip_at=clip_at,
        key_added=key_added,
        **kwargs,
    )
