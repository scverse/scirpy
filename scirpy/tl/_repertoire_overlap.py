from anndata import AnnData
from typing import Union, Tuple
from scipy.spatial import distance as sc_distance
from scipy.cluster import hierarchy as sc_hierarchy
import pandas as pd
import numpy as np
from ..util import _is_na, _normalize_counts
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def repertoire_overlap(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clone_id",
    overlap_measure: str = "jaccard",
    overlap_threshold: Union[None, float] = None,
    fraction: Union[None, str, bool] = None,
    inplace: bool = True,
    added_key: str = "repertoire_overlap",
) -> Union[None, Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Compute distance between cell groups based on clonotype overlap.

    Adds parwise overlaps, distance matrix and linkage to `uns`.

    .. warning::

        This function is experimental and is likely to change in the future.

    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Column with group labels (e.g. samples, tussue source, diagnosis, etc).
    target_col
        Category that overlaps among groups (`clone_id` by default, but can
        in principle be any group or cluster)
    overlap_measure
        Any distance measure accepted by `scipy.spatial.distance`; by default it is `jaccard`.
    overlap_threshold
        The minimum required weight to accept presence.
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    inplace
        Whether results should be added to `uns` or returned directly.
    added_key
        Results will be added to `uns` under this key.


    Returns
    -------
    A DataFrame used by the pairwise scatterplot, distance matrix and linkage.
    """

    # Remove NA rows
    na_mask = _is_na(adata.obs[groupby]) | _is_na(adata.obs[target_col])
    df = adata.obs.loc[~na_mask, :]

    # Normalize to fractions
    df = df.assign(
        cell_weights=_normalize_counts(adata.obs, fraction)
        if isinstance(fraction, (bool, str)) or fraction is None
        else fraction
    )

    # Create a weighted matrix of clonotypes
    df = (
        df.groupby([target_col, groupby], observed=True)
        .agg({"cell_weights": "sum"})
        .reset_index()
    )
    df = df.pivot(index=groupby, columns=target_col, values="cell_weights")
    df = df.fillna(0)

    # Create a table of clonotype presence
    if (
        overlap_threshold is None
    ):  # Consider a fuction that finds an optimal threshold...
        overlap_threshold = 0
    pr_df = df.applymap(lambda x: 1 if x > overlap_threshold else 0)

    # Compute distances and linkage
    distM = sc_distance.pdist(pr_df, overlap_measure)
    linkage = sc_hierarchy.linkage(distM)

    if inplace:

        # Store calculated data
        adata.uns[added_key] = {
            "weighted": df,
            "distance": distM,
            "linkage": linkage,
        }

        return

    else:
        return df, distM, linkage
