from locale import normalize

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sc_hierarchy
from scipy.spatial import distance as sc_distance

from scirpy.util import DataHandler, _is_na, _normalize_counts


@DataHandler.inject_param_docs()
def repertoire_overlap(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    overlap_measure: str = "jaccard",
    overlap_threshold: None | float = None,
    fraction: str | bool = False,
    inplace: bool = True,
    added_key: str = "repertoire_overlap",
    airr_mod: str = "airr",
) -> None | tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """\
    Compute distance between cell groups based on clonotype overlap.

    Adds parwise overlaps, distance matrix and linkage to `uns`.

    .. warning::

        This function is experimental and is likely to change in the future.

    Parameters
    ----------
    {adata}
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
        providing cell weights directly. Setting it to `False`, assigns equal weight
        to all cells.
    inplace
        Whether results should be added to `uns` or returned directly.
    added_key
        Results will be added to `uns` under this key.
    {airr_mod}


    Returns
    -------
    A DataFrame used by the pairwise scatterplot, distance matrix and linkage.
    """
    params = DataHandler(adata, airr_mod)
    obs = params.get_obs([groupby, target_col])
    if isinstance(normalize, str):
        obs[normalize] = params.get_obs(normalize)

    # Remove NA rows
    na_mask = _is_na(obs[groupby]) | _is_na(obs[target_col])
    df = obs.loc[~na_mask, :].copy()

    # Normalize to fractions
    df["cell_weights"] = (
        _normalize_counts(df, fraction) if isinstance(fraction, bool | str) or fraction is None else fraction
    )

    # Create a weighted matrix of clonotypes
    df = df.groupby([target_col, groupby], observed=True).agg({"cell_weights": "sum"}).reset_index()
    df = df.pivot(index=groupby, columns=target_col, values="cell_weights")
    df = df.fillna(0)

    # Create a table of clonotype presence
    if overlap_threshold is None:  # Consider a fuction that finds an optimal threshold...
        overlap_threshold = 0
    pr_df = df.applymap(lambda x: 1 if x > overlap_threshold else 0)

    # Compute distances and linkage
    distM = sc_distance.pdist(pr_df, overlap_measure)  # type:ignore
    linkage = sc_hierarchy.linkage(distM)

    if inplace:
        # Store calculated data
        params.adata.uns[added_key] = {
            "weighted": df,
            "distance": distM,
            "linkage": linkage,
        }

        return

    else:
        return df, distM, linkage
