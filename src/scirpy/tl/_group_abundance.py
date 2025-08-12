from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from scirpy.get import _has_ir
from scirpy.util import DataHandler, _is_na, _normalize_counts


def _group_abundance(
    ir_obs: pd.DataFrame,
    groupby: str,
    target_col: str,
    *,
    fraction: None | str | bool = None,
    sort: Literal["count", "alphabetical"] | Sequence[str] = "count",
) -> pd.DataFrame:
    # remove NA rows
    na_mask = _is_na(ir_obs[groupby]) | _is_na(ir_obs[target_col])
    ir_obs = ir_obs.loc[~na_mask, :]

    # normalize to fractions
    scale_vector = _normalize_counts(ir_obs, normalize=fraction, default_col=groupby)
    ir_obs = ir_obs.assign(count=1, weight=scale_vector)

    # Calculate distribution of lengths in each group. Use sum instead of count
    # to reflect weights
    group_counts = (
        ir_obs.loc[:, [groupby, target_col, "count", "weight"]]
        .groupby([groupby, target_col], observed=True)
        .sum()
        .reset_index()
        .rename(columns={"weight": "weighted_count"})
    )

    result_df = group_counts.pivot(index=groupby, columns=target_col, values="weighted_count").fillna(value=0.0)

    # required that we can still sort by abundance even if normalized
    result_df_count = group_counts.pivot(index=groupby, columns=target_col, values="count").fillna(value=0.0)

    # By default, the most abundant group should be the first on the plot,
    # therefore we need their order
    if isinstance(sort, str) and sort == "alphabetical":
        ranked_target = sorted(result_df.index)
    elif isinstance(sort, str) and sort == "count":
        ranked_target = result_df_count.apply(np.sum, axis=1).sort_values(ascending=False).index.values
    else:
        ranked_target = sort

    ranked_groups = result_df_count.apply(np.sum, axis=0).sort_values(ascending=False).index.values
    result_df = result_df.loc[ranked_target, ranked_groups]

    return result_df


@DataHandler.inject_param_docs()
def group_abundance(
    adata: DataHandler.TYPE,
    groupby: str,
    target_col: str = "has_ir",
    *,
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    fraction: None | str | bool = None,
    sort: Literal["count", "alphabetical"] | Sequence[str] = "count",
) -> pd.DataFrame:
    """\
    Summarizes the number/fraction of cells of a certain category by a certain group.

    Ignores NaN values.

    Parameters
    ----------
    {adata}
    groupby
        Group by this column from `obs`. E.g, sample, or group.
    target_col
        Caregorical variable from `obs` according to which the abundance/fractions
        will be computed. This defaults to "has_ir", simply counting
        the number of cells with a detected :term:`IR` by group.
    {airr_mod}
    {airr_key}
    {chain_idx_key}
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized.
    sort
        How to arrange the dataframe columns.
        Default is by the category count ("count").
        Other options are "alphabetical" or to provide a list of categories.
        By providing an explicit list, the DataFrame can also be subsetted to
        specific categories.

    Returns
    -------
    Returns a data frame with the number (or fraction) of cells per group.
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)

    # has_ir column not present by default since new data structure. As a workaround, we manually add it.
    # Eventually, this function needs to  be rewritten, see #232
    get_cols = [x for x in [groupby, target_col] if x != "has_ir"]
    if isinstance(fraction, str):
        get_cols.append(fraction)
    ir_obs = params.get_obs(get_cols)
    ir_obs["has_ir"] = "False"
    ir_obs.loc[params.adata.obs_names, "has_ir"] = _has_ir(params).astype(str)

    return _group_abundance(ir_obs, groupby, target_col=target_col, fraction=fraction, sort=sort)
