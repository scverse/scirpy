import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from scirpy.util import DataHandler, _is_na, _normalize_counts


def _clip_and_count(
    adata: DataHandler.TYPE,
    target_col: str,
    *,
    groupby: str | None | list[str] = None,
    breakpoints: Sequence[int] = (1, 2, 3),
    inplace: bool = True,
    key_added: str | None = None,
    airr_mod="airr",
) -> None | pd.Series:
    """Counts the number of identical entries in `target_col`
    for each group in `group_by`.

    `nan`s in the input remain `nan` in the output.
    """
    params = DataHandler(adata, airr_mod)
    if not len(breakpoints):
        raise ValueError("Need to specify at least one breakpoint.")

    categories = [f"<= {b}" for b in breakpoints] + [f"> {breakpoints[-1]}", "nan"]

    @np.vectorize
    def _get_interval(value: int) -> str:
        """Return the interval of `value`, given breakpoints."""
        for b in breakpoints:
            if value <= b:
                return f"<= {b}"
        return f"> {b}"

    groupby = [groupby] if isinstance(groupby, str) else groupby
    groupby_cols = [target_col] if groupby is None else groupby + [target_col]
    obs = params.get_obs(groupby_cols)

    clonotype_counts = (
        obs.groupby(groupby_cols, observed=True)
        .size()
        .reset_index(name="tmp_count")
        .assign(tmp_count=lambda X: pd.Categorical(_get_interval(X["tmp_count"].values), categories=categories))
    )
    clipped_count = obs.merge(clonotype_counts, how="left", on=groupby_cols)["tmp_count"]
    clipped_count[_is_na(obs[target_col])] = "nan"
    clipped_count.index = obs.index

    if inplace:
        key_added = f"{target_col}_clipped_count" if key_added is None else key_added
        params.set_obs(key_added, clipped_count)
    else:
        return clipped_count


@DataHandler.inject_param_docs()
def clonal_expansion(
    adata: DataHandler.TYPE,
    *,
    target_col: str = "clone_id",
    expanded_in: str | None = None,
    breakpoints: Sequence[int] = (1, 2),
    clip_at: int | None = None,
    key_added: str = "clonal_expansion",
    inplace: bool = True,
    **kwargs,
) -> None | pd.Series:
    """\
    Adds a column to `obs` recording which clonotypes are expanded.

    `nan`s in the clonotype column remain `nan` in the output.

    Parameters
    ----------
    {adata}
    target_col
        Column containing the clontype annoataion
    expanded_in
        Calculate clonal expansion within groups. Usually makes sense to set
        this to the column containing sample annotation. If set to None,
        a clonotype counts as expanded if there's any cell of the same clonotype
        across the entire dataset.
    breakpoints
        summarize clonotypes with a size smaller or equal than the specified numbers
        into groups. For instance, if this is (1, 2, 5), there will be four categories:

        * all clonotypes with a size of 1 (singletons)
        * all clonotypes with a size of 2
        * all clonotypes with a size between 3 and 5 (inclusive)
        * all clonotypes with a size > 5
    clip_at
        This argument is superseded by `breakpoints` and is only kept for backwards-compatibility.
        Specifying a value of `clip_at = N` equals to specifying `breakpoints = (1, 2, 3, ..., N)`
        Specifying both `clip_at` overrides `breakpoints`.
    {key_added}
    {inplace}
    {airr_mod}

    Returns
    -------
    Depending on the value of inplace, adds a column to adata or returns
    a Series with the clipped count per cell.
    """
    if clip_at is not None:
        breakpoints = list(range(1, clip_at))
        warnings.warn("The argument `clip_at` is deprecated. Please use `brekpoints` instead.", category=FutureWarning)
    return _clip_and_count(
        adata,
        target_col,
        groupby=expanded_in,
        breakpoints=breakpoints,
        key_added=key_added,
        inplace=inplace,
        **kwargs,
    )


@DataHandler.inject_param_docs()
def summarize_clonal_expansion(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    summarize_by: Literal["cell", "clone_id"] = "cell",
    normalize: bool = False,
    airr_mod: str = "airr",
    **kwargs,
) -> pd.DataFrame:
    """
    Summarizes clonal expansion by a grouping variable.

    Removes all entries with `NaN` in `target_col` prior to summarization.

    Parameters
    ----------
    {adata}
    groupby
        summarize by this column in `adata.obs`.
    target_col
        column in obs which holds the clonotype information
    summarize_by
        Can be either `cell` to count cells belonging to a clonotype (the default),
        or "clone_id" to count clonotypes. The former leads to a over-representation
        of expanded clonotypes but better represents the fraction of expanded cells.
    normalize
        If `False`, plot absolute cell counts. If `True`, scale each group to 1.
    {airr_mod}
    **kwargs
        Additional arguments passed to :func:`clonal_expansion`.

    Returns
    -------
    A DataFrame with one row for each unique value in `groupby`.
    """
    params = DataHandler(adata, airr_mod)
    if summarize_by not in ["clone_id", "cell"]:
        raise ValueError("Invalue value for `summarize_by`. ")

    expansion = clonal_expansion(params, target_col=target_col, inplace=False, **kwargs)

    tmp_col = target_col + "_clipped_count"
    tmp_col_weight = target_col + "_weight"

    obs = params.get_obs([groupby, target_col])
    obs[tmp_col] = expansion

    # filter NA values
    obs = obs.loc[~_is_na(obs[target_col]), :]

    if summarize_by == "clone_id":
        obs.drop_duplicates(inplace=True)

    # add normalization vector
    size_vector = _normalize_counts(obs, normalize, groupby)
    obs[tmp_col_weight] = size_vector

    obs = (
        obs.groupby([groupby, tmp_col], observed=True)[tmp_col_weight]
        .sum()
        .reset_index()
        .pivot(index=groupby, columns=tmp_col, values=tmp_col_weight)
        .fillna(0)
    )

    return obs
