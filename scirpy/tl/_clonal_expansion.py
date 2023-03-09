from typing import List, Literal, Union

import pandas as pd

from ..util import DataHandler, _is_na, _normalize_counts


def _clip_and_count(
    adata: DataHandler.TYPE,
    target_col: str,
    *,
    groupby: Union[str, None, List[str]] = None,
    clip_at: int = 3,
    inplace: bool = True,
    key_added: Union[str, None] = None,
    fraction: bool = True,
    airr_mod="airr",
) -> Union[None, pd.Series]:
    """Counts the number of identical entries in `target_col`
    for each group in `group_by`.

    `nan`s in the input remain `nan` in the output.
    """
    params = DataHandler(adata, airr_mod)
    if target_col not in params.adata.obs.columns:
        raise ValueError("`target_col` not found in obs.")

    groupby = [groupby] if isinstance(groupby, str) else groupby
    groupby_cols = [target_col] if groupby is None else groupby + [target_col]
    clonotype_counts = (
        params.adata.obs.groupby(groupby_cols, observed=True)
        .size()
        .reset_index(name="tmp_count")
        .assign(
            tmp_count=lambda X: [
                ">= {}".format(min(n, clip_at)) if n >= clip_at else str(n)
                for n in X["tmp_count"].values
            ]
        )
    )
    clipped_count = params.adata.obs.merge(
        clonotype_counts, how="left", on=groupby_cols
    )["tmp_count"]
    clipped_count[_is_na(params.adata.obs[target_col])] = "nan"
    clipped_count.index = params.adata.obs.index

    if inplace:
        key_added = (
            "{}_clipped_count".format(target_col) if key_added is None else key_added
        )
        params.adata.obs[key_added] = clipped_count
    else:
        return clipped_count


@DataHandler.inject_param_docs()
def clonal_expansion(
    adata: DataHandler.TYPE,
    *,
    target_col: str = "clone_id",
    expanded_in: Union[str, None] = None,
    clip_at: int = 3,
    key_added: str = "clonal_expansion",
    inplace: bool = True,
    **kwargs,
) -> Union[None, pd.Series]:
    """Adds a column to `obs` recording which clonotypes are expanded.

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
    clip_at:
        All clonotypes with more than `clip_at` clones will be summarized into
        a single category
    key_added
        Key under which the results will be added to `obs`.
    inplace
        If True, adds a column to `obs`. Otherwise returns an array
        with the clipped counts.
    {airr_mod}

    Returns
    -------
    Depending on the value of inplace, adds a column to adata or returns
    a Series with the clipped count per cell.
    """
    return _clip_and_count(
        adata,
        target_col,
        groupby=expanded_in,
        clip_at=clip_at,
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
