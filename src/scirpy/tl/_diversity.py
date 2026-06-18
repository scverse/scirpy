import warnings
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import pandas as pd

from scirpy.util import DataHandler, _is_na


def _shannon_entropy(counts: np.ndarray):
    """Normalized shannon entropy according to
    https://math.stackexchange.com/a/945172
    """
    freqs = counts / np.sum(counts)
    np.testing.assert_almost_equal(np.sum(freqs), 1)

    if len(freqs) == 1:
        # the formula below is not defined for n==1
        return 0
    else:
        return -np.sum((freqs * np.log(freqs)) / np.log(len(freqs)))


def _dxx(counts: np.ndarray, *, percentage: int):
    """
    D50/DXX according to https://patents.google.com/patent/WO2012097374A1/en

    Parameters
    ----------
    percentage
        Percentage of J
    """
    freqs = counts / np.sum(counts)
    np.testing.assert_almost_equal(np.sum(freqs), 1)

    freqs = np.sort(freqs)[::-1]
    prop, i = 0, 0

    while prop < (percentage / 100):
        prop += freqs[i]
        i += 1

    return i / len(freqs) * 100


def _import_hillrep():
    """Lazily import :mod:`hillrep`, with an actionable error if it is missing."""
    try:
        import hillrep
    except ImportError:
        raise ImportError(
            "Coverage-based Hill diversity requires the `hillrep` package. "
            "You can install it with `pip install hillrep`."
        ) from None
    return hillrep


def _coverage_hill_profile(
    counts_by_group: dict[str, np.ndarray],
    q_values: list[float],
) -> pd.DataFrame:
    """Coverage-standardized Hill profile for a set of groups.

    This is the only place that talks to the estimation backend. It standardizes
    all groups to a common sample coverage (iNEXT's ``Cmax`` rule) and returns the
    point estimates of the Hill numbers, and it warns when the groups cannot be
    compared fairly at a shared coverage.

    Returns a ``DataFrame`` indexed by diversity order ``q`` with one column per group.
    """
    hillrep = _import_hillrep()

    assessment = hillrep.assess(counts_by_group)
    if assessment.verdict != "reliable":
        warnings.warn(
            "The groups cannot be compared at a fully reliable common coverage. "
            "Read the profile with caution.\n" + assessment.summary(),
            stacklevel=3,
        )

    tidy = hillrep.compare(counts_by_group, level="coverage", q=q_values, n_boot=0)
    profile = tidy.pivot(index="order_q", columns="assemblage", values="qD")
    profile.index.name = None
    profile.columns.name = None
    # preserve the input group order rather than the alphabetical pivot order
    return profile[list(counts_by_group)]


@DataHandler.inject_param_docs()
def hill_diversity_profile(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    airr_mod: str = "airr",
    q_min: float = 0,
    q_max: float = 2,
    q_step: float = 1,
) -> pd.DataFrame:
    """\
    Computes a coverage-standardized Hill diversity profile for a range of diversity orders (`q`).

    Hill numbers unify the common alpha diversity indices into a single family indexed
    by an order `q` (`q=0` is observed richness, `q=1` is the exponential of Shannon
    entropy, `q=2` is the inverse Simpson index). Naively plugging observed frequencies
    into the Hill formula yields values that grow with sequencing depth, so two samples
    sequenced to different depth look different even when the underlying repertoire is
    the same. This function instead standardizes all groups to a common sample coverage
    before reporting the profile, following the iNEXT framework
    `(Chao et al. 2014; Hsieh, Ma & Chao 2016) <https://doi.org/10.1890/13-0133.1>`__,
    so the profiles are comparable across depth. The estimation is delegated to the
    `hillrep <https://github.com/KilianMaire/hillrep>`__ package.

    When the groups cannot be standardized to a fully reliable shared coverage (for
    example because one group is heavily undersampled), a warning is emitted: sometimes
    the honest answer is that a fair comparison is not possible.

    Parameters
    ----------
    {adata}
    groupby
        Column of `obs` by which the grouping will be performed.
    target_col
        Column containing the clonotype annotation.
    {airr_mod}
    q_min
        Lowest (start) diversity order.
    q_max
        Highest (end) diversity order.
    q_step
        Step between consecutive diversity orders.

    Returns
    -------
    A `DataFrame` with one row per diversity order `q` and one column per group. The
    output flows directly into :func:`~scirpy.tl.convert_hill_table` and into plotting
    libraries such as seaborn.
    """
    params = DataHandler(adata, airr_mod)
    ir_obs = params.get_obs([target_col, groupby])
    ir_obs = ir_obs.loc[~_is_na(ir_obs[target_col]), :]
    clono_counts = ir_obs.groupby([groupby, target_col], observed=True).size().reset_index(name="count")

    counts_by_group = {
        str(k): cast(
            np.ndarray,
            cast(pd.Series, clono_counts.loc[clono_counts[groupby] == k, "count"]).to_numpy(),
        )
        for k in sorted(ir_obs[groupby].dropna().unique())
    }

    q_values = [float(q) for q in np.arange(q_min, q_max + q_step, q_step)]
    return _coverage_hill_profile(counts_by_group, q_values)


def convert_hill_table(
    diversity_profile: pd.DataFrame,
    convert_to: Literal["diversity", "evenness_factor", "relative_evenness"] = "diversity",
) -> pd.DataFrame:
    """\
    Converts a profile from :func:`~scirpy.tl.hill_diversity_profile` into other alpha diversity indices.

    See `Daly et al. 2018 <https://doi.org/10.1093/bib/bbx019>`__ for an overview of
    the indices and evenness measures.

    Parameters
    ----------
    diversity_profile
        A `DataFrame` produced by :func:`~scirpy.tl.hill_diversity_profile`. It must
        contain the diversity orders `0`, `1` and `2` in its index.
    convert_to
        Which conversion to perform:

        * `"diversity"` - the classical indices (observed richness, Shannon entropy,
          inverse Simpson, Gini-Simpson) derived from the Hill numbers.
        * `"evenness_factor"` - each Hill number divided by the observed richness.
        * `"relative_evenness"` - the log of each Hill number over the log of the
          observed richness.

    Returns
    -------
    A `DataFrame` whose rows are the requested indices (or diversity orders) and whose
    columns are the groups.
    """
    for required_q in (0, 1, 2):
        if required_q not in diversity_profile.index:
            raise ValueError(
                f"The profile is missing diversity order q={required_q}. "
                "`convert_hill_table` requires the orders 0, 1 and 2."
            )

    if convert_to == "diversity":
        richness = diversity_profile.loc[0]
        inverse_simpson = diversity_profile.loc[2]
        return pd.DataFrame(
            {
                "Observed richness": richness,
                "Shannon entropy": np.log(diversity_profile.loc[1]),
                "Inverse Simpson": inverse_simpson,
                "Gini-Simpson": 1 - 1 / inverse_simpson,
            }
        ).T

    if convert_to == "evenness_factor":
        return diversity_profile / diversity_profile.loc[0]

    if convert_to == "relative_evenness":
        return np.log(diversity_profile) / np.log(diversity_profile.loc[0])

    raise ValueError(
        f"Invalid `convert_to` value {convert_to!r}. Choose 'diversity', 'evenness_factor' or 'relative_evenness'."
    )


@DataHandler.inject_param_docs()
def alpha_diversity(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: str | Callable[[np.ndarray], int | float] = "normalized_shannon_entropy",
    inplace: bool = True,
    key_added: None | str = None,
    airr_mod: str = "airr",
    **kwargs,
) -> pd.DataFrame | None:
    """\
    Computes the alpha diversity of clonotypes within a group.

    Use a metric out of  `normalized_shannon_entropy`, `D50`, `DXX`, and `scikit-bio’s alpha diversity metrics
    <http://scikit-bio.org/docs/latest/generated/skbio.diversity.alpha.html#module-skbio.diversity.alpha>`__.
    Alternatively, provide a custom function to calculate the diversity based on count vectors
    as explained here `<http://scikit-bio.org/docs/latest/diversity.html>`__

    Normalized shannon entropy:
        Uses the `Shannon Entropy <https://mathworld.wolfram.com/Entropy.html>`__ as
        diversity measure. The Entrotpy gets
        `normalized to group size <https://math.stackexchange.com/a/945172>`__.

    D50:
        D50 is a measure of the minimum number of distinct clonotypes totalling greater than 50% of total clonotype
        counts in a given group, as a percentage out of the total number of clonotypes.
        Adapted from `<https://patents.google.com/patent/WO2012097374A1/en>`__.

    DXX:
        Similar to D50 where XX indicates the percentage of total clonotype counts threshold.
        Requires to pass the `percentage` keyword argument which can be within 0 and
        100.


    Ignores NaN values.

    Parameters
    ----------
    {adata}
    groupby
        Column of `obs` by which the grouping will be performed.
    target_col
        Column on which to compute the alpha diversity
    metric
        A metric used for diversity estimation out of `normalized_shannon_entropy`,
        `D50`, `DXX`, any of scikit-bio’s alpha diversity metrics, or a custom function.
    {inplace}
    {key_added}
        Defaults to `alpha_diversity_{{target_col}}`.
    {airr_mod}
    **kwargs
        Additional arguments passed to the metric function.

    Returns
    -------
    Depending on the value of inplace returns a DataFrame with the alpha diversity
    for each group or adds a column to `adata.obs`.
    """
    params = DataHandler(adata, airr_mod)
    ir_obs = params.get_obs([target_col, groupby])
    ir_obs = ir_obs.loc[~_is_na(ir_obs[target_col]), :]
    clono_counts = ir_obs.groupby([groupby, target_col], observed=True).size().reset_index(name="count")

    diversity = {}
    for k in sorted(ir_obs[groupby].dropna().unique()):
        tmp_counts = cast(
            np.ndarray,
            cast(pd.Series, clono_counts.loc[clono_counts[groupby] == k, "count"]).values,
        )

        if isinstance(metric, str):
            if metric == "normalized_shannon_entropy":
                diversity[k] = _shannon_entropy(tmp_counts)
            elif metric == "D50":
                diversity[k] = _dxx(tmp_counts, percentage=50)
            elif metric == "DXX":
                if "percentage" in kwargs:
                    diversity[k] = _dxx(tmp_counts, percentage=cast(int, kwargs.get("percentage")))
                else:
                    raise ValueError("DXX requires the `percentage` keyword argument, which can range from 0 to 100.")
            else:
                # make skbio an optional dependency
                try:
                    import skbio.diversity
                except ImportError:
                    raise ImportError(
                        "Using scikit-bio’s alpha diversity metrics requires the "
                        "installation of `scikit-bio`. You can install it with "
                        "`pip install scikit-bio`."
                    ) from None
                else:
                    # skbio.diversity takes count vectors as input and
                    # takes care of unknown metrics
                    diversity[k] = skbio.diversity.alpha_diversity(metric, tmp_counts).values[0]
        else:
            # calculate diversity using custom function
            diversity[k] = metric(tmp_counts)

    if inplace:
        metric_name = metric if isinstance(metric, str) else metric.__name__
        key_added = f"{metric_name}_{target_col}" if key_added is None else key_added
        params.set_obs(key_added, params.get_obs(groupby).map(diversity))
    else:
        return pd.DataFrame().from_dict(diversity, orient="index")
