from collections.abc import Callable
from typing import cast

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
        params.set_obs(key_added, params.adata.obs[groupby].map(diversity))
    else:
        return pd.DataFrame().from_dict(diversity, orient="index")
