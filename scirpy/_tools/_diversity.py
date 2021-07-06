import numpy as np
from ..util import _is_na
from anndata import AnnData
import pandas as pd
from typing import Union
from ..io._util import _check_upgrade_schema


@_check_upgrade_schema()
def alpha_diversity(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = "clone_id",
    metric: str = "normalized_shannon_entropy",
    perc: float = 0.5,
    inplace: bool = True,
    key_added: Union[None, str] = None,
) -> pd.DataFrame:
    """Computes the alpha diversity of clonotypes within a group.

    Use a metric out of normalized shannon entropy, D50, DXX, and `scikit-bio’s alpha diversity metrics
    <http://scikit-bio.org/docs/latest/generated/skbio.diversity.alpha.html#module-skbio.diversity.alpha>`__.


    Normalized shannon entropy:
    Uses the `Shannon Entropy <https://mathworld.wolfram.com/Entropy.html>`__ as
    diversity measure. The Entrotpy gets
    `normalized to group size <https://math.stackexchange.com/a/945172>`__.

    D50:
    The diversity index (D50) is a measure of the diversity of an immune repertoire of J individual cells
    (the total number of CDR3s) composed of S distinct CDR3s in a ranked dominance configuration where ri
    is the abundance of the ith most abundant CDR3, r1 is the abundance of the most abundant CDR3, r2 is the
    abundance of the second most abundant CDR3, and so on. C is the minimum number of distinct CDR3s,
    amounting to >50% of the total sequencing reads. D50 therefore is given by C/S x 100.
    `<https://patents.google.com/patent/WO2012097374A1/en>`__.

    DXX:
    Similar to D50 where XX indicates the percent of J (the total number of CDR3s)

    Ignores NaN values.

    Parameters
    ----------
    adata
        Annotated data matrix
    groupby
        Column of `obs` by which the grouping will be performed.
    target_col
        Column on which to compute the alpha diversity
    metric
        A metric used for diversity estimation out of normalized shannon entropy, D50, DXX,
        and scikit-bio’s alpha diversity metrics
    perc
        Desired percent for the DXX metric
    inplace
        If `True`, add a column to `obs`. Otherwise return a DataFrame
        with the alpha diversities.
    key_added
        Key under which the alpha diversity will be stored if inplace is `True`.
        Defaults to `alpha_diversity_{target_col}`.

    Returns
    -------
    Depending on the value of inplace returns a DataFrame with the alpha diversity
    for each group or adds a column to `adata.obs`.
    """
    # Could rely on skbio.math if more variants are required.
    def _shannon_entropy(freq):
        """Normalized shannon entropy according to
        https://math.stackexchange.com/a/945172
        """
        if len(freq) == 1:
            # the formula below is not defined for n==1
            return 0
        else:
            return -np.sum((freq * np.log(freq)) / np.log(len(freq)))

    def _dxx(freq, perc=0.5):
        """
        D50/DXX according to https://patents.google.com/patent/WO2012097374A1/en
        """
        freq = np.sort(freq)[::-1]
        prop, i = 0, 0

        while prop < perc:
            prop += freq[i]
            i += 1

        return i / len(freq) * 100

    ir_obs = adata.obs.loc[~_is_na(adata.obs[target_col]), :]
    clono_counts = (
        ir_obs.groupby([groupby, target_col], observed=True)
        .size()
        .reset_index(name="count")
    )

    diversity = dict()
    for k in sorted(ir_obs[groupby].unique()):
        tmp_counts = clono_counts.loc[clono_counts[groupby] == k, "count"].values

        if metric in ["normalized_shannon_entropy", "D50", "DXX"]:
            # calculate frequencies for these two metrics
            tmp_freqs = tmp_counts / np.sum(tmp_counts)
            np.testing.assert_almost_equal(np.sum(tmp_freqs), 1)

            if metric == "normalized_shannon_entropy":
                diversity[k] = _shannon_entropy(tmp_freqs)
            elif metric == "D50":
                diversity[k] = _dxx(tmp_freqs)
            else:
                diversity[k] = _dxx(tmp_freqs, perc=perc)
        else:
            # make skbio an optional dependency
            try:
                import skbio
            except ImportError:
                raise ImportError(
                    "Using scikit-bio’s alpha diversity metrics requires the "
                    "installation of `scikit-bio`. You can install it with "
                    "`pip install scikit-bio`."
                )
            else:
                # skbio.diversity takes count vectors as input and
                # takes care of unknown metrics
                diversity[k] = skbio.diversity.alpha_diversity(
                    metric, tmp_counts
                ).values[0]

    if inplace:
        key_added = f"{metric}_{target_col}" if key_added is None else key_added
        adata.obs[key_added] = adata.obs[groupby].map(diversity)
    else:
        return pd.DataFrame().from_dict(diversity, orient="index")
