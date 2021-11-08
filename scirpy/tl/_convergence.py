from anndata import AnnData
from typing import Optional
import pandas as pd
from pandas.core.arrays.categorical import Categorical
from ..io._util import _check_upgrade_schema
from ..util import _is_na


@_check_upgrade_schema()
def clonotype_convergence(
    adata: AnnData,
    *,
    key_coarse: str,
    key_fine: str,
    key_added: str = "is_convergent",
    inplace=True,
) -> Optional[Categorical]:
    """
    Finds evidence for :term:`Convergent evolution of clonotypes`.

    Compares different definitions of :term:`clonotypes <Clonotype>` or
    :term:`clonotype clusters <Clonotype cluster>` (e.g. clonotypes defined by
    nucleotide sequence identity and clonotype clusters defined by amino acid
    sequence identity). Annotates cells as *convergent*, if a "coarse" clonotype
    definition (amino acid sequence identity in the example) contains multiple
    "fine" clonotypes (nucleotide sequence identity in the example).

    Clonotype definitions may be derived using :func:`scirpy.tl.define_clonotypes` or
    :func:`scirpy.tl.define_clonotype_clusters`.

    Parameters
    ----------
    adata
        Annotated data matrix
    key_coarse
        Key in adata.obs holding the "coarse" clonotype cluster defintion. E.g.
        `ct_cluster_aa_identity`.
    key_fine
        Key in adata.obs holding the "fine" clonotype/clonotype cluster definition.
        E.g. `clone_id`
    key_added
        Key under which the result is stored in `adata.obs`.
    inplace
        If True, a column with the result will be stored in `adata.obs`. Otherwise
        the result is returned.

    Returns
    -------
    Depending on the value of `inplace`, either returns or adds to `adata` a categorical
    vector indicating for each cell whether it belongs to a "convergent clonotype".
    """
    convergence_df = (
        adata.obs.loc[:, [key_coarse, key_fine]]
        .groupby([key_coarse, key_fine], observed=True)
        .size()
        .reset_index()
        .groupby(key_coarse)
        .size()
        .reset_index()
    )
    convergent_clonotypes = convergence_df.loc[convergence_df[0] > 1, key_coarse]
    result = adata.obs[key_coarse].isin(convergent_clonotypes)
    result = pd.Categorical(
        ["convergent" if x else "not convergent" for x in result],
        categories=["convergent", "not convergent", "nan"],
    )
    result[_is_na(adata.obs[key_fine]) | _is_na(adata.obs[key_coarse])] = "nan"
    if inplace:
        adata.obs[key_added] = result
    else:
        return result
