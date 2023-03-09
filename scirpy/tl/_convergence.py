from typing import Optional

import pandas as pd

from ..util import DataHandler


@DataHandler.inject_param_docs()
def clonotype_convergence(
    adata: DataHandler.TYPE,
    *,
    key_coarse: str,
    key_fine: str,
    key_added: str = "is_convergent",
    inplace=True,
    airr_mod: str = "airr",
) -> Optional[pd.Series]:
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
    {adata}
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
    params = DataHandler(adata, airr_mod)
    obs = params.get_obs([key_coarse, key_fine])
    convergence_df = (
        obs.groupby([key_coarse, key_fine], observed=True)
        .size()
        .reset_index()
        .groupby(key_coarse)
        .size()
        .reset_index()
    )
    convergent_clonotypes = convergence_df.loc[convergence_df[0] > 1, key_coarse]
    result = obs[key_coarse].isin(convergent_clonotypes)
    result = pd.Series(
        pd.Categorical(
            ["convergent" if x else "not convergent" for x in result],
            categories=["convergent", "not convergent"],
        ),
        index=obs.index,
    )
    if inplace:
        # Let's store in both anndata and mudata. Depending on which columns
        # were chosen, they might originate from either mudata or anndata.
        # TODO #356: can we do that more systematically and maybe handle through DataHandler?
        # possibly add additional attribute where to store...
        try:
            params.mdata.obs[f"{airr_mod}:{key_added}"] = result
        except AttributeError:
            pass
        params.adata.obs[key_added] = result
    else:
        return result
