from typing import Sequence, Literal
import pandas as pd
import numpy as np
import awkward._v2 as ak


def airr(
    adata,
    airr_variable,
    chain: Literal["vj1", "vdj1", "vj2", "vdj2"],
):
    """Retrieve AIRR variables for each cell"""
    # TODO VJ_1 vs vj1?
    # chain = chain.lower().replace("_", "")
    # valid_chains = ["vj1", "vdj1", "vj2", "vdj2"]
    # if chain not in valid_chains:
    #     raise ValueError(
    #         f"Invalid value for chain. Valid values are {', '.join(valid_chains)}"
    #     )
    if airr_variable not in adata.var_names:
        raise ValueError("airr_variable is not in adata.var_names")

    idx = adata.obsm["chain_indices"][chain]
    # TODO ensure that this doesn't get converted to something not supporting missing values
    # when saving anndata
    mask = ~pd.isnull(idx)
    result = np.full(idx.shape, fill_value=None, dtype=object)
    # to_numpy fails with some dtypes and missing values are only supported through masked arrays
    # TODO to_numpy would surely be faster than to_list. Maybe this can be updated in the future.
    result[mask] = ak.to_list(
        adata.X[
            np.where(mask)[0],
            np.where(adata.var_names == airr_variable)[0],
            idx[mask].astype(int),
        ],
    )
    return pd.Series(result, index=adata.obs_names)


def most_frequent(array: Sequence, n=10):
    return pd.Series(array).value_counts().index[:n].tolist()
