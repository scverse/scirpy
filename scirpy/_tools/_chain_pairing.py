from ..util import _is_na, _is_true
from anndata import AnnData
from typing import Union
import numpy as np
from scanpy import logging


def chain_pairing(
    adata: AnnData, *, inplace: bool = True, key_added: str = "chain_pairing"
) -> Union[None, np.ndarray]:
    """Categorize cells based on how many TRA and TRB chains they have.

    Parameters
    ----------
    adata
        Annotated data matrix
    inplace
        If True, adds a column to adata.obs
    key_added
        Column name to add to 'obs'

    Returns
    -------
    Depending on the value of `inplace`, either
    returns a Series with a chain pairing category for each cell 
    or adds a `chain_pairing` column to `adata`. 
    
    """
    x = adata.obs
    string_length = len("Two full chains")
    results = np.empty(dtype=f"<U{string_length}", shape=(x.shape[0],))

    logging.debug("Done initalizing")

    mask_has_tcr = _is_true(x["has_tcr"].values)
    mask_multichain = mask_has_tcr & _is_true(x["multi_chain"].values)

    logging.debug("Done with masks part 1")

    mask_has_tra1 = ~_is_na(x["TRA_1_cdr3"].values)
    mask_has_trb1 = ~_is_na(x["TRB_1_cdr3"].values)
    mask_has_tra2 = ~_is_na(x["TRA_2_cdr3"].values)
    mask_has_trb2 = ~_is_na(x["TRB_2_cdr3"].values)

    logging.debug("Done with masks part 2")

    for m in [mask_has_tra1, mask_has_trb1, mask_has_tra2, mask_has_trb2]:
        # no cell can have a cdr3 sequence but no TCR
        assert np.setdiff1d(np.where(m)[0], np.where(mask_has_tcr)[0]).size == 0

    results[~mask_has_tcr] = "No TCR"
    results[mask_has_tra1] = "Orphan alpha"
    results[mask_has_trb1] = "Orphan beta"
    results[mask_has_tra1 & mask_has_trb1] = "Single pair"
    results[mask_has_tra1 & mask_has_trb1 & mask_has_tra2] = "Extra alpha"
    results[mask_has_tra1 & mask_has_trb1 & mask_has_trb2] = "Extra beta"
    results[
        mask_has_tra1 & mask_has_trb1 & mask_has_tra2 & mask_has_trb2
    ] = "Two full chains"
    results[mask_multichain] = "Multichain"

    assert "" not in results, "One or more chains are not characterized"

    if inplace:
        adata.obs[key_added] = results
    else:
        return results
