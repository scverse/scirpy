from .._util import _is_na, _is_true
from anndata import AnnData
from typing import Union
import pandas as pd
import numpy as np


def _decide_chain_cat(x: pd.Series) -> str:
    """Helper function of `chain_pairing`. Associates categories to  a cell based 
    on how many TRA and TRB chains they have.

    Parameters
    ----------
    x
        A Series contaning immune receptor chain information of a single cell.

    Returns
    -------
    The category the cell belongs to. 
    
    """
    if _is_true(x["has_tcr"]):
        if not _is_true(x["multi_chain"]):
            if not _is_na(x["TRA_1_cdr3"]):
                if not _is_na(x["TRB_1_cdr3"]):
                    if not _is_na(x["TRA_2_cdr3"]):
                        if not _is_na(x["TRB_2_cdr3"]):
                            return "Two full chains"
                        else:
                            return "Extra alpha"
                    else:
                        if not _is_na(x["TRB_2_cdr3"]):
                            return "Extra beta"
                        else:
                            return "Single pair"
                else:
                    return "Orphan alpha"
            else:
                if not _is_na(x["TRB_1_cdr3"]):
                    return "Orphan beta"
        else:
            return "Multichain"
    else:
        return "No TCR"
    assert False, "Chain not characterized"


def chain_pairing(
    adata: AnnData, *, inplace: bool = True, key_added: str = "chain_pairing"
) -> Union[None, np.ndarray]:
    """Associate categories to cells based on how many TRA and TRB chains they have.

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

    cp_col = (
        adata.obs.loc[
            :,
            [
                "has_tcr",
                "multi_chain",
                "TRA_1_cdr3",
                "TRA_2_cdr3",
                "TRB_1_cdr3",
                "TRB_2_cdr3",
            ],
        ]
        .apply(_decide_chain_cat, axis=1)
        .values
    )

    if inplace:
        adata.obs[key_added] = cp_col
    else:
        return cp_col
