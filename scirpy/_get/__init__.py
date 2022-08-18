from typing import Literal
import pandas as pd
import numpy as np


def get(*args, **kwargs):
    return AirrGetter(*args, **kwargs)


class AirrGetter:
    def __init__(
        self,
        adata,
        variable,
        receptor_arm: Literal["VJ", "VDJ"],
        dual_ir: Literal[1, 2] = 1,
    ):
        pass

    def _airr(
        adata,
        airr_variable,
        receptor_arm: Literal["VJ", "VDJ"],
        dual_ir: Literal[1, 2] = 1,
    ):
        """Retrieve AIRR variables for each cell"""
        if receptor_arm not in ["VJ", "VDJ"]:
            raise ValueError("Valid values for receptor_arm are: 'VJ', 'VDJ'")
        if dual_ir not in [1, 2]:
            raise ValueError("Valid values for dual_ir are: 1, 2")

        idx = adata.obsm["chain_indices"][f"{receptor_arm}_{dual_ir}"]
        # TODO ensure that this doesn't get converted to something not supporting missing values
        # when saving anndata
        mask = ~pd.isnull(idx)
        result = np.full(idx.shape, fill_value=None, dtype=object)
        result[mask] = adata.X[
            np.where(mask)[0],
            np.where(adata.var_names == airr_variable)[0],
            idx[mask].astype(int),
        ]
        return result

    def most_frequent(self, n=10):
        pass
