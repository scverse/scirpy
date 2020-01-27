import pandas as pd
from typing import Union, List
import numpy as np
from anndata import AnnData


def merge_with_tcr(
    adata: AnnData,
    adata_tcr: AnnData,
    *,
    how: str = "left",
    on: Union[List[str], str] = None,
    left_index: bool = True,
    right_index: bool = True,
    validate: str = "one_to_one",
    **kwargs
) -> None:
    """Integrate the TCR AnnData into an existing AnnData object with transcriptomics data.  

    Will keep all objects from `adata_tx` and integrate `obs` from adata_tcr
    into `adata_tx`. Everything other than `.obs` from adata_tcr will be lost. 

    `.obs` will be merged using `pandas.merge`. Additional kwargs are passed to 
    `pandas.merge`. 
    
    adata
        AnnData with the transcriptomics data. Will be modified inplace. 
    adata_tcr
        AnnData with the TCR data
    on
        Columns to join on. Default: The index and "batch", if it exists in both `obs`. 
    """
    if on is None:
        if ("batch" in adata.obs.columns) and ("batch" in adata_tcr.obs.columns):
            on = "batch"

    adata.obs = adata.obs.merge(
        adata_tcr.obs,
        how=how,
        on=on,
        left_index=left_index,
        right_index=right_index,
        validate=validate,
        **kwargs
    )


def filter_tcrs(adata: AnnData, mask):
    """remove certain TCRs, but keep the cells
    
    Parameters
    ----------
    adata : AnnData
        [description]
    mask : [type]
        [description]
    """
    pass


def define_clonotypes(adata: AnnData, *, flavor: str = "paired") -> None:
    """Define clonotypes based on CDR3 region. 
    
    Parameters
    ----------
    adata
    flavor
        Currently, only "paried" is supported. 
    
    """
    assert flavor == "paired", "Other flavors currently not supported"

    clonotype_col = np.array(
        [
            "clonotype_{}".format(x)
            for x in adata.obs.groupby(
                ["TRA_1_cdr3", "TRB_1_cdr3", "TRA_2_cdr3", "TRA_2_cdr3"]
            ).ngroup()
        ]
    )
    # TODO this check needs to be improved (or make sure that it's always categorical)
    clonotype_col[adata.obs["has_tcr"] != "True"] = np.nan
    adata.obs["clonotype"] = clonotype_col
