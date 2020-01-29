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


def _process_tcr_cell(tcr_cell: object):
    """Filter chains to our working model of TCRs

    i.e.
     * each cell can contain at most four chains. 
     * remove non-productive chains
     * at most two copies for each chain chain_type (alpha, beta)
     * if there are more than four chains, the most abundant ones will be taken
       and a warning raised. 

    
    Parameters
    ----------
    adata : AnnData
        [description]
    """
    for tcr_obj in adata.obs["tcr_objs"]:
        if pd.isnull(tcr_obj):
            continue
        tra_chains = sorted(
            [x for x in tcr_obj.chains if x.chain_type == "TRA" and x.is_productive],
            key=lambda x: x.expr,
            reverse=True,
        )
        trb_chains = sorted(
            [x for x in tcr_obj.chains if x.chain_type == "TRB" and x.is_productive],
            key=lambda x: x.expr,
            reverse=True,
        )
        if len(tra_chains) > 2:
            # TODO logging
            print(
                "More than 2 TRA chains for cell {}: {}. "
                "Truncated to two most abundant productive chains. ".format(
                    tcr_obj.cell_id, len(tra_chains)
                )
            )
            tra_chains = tra_chains[:2]
        if len(trb_chains) > 2:
            print(
                "More than 2 TRB chains for cell {}: {}. "
                "Truncated to two most abundant productive chains. ".format(
                    tcr_obj.cell_id, len(trb_chains)
                )
            )
            trb_chains = trb_chains[:2]
        tcr_obj.chains = tra_chains + trb_chains


def filter_tcrs(adata: AnnData, mask):
    """remove certain TCRs, but keep the cells
    
    Parameters
    ----------
    adata : AnnData
        [description]
    mask : [chain_type]
        [description]
    """
    pass
