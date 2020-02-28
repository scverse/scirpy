from typing import Union, List
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
    
    Parameters
    ----------
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
