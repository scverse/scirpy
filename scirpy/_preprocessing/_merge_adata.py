from typing import Union, List
from anndata import AnnData
from ..io._io import _sanitize_anndata


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
    """Merge TCR data with transcriptomics data into a single :class:`~anndata.AnnData` 
    object. 

    :ref:`Reading in TCR data<importing-data>` results in an :class:`~anndata.AnnData`
    object with TCR information stored in `obs`. Use this function to merge
    it with another :class:`~anndata.AnnData` which contains transcriptomics data. 

    Will keep all objects (e.g. `neighbors`, `umap`) from `adata` and integrate 
    `obs` from `adata_tcr` into `adata`. 
    Everything other than `.obs` from `adata_tcr` will be discarded. 

    This function uses :func:`pandas.merge` to join the two `.obs` data frames. 

    Modifies `adata` inplace. 

    Parameters
    ----------
    adata
        AnnData with the transcriptomics data. Will be modified inplace. 
    adata_tcr
        AnnData with the TCR data
    on
        Columns to join on. Default: The index and "batch", if it exists in both `obs`.
    left_index
        See :func:`pandas.merge`. 
    right_index
        See :func:`pandas.merge`.  
    validate
        See :func:`pandas.merge`. 
    **kwargs
        Additional kwargs are passed to :func:`pandas.merge`. 
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

    _sanitize_anndata(adata)
