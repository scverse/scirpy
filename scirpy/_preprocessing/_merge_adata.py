from typing import Union, List, Dict
from anndata import AnnData
from ..io._convert_anndata import (
    _sanitize_anndata,
    IR_OBS_COLS,
    to_ir_objs,
    from_ir_objs,
)
from ..io._datastructures import IrCell
from scanpy import logging
import itertools


def merge_ir_adatas(adata1: AnnData, adata2: AnnData) -> AnnData:
    """
    Merge two AnnData objects with :term:`IR` information (e.g. BCR with TCR).

    Decomposes the IR information back into :class:`scirpy.io.IrCell` objects
    and merges them on a chain-level. If both objects contain the same cell-id, and
    the same chains, the corresponding row in `adata.obs` will be unchanged.
    If both objects contain the same cell-id, but different chains, the chains
    will be merged and the cell annotated as :term:`ambiguous<Receptor type>` or
    :term:`multi-chain<Multichain-cell>` if appropriate.

    Discards everything not in `adata.obs` (e.g. gene expression, UMAP coordinates,
    etc.) and keeps non-IR related columns from `adata1`, but discards them from
    `adata2`.

    Parameters
    ----------
    adata
        first AnnData object containing IR information
    adata2
        second AnnData object containint IR information

    Returns
    -------
    Single adata object with the merged IR information.
    """
    ir_objs1 = to_ir_objs(adata1)
    ir_objs2 = to_ir_objs(adata2)
    cell_dict: Dict[str, IrCell] = dict()
    for cell in itertools.chain(ir_objs1, ir_objs2):
        try:
            tmp_cell = cell_dict[cell.cell_id]
            tmp_cell.multi_chain |= cell.multi_chain
            tmp_cell.chains.extend(cell.chains)
        except KeyError:
            cell_dict[cell.cell_id] = cell

    # remove duplicate chains
    for cell in cell_dict.values():
        cell.chains = list(set(cell.chains))

    adata_merged = from_ir_objs(cell_dict.values())
    obs_merged = adata1.obs.drop(IR_OBS_COLS, axis="columns").merge(
        adata_merged.obs,
        left_index=True,
        right_index=True,
        validate="one_to_one",
        on=None,
    )
    return obs_merged


def merge_with_ir(
    adata: AnnData,
    adata_ir: AnnData,
    *,
    how: str = "left",
    on: Union[List[str], str] = None,
    left_index: bool = True,
    right_index: bool = True,
    validate: str = "one_to_one",
    **kwargs
) -> None:
    """Merge adaptive immune receptor (:term:`IR`) data with transcriptomics data into a
    single :class:`~anndata.AnnData` object.

    :ref:`Reading in IR data<importing-data>` results in an :class:`~anndata.AnnData`
    object with IR information stored in `obs`. Use this function to merge
    it with another :class:`~anndata.AnnData` which contains transcriptomics data.

    Will keep all objects (e.g. `neighbors`, `umap`) from `adata` and integrate
    `obs` from `adata_ir` into `adata`.
    Everything other than `.obs` from `adata_ir` will be discarded.

    This function uses :func:`pandas.merge` to join the two `.obs` data frames.

    Modifies `adata` inplace.

    Parameters
    ----------
    adata
        AnnData with the transcriptomics data. Will be modified inplace.
    adata_ir
        AnnData with the adaptive immune receptor (IR) data
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
        # since we are merging on index, no additional columns will be considered
        # when on is None.
        if ("batch" in adata.obs.columns) and ("batch" in adata_ir.obs.columns):
            on = "batch"

    if "has_ir" in adata.columns:
        logging.warning(
            "It seems you already have immune receptor (IR) data in `adata`. "
            "Merging IR objects by chain. "
        )
        adata.obs = merge_ir_adatas(adata, adata_ir)
    else:
        adata.obs = adata.obs.merge(
            adata_ir.obs,
            how=how,
            on=on,
            left_index=left_index,
            right_index=right_index,
            validate=validate,
            **kwargs
        )

    _sanitize_anndata(adata)
