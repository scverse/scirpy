from textwrap import indent
from typing import Union, List, Dict
from anndata import AnnData
from ..io._convert_anndata import (
    _sanitize_anndata,
    to_airr_cells,
    from_airr_cells,
)
from ..io._datastructures import AirrCell
from ..io._util import _check_upgrade_schema
from scanpy import logging
import itertools
import pandas as pd


@_check_upgrade_schema(check_args=(0, 1))
def merge_airr_chains(adata: AnnData, adata2: AnnData) -> None:
    """
    Merge two AnnData objects with :term:`IR` information (e.g. BCR with TCR).

    Decomposes the IR information back into :class:`scirpy.io.AirrCell` objects
    and merges them on a chain-level. If both objects contain the same cell-id, and
    the same chains, the corresponding row in `adata.obs` will be unchanged.
    If both objects contain the same cell-id, but different chains, the chains
    will be merged into a single cell such that it can be annotated as
    :term:`ambiguous<Receptor type>` or :term:`multi-chain<Multichain-cell>`
    if appropriate.  If a cell contains both TCR and BCR chains, they will both
    be kept and can be identified as `ambiguous` using the :func:`scirpy.tl.chain_qc`
    function.

    The function performs a "left join", i.e. all cells not present in `adata` will
    be discarded. Of `adata2` the function only retains information from `obs`.

    To simply add IR information onto an existing `AnnData` object with transcriptomics
    data, see :func:`~scirpy.pp.merge_with_ir` (this function can do this, too, but
    `merge_with_ir` is more efficient).

    Modifies `adata` inplace.

    Parameters
    ----------
    adata
        first AnnData object containing IR information
    adata2
        second AnnData object containing IR information
    """
    ir_objs1 = to_airr_cells(adata)
    ir_objs2 = to_airr_cells(adata2)
    cell_dict: Dict[str, AirrCell] = dict()
    for cell in itertools.chain(ir_objs1, ir_objs2):
        try:
            tmp_cell = cell_dict[cell.cell_id]
            # this is a legacy operation. With adatas generated with scirpy
            # >= 0.7 this isn't necessary anymore, as all chains are preserved.
            tmp_cell["multi_chain"] |= cell["multi_chain"]
            for tmp_chain in cell.chains:
                tmp_cell.add_chain(tmp_chain)
            # add cell-level attributes
            tmp_cell.update(cell)
        except KeyError:
            cell_dict[cell.cell_id] = cell

    # remove duplicate chains
    # https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python
    for cell in cell_dict.values():
        cell._chains = [dict(t) for t in set(tuple(d.items()) for d in cell.chains)]

    # only keep entries that are in `adata` and ensure consistent ordering
    adata.obs = from_airr_cells(cell_dict.values()).obs.reindex(adata.obs_names)


@_check_upgrade_schema(check_args=(1,))
def merge_with_ir(
    adata: AnnData, adata_ir: AnnData, on: Union[List[str], None] = None, **kwargs
) -> None:
    """Merge adaptive immune receptor (:term:`IR`) data with transcriptomics data into a
    single :class:`~anndata.AnnData` object.

    :ref:`Reading in IR data<importing-data>` results in an :class:`~anndata.AnnData`
    object with IR information stored in `obs`. Use this function to merge
    it with another :class:`~anndata.AnnData` containing transcriptomics data.
    To add additional IR data on top of on top of an :class:`~anndata.AnnData`
    object that already contains IR information (e.g. :term:`BCR` on top of
    :term:`TCR` data.), see :func:`~scirpy.pp.merge_airr_chains`.

    Merging keeps all objects (e.g. `neighbors`, `umap`) from `adata` and integrates
    `obs` from `adata_ir` into `adata`. Everything other than `.obs` from `adata_ir`
    will be discarded.

    This function is a thin wrapper around :func:`pandas.merge`. The function performs
    a "left join", i.e. all cells not present in `adata` will be discarded.

    Modifies `adata` inplace.

    Parameters
    ----------
    adata
        AnnData with the transcriptomics data. Will be modified inplace.
    adata_ir
        AnnData with the adaptive immune receptor (IR) data
    on
        Merge on columns in addition to 'index'. Defaults to "batch" if present in
        both `obs` data frames.
    **kwargs
        Passed to :func:`pandas.merge`.
    """
    if len(kwargs):
        raise ValueError(
            "Since scirpy v0.5, this function always performs a 'left' merge "
            "on the index and does not accept any additional parameters any more."
        )
    if not adata.obs_names.is_unique:
        raise ValueError("obs names of `adata` need to be unique for merging.")
    if not adata.obs_names.is_unique:
        raise ValueError("obs_names of `adata_ir` need to be unique for merging.")
    if on is None and "batch" in adata.obs.columns and "batch" in adata_ir.obs.columns:
        on = ["batch"]

    if "has_ir" in adata.obs.columns:
        raise ValueError(
            "It seems you already have immune receptor (IR) data in `adata`. "
            "Please use `ir.pp.merge_airr_chains` instead. "
        )

    # Since pandas does not support both merge on index and columns, we
    # need to name the index, and use the index name in `on`.
    orig_index_name = adata.obs.index.name
    if "obs_names" in adata.obs.columns or "obs_names" in adata_ir.obs.columns:
        raise ValueError("This doesn't work if there's a column named 'obs_names'. ")
    adata.obs.index.name = "obs_names"
    adata_ir.obs.index.name = "obs_names"
    if on is None:
        on = list()
    on.insert(0, "obs_names")

    adata.obs = adata.obs.merge(
        adata_ir.obs, how="left", on=on, validate="one_to_one", **kwargs
    )

    adata.obs.index.name = orig_index_name

    _sanitize_anndata(adata)
