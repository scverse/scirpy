from typing import Union, List, Dict
from anndata import AnnData
from ..io._convert_anndata import (
    _sanitize_anndata,
    to_ir_objs,
    from_ir_objs,
)
from ..io._datastructures import AirrCell
from scanpy import logging
import itertools
import pandas as pd


def _merge_ir_obs(adata: AnnData, adata2: AnnData) -> pd.DataFrame:
    """
    Merge two AnnData objects with :term:`IR` information (e.g. BCR with TCR).

    Decomposes the IR information back into :class:`scirpy.io.AirrCell` objects
    and merges them on a chain-level. If both objects contain the same cell-id, and
    the same chains, the corresponding row in `adata.obs` will be unchanged.
    If both objects contain the same cell-id, but different chains, the chains
    will be merged into a single cell such that it can be annotated as
    :term:`ambiguous<Receptor type>` or :term:`multi-chain<Multichain-cell>`
    if appropriate.

    Discards all non IR information from both adatas, including non-IR columns in obs.

    Parameters
    ----------
    adata
        first AnnData object containing IR information
    adata2
        second AnnData object containint IR information

    Returns
    -------
    Merged IR obs data frame.
    """
    ir_objs1 = to_ir_objs(adata)
    ir_objs2 = to_ir_objs(adata2)
    cell_dict: Dict[str, AirrCell] = dict()
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

    return from_ir_objs(cell_dict.values()).obs


def merge_with_ir(
    adata: AnnData, adata_ir: AnnData, on: Union[List[str], None] = None, **kwargs
) -> None:
    """Merge adaptive immune receptor (:term:`IR`) data with transcriptomics data into a
    single :class:`~anndata.AnnData` object.

    :ref:`Reading in IR data<importing-data>` results in an :class:`~anndata.AnnData`
    object with IR information stored in `obs`. Use this function to merge
    it with another :class:`~anndata.AnnData` which contains transcriptomics data. You
    can also use it to add additional IR data on top of an :class:`~anndata.AnnData`
    object that already contains IR information (e.g. :term:`BCR` on top of
    :term:`TCR` data. )

    Merging keeps all objects (e.g. `neighbors`, `umap`) from `adata` and integrates
    `obs` from `adata_ir` into `adata`. Everything other than `.obs` from `adata_ir`
    will be discarded.

    If `adata` does not contain IR information, this function simply
    uses :func:`pandas.merge` to join the two `.obs` data frames.  If `adata` already
    contains IR information, it merges the IR information on a chain-level. This is
    useful, e.g. when adding :term:`BCR` data on top of a dataset that already contains
    :term:`TCR` data. If a cell contains both TCR and BCR chains, they will both
    be kept and can be identified as `ambiguous` using the :func:`scirpy.tl.chain_qc`
    function.

    Merging is performed in two steps: (1) merge all non-IR columns from
    `adata.obs` with all non-IR columns from `adata_ir.obs` (2) merge the result
    of step (1) with all IR columns.

    Modifies `adata` inplace.

    Parameters
    ----------
    adata
        AnnData with the transcriptomics data. Will be modified inplace.
    adata_ir
        AnnData with the adaptive immune receptor (IR) data
    on
        Merge on columns in addition to 'index'. Only applies to the first merge
        step (non IR-columns). Defaults to "batch" if present in both `obs`
        data frames.
    **kwargs
        Passed to the *first* merge step. See :func:`pandas.merge`.
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
        logging.warning(
            "It seems you already have immune receptor (IR) data in `adata`. "
            "Merging IR objects by chain. "
        )
        ir_obs = _merge_ir_obs(adata, adata_ir)
        non_ir_obs_left = adata.obs.drop(IR_OBS_COLS, axis="columns", errors="ignore")
        non_ir_obs_right = adata_ir.obs.drop(
            IR_OBS_COLS, axis="columns", errors="ignore"
        )
    else:
        ir_obs = adata_ir.obs[IR_OBS_COLS]
        non_ir_obs_left = adata.obs
        non_ir_obs_right = adata_ir.obs.drop(
            IR_OBS_COLS, axis="columns", errors="ignore"
        )

    # Since pandas does not support both merge on index and columns, we
    # need to name the index, and use the index name in `on`.
    orig_index_name = adata.obs.index.name
    if (
        "obs_names" in non_ir_obs_left.columns
        or "obs_names" in non_ir_obs_right.columns
    ):
        raise ValueError("This doesn't work if there's a column name 'obs_names'. ")
    non_ir_obs_left.index.name = "obs_names"
    non_ir_obs_right.index.name = "obs_names"
    ir_obs.index.name = "obs_names"
    if on is None:
        on = list()
    on.insert(0, "obs_names")

    adata.obs = non_ir_obs_left.merge(
        non_ir_obs_right,
        how="left",
        on=on,
        validate="one_to_one",
        **kwargs,
    ).merge(
        ir_obs,
        how="left",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    )

    adata.obs.index.name = orig_index_name

    _sanitize_anndata(adata)
