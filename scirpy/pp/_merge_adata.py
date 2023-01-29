import itertools
from typing import Dict

from anndata import AnnData

from ..io._convert_anndata import from_airr_cells, to_airr_cells
from ..io._datastructures import AirrCell
from ..io._legacy import _check_upgrade_schema


# TODO #356: can this be achieved with a join at the AnnData level (i.e. anndata itself merging the awkward array?)
# No, I don't think so. So we still need this function, but it can be simplified (just need to rebuild the
# awkward array, and call `index_chains` again. )
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
    # Compute `include_fields` to avoid including fields that are part of
    # empty airr cells as part of the rearrangement standard, but not included
    # in the original anndata object.
    include_fields = set(
        x.split("_", maxsplit=3)[-1] if x.startswith("IR_") else x
        for x in itertools.chain(adata.obs.columns, adata2.obs.columns)
    )
    cell_dict: Dict[str, AirrCell] = dict()
    for cell in itertools.chain(ir_objs1, ir_objs2):
        try:
            tmp_cell = cell_dict[cell.cell_id]
        except KeyError:
            cell_dict[cell.cell_id] = cell
        else:
            # this is a legacy operation. With adatas generated with scirpy
            # >= 0.7 this isn't necessary anymore, as all chains are preserved.
            tmp_cell["multi_chain"] = bool(tmp_cell["multi_chain"]) | bool(
                cell["multi_chain"]
            )
            for tmp_chain in cell.chains:
                tmp_cell.add_chain(tmp_chain)
            # add cell-level attributes
            tmp_cell.update(cell)

    # TODO #356: make this a parameter (e.g. drop_dulicates)?
    # remove duplicate chains
    # https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python
    for cell in cell_dict.values():
        cell._chains = [dict(t) for t in set(tuple(d.items()) for d in cell.chains)]

    # only keep entries that are in `adata` and ensure consistent ordering
    adata.obs = from_airr_cells(
        cell_dict.values(), include_fields=include_fields
    ).obs.reindex(adata.obs_names)
