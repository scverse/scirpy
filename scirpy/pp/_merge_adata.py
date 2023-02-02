import itertools
from typing import Dict

from anndata import AnnData

from ..io._convert_anndata import from_airr_cells, to_airr_cells
from ..io._datastructures import AirrCell
from ..io._legacy import _check_upgrade_schema


@_check_upgrade_schema(check_args=(0, 1))
def merge_airr(
    adata: AnnData,
    adata2: AnnData,
    *,
    airr_key="airr",
    airr_key2="airr",
    drop_duplicate_chains=True,
    **kwargs,
) -> AnnData:
    """
    Merge two AnnData objects with :term:`IR` information (e.g. BCR with TCR).

    Decomposes the IR information back into :class:`scirpy.io.AirrCell` objects
    and merges them on a chain-level. If both objects contain the same cell-id, and
    the same chains, the corresponding row in `adata.obs` will be unchanged (if `drop_duplicate_chains` is `True`).
    If both objects contain the same cell-id, but different chains, the chains
    will be merged into a single cell such that it can be annotated as
    :term:`ambiguous<Receptor type>` or :term:`multi-chain<Multichain-cell>`
    if appropriate. If a cell contains both TCR and BCR chains, they will both
    be kept and can be identified as `ambiguous` using the :func:`scirpy.tl.chain_qc`
    function.

    The function performs a "full join", i.e. all cells from both objects
    will be retained. All information except `.obsm[airr_key]` and
    `.obs` will be lost.

    TODO #356: refer to tutorial showing how to use MuData

    Parameters
    ----------
    adata
        first AnnData object containing IR information
    adata2
        second AnnData object containing IR information
    airr_key
        key under which the AIRR information is stored in `adata.obsm`
    airr_key2
        key under which the AIRR information is stored in `adata2.obsm`
    drop_duplicate_chains
        If True, if there are identical chains associated with the same cell
        only one of them is kept.
    **kwargs
        passed to :func:`~scirpy.io.from_airr_cells`

    Returns
    -------
    new AnnData object with merged AIRR data.
    """
    ir_objs1 = to_airr_cells(adata, airr_key=airr_key)
    ir_objs2 = to_airr_cells(adata2, airr_key=airr_key2)

    cell_dict: Dict[str, AirrCell] = dict()
    for cell in itertools.chain(ir_objs1, ir_objs2):
        try:
            tmp_cell = cell_dict[cell.cell_id]
        except KeyError:
            cell_dict[cell.cell_id] = cell
        else:
            for tmp_chain in cell.chains:
                tmp_cell.add_chain(tmp_chain)
            # add cell-level attributes
            tmp_cell.update(cell)

    if drop_duplicate_chains:
        # remove duplicate chains
        # https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python
        #
        # use dict.fromkeys() instead of set() to obtain a reproducible ordering
        for cell in cell_dict.values():
            cell._chains = [
                dict(t) for t in dict.fromkeys(tuple(d.items()) for d in cell.chains)
            ]

    return from_airr_cells(cell_dict.values(), **kwargs)
