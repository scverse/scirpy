import itertools

from anndata import AnnData

from scirpy.io._convert_anndata import from_airr_cells, to_airr_cells
from scirpy.io._datastructures import AirrCell
from scirpy.util import DataHandler


@DataHandler.inject_param_docs()
def merge_airr(
    adata: DataHandler.TYPE,
    adata2: DataHandler.TYPE,
    *,
    airr_mod="airr",
    airr_mod2="airr",
    airr_key="airr",
    airr_key2="airr",
    drop_duplicate_chains=True,
    **kwargs,
) -> AnnData:
    """\
    Merge two AnnData objects with :term:`IR` information (e.g. BCR with TCR).

    Decomposes the IR information back into :class:`scirpy.io.AirrCell` objects
    and merges them on a chain-level. If both objects contain the same cell-id, and
    the same chains, the corresponding row in `adata.obsm["airr"]` will be unchanged (if `drop_duplicate_chains` is `True`).
    If both objects contain the same cell-id, but different chains, the chains
    will be merged into a single cell such that it can be annotated as
    :term:`ambiguous<Receptor type>` or :term:`multi-chain<Multichain-cell>`
    if appropriate. If a cell contains both TCR and BCR chains, they will both
    be kept and can be identified as `ambiguous` using the :func:`scirpy.tl.chain_qc`
    function.

    The function performs a "outer join", i.e. all cells from both objects
    will be retained. All information except `.obsm[airr_key]` and
    `.obs` will be lost.

    .. note::

        There is no need to use this function for the following use-cases:

         * If you want to merge AIRR data with transcriptomics data, use :class:`~mudata.MuData` instead,
           as shown in :ref:`multimodal-data`.
         * If you want to concatenante mutliple :class:`~anndata.AnnData` objects, use :func:`anndata.concat` instead,
           as shown in :ref:`combining-samples`.

    Parameters
    ----------
    adata
        first AnnData object containing IR information
    adata2
        second AnnData object containing IR information
    {airr_mod}
    airr_mod2
        Like `airr_mod`, but for adata2
    {airr_key}
    airr_key2
        Like `airr_key`, but for adata2
    drop_duplicate_chains
        If True, if there are identical chains associated with the same cell
        only one of them is kept.
    **kwargs
        passed to :func:`~scirpy.io.from_airr_cells`

    Returns
    -------
    new AnnData object with merged AIRR data.
    """
    ir_objs1 = to_airr_cells(adata, airr_mod=airr_mod, airr_key=airr_key)
    ir_objs2 = to_airr_cells(adata2, airr_mod=airr_mod2, airr_key=airr_key2)

    cell_dict: dict[str, AirrCell] = {}
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
            cell._chains = [dict(t) for t in dict.fromkeys(tuple(d.items()) for d in cell.chains)]

    return from_airr_cells(cell_dict.values(), **kwargs)
