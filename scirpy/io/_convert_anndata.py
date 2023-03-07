"""Convert IrCells to AnnData and vice-versa"""
from typing import Iterable, List, cast

import awkward as ak
import pandas as pd
from anndata import AnnData

from .. import __version__
from ..util import _doc_params, _ParamsCheck
from ._datastructures import AirrCell
from ._util import _IOLogger, doc_working_model


@_doc_params(doc_working_model=doc_working_model)
def from_airr_cells(airr_cells: Iterable[AirrCell], key_added: str = "airr") -> AnnData:
    """\
    Convert a collection of :class:`~scirpy.io.AirrCell` objects to :class:`~anndata.AnnData`.

    This is useful for converting arbitrary data formats into
    the scirpy :ref:`data-structure`.

    {doc_working_model}

    Parameters
    ----------
    airr_cells
        A list of :class:`~scirpy.io.AirrCell` objects
    index_chains
        If `True`, automatically run :func:`~scirpy.pp.index_chains` with 
        default parameters. 

    Returns
    -------
    :class:`~anndata.AnnData` object with :term:`IR` information in `obs`.

    """
    import awkward as ak

    # data frame from cell-level attributes
    obs = pd.DataFrame.from_records(iter(airr_cells)).set_index("cell_id")
    # AnnData requires indices to be strings
    # A range index would automatically be converted by AnnData, but then the `obsm` object doesn't
    # match the index anymore.
    obs.index = obs.index.astype(str)

    obsm = {
        key_added: ak.Array((c.chains for c in airr_cells)),
    }

    adata = AnnData(
        X=None,
        obs=obs,
        obsm=obsm,
        uns={"scirpy_version": __version__},
    )

    return adata


@_ParamsCheck.inject_param_docs()
def to_airr_cells(
    adata: _ParamsCheck.TYPE, *, airr_mod: str = "airr", airr_key: str = "airr"
) -> List[AirrCell]:
    """\
    Convert an adata object with IR information back to a list of :class:`~scirpy.io.AirrCell`
    objects.
    Inverse function of :func:`from_airr_cells`.

    Parameters
    ----------
    {adata}
    {airr_mod}
    {airr_key}

    Returns
    -------
    List of :class:`~scirpy.io.AirrCell` objects.
    """
    cells = []
    logger = _IOLogger()

    params = _ParamsCheck(adata, airr_mod, airr_key)

    for (cell_id, row), chains in zip(params.adata.obs.iterrows(), params.airr):
        tmp_cell = AirrCell(cast(str, cell_id), logger=logger)
        # add cell-level metadata
        tmp_cell.update(row)
        # convert awkward records to list of dicts
        for chain in ak.to_list(chains):
            tmp_cell.add_chain(chain)
        cells.append(tmp_cell)

    return cells
