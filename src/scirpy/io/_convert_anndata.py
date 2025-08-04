"""Convert IrCells to AnnData and vice-versa"""

from collections.abc import Iterable
from importlib.metadata import version
from typing import cast

import awkward as ak
import pandas as pd
from anndata import AnnData

from scirpy.util import DataHandler, _doc_params, tqdm

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
        key_added: ak.Array(c.chains for c in airr_cells),
    }

    adata = AnnData(
        X=None,
        obs=obs,
        obsm=obsm,
        uns={"scirpy_version": version("scirpy")},
    )

    return adata


@DataHandler.inject_param_docs()
def to_airr_cells(adata: DataHandler.TYPE, *, airr_mod: str = "airr", airr_key: str = "airr") -> list[AirrCell]:
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

    params = DataHandler(adata, airr_mod, airr_key)

    # in chunks of 5000-10000 this is fastest. Not sure why there is additional
    # overhead when running `to_list` on the full array. It's anyway friendlier to memory this way.
    # see also: index_chains
    CHUNKSIZE = 5000
    for i in tqdm(range(0, len(params.airr), CHUNKSIZE)):
        tmp_airr = ak.to_list(params.airr[i : i + CHUNKSIZE])
        tmp_obs = params.adata.obs.iloc[i : i + CHUNKSIZE].to_dict(orient="index")

        for (cell_id, row), chains in zip(tmp_obs.items(), tmp_airr, strict=False):
            tmp_cell = AirrCell(cast(str, cell_id), logger=logger)
            # add cell-level metadata
            tmp_cell.update(row)
            # convert awkward records to list of dicts
            for chain in chains:
                tmp_cell.add_chain(chain)
            cells.append(tmp_cell)

    return cells
