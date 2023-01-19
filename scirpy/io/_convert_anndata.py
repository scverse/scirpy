"""Convert IrCells to AnnData and vice-versa"""
from anndata import AnnData
from ..util import _doc_params, _is_true2, _is_false2
from ._util import doc_working_model
from ._datastructures import AirrCell
import pandas as pd
from typing import Iterable, List, cast
from .. import __version__
from pandas.api.types import is_object_dtype
import awkward as ak
from ._util import _IOLogger
from ._legacy import _check_upgrade_schema


@_doc_params(doc_working_model=doc_working_model)
def from_airr_cells(airr_cells: Iterable[AirrCell]) -> AnnData:
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
        "airr": ak.Array((c.chains for c in airr_cells)),
        # TODO call chains
        # "chain_indices": pd.DataFrame.from_records(
        #     (c.chain_indices() for c in airr_cells)
        # ).set_index(obs.index),
    }

    return AnnData(
        X=None,
        obs=obs,
        obsm=obsm,
        uns={"scirpy_version": __version__},
    )


@_check_upgrade_schema()
def to_airr_cells(adata: AnnData, *, airr_key: str = "airr") -> List[AirrCell]:
    """
    Convert an adata object with IR information back to a list of :class:`~scirpy.io.AirrCell`
    objects.
    Inverse function of :func:`from_airr_cells`.
    Parameters
    ----------
    adata
        annotated data matrix with :term:`IR` annotations.
    airr_key
        Key in `adata.obsm` under which the AwkwardArray with AIRR information is stored

    Returns
    -------
    List of :class:`~scirpy.io.AirrCell` objects.
    """
    cells = []
    logger = _IOLogger()

    for (cell_id, row), chains in zip(adata.obs.iterrows(), adata.obsm[airr_key]):
        tmp_cell = AirrCell(cast(str, cell_id), logger=logger)
        # add cell-level metadata
        tmp_cell.update(row)
        # convert awkward records to list of dicts
        for chain in ak.to_list(chains):
            tmp_cell.add_chain(chain)
        cells.append(tmp_cell)

    return cells
