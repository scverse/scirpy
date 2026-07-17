"""Code to support older version of the scipy data structure."""

import itertools
from importlib.metadata import version
from typing import cast

import awkward as ak
import packaging.version
from anndata import AnnData

from scirpy.util import DataHandler, _is_na2

from ._datastructures import AirrCell
from ._util import _IOLogger


def upgrade_schema(adata: AnnData) -> AnnData:
    """Update older versions of a scirpy anndata object to the latest schema.

    Returns a new AnnData object.

    Parameters
    ----------
    adata
        annotated data matrix
    """
    from ._convert_anndata import from_airr_cells

    # Raise error if already up to date
    if ("airr" in adata.obsm and isinstance(adata.obsm["airr"], ak.Array)) or (
        packaging.version.parse(adata.uns.get("scirpy_version", "v0.0.0")) >= packaging.version.parse("0.13.0")
    ):
        raise ValueError("Your AnnData object seems already up-to-date with scirpy v0.13")
    # Raise error if very old schema
    DataHandler.check_schema_pre_v0_7(adata)

    airr_cells = _obs_schema_to_airr_cells(adata)
    tmp_adata = from_airr_cells(airr_cells)
    adata.obsm["airr"] = tmp_adata.obsm["airr"]
    adata.uns["scirpy_version"] = version("scirpy")
    adata.obs = tmp_adata.obs


def _obs_schema_to_airr_cells(adata: AnnData) -> list[AirrCell]:
    """
    Convert a legacy adata object with IR information in adata.obs back to a list of
    :class:`~scirpy.io.AirrCell` objects.

    Parameters
    ----------
    adata
        annotated data matrix with :term:`IR` annotations.

    Returns
    -------
    List of :class:`~scirpy.io.AirrCell` objects.
    """
    cells = []
    logger = _IOLogger()

    obs = adata.obs.copy()
    ir_cols = obs.columns[obs.columns.str.startswith("IR_")]
    other_cols = set(adata.obs.columns) - set(ir_cols)
    for cell_id, row in obs.iterrows():
        tmp_ir_cell = AirrCell(cast(str, cell_id), logger=logger)

        # add cell-level attributes
        for col in other_cols:
            # skip these columns: we want to use the index as cell id,
            # extra_chains and has_ir get added separately
            if col in ("cell_id", "extra_chains", "has_ir"):
                continue
            tmp_ir_cell[col] = row[col]

        # add chain level attributes
        chains = {
            (junction_type, chain_id): AirrCell.empty_chain_dict()
            for junction_type, chain_id in itertools.product(["VJ", "VDJ"], ["1", "2"])
        }
        for tmp_col in ir_cols:
            _, junction_type, chain_id, key = tmp_col.split("_", maxsplit=3)
            chains[(junction_type, chain_id)][key] = row[tmp_col]

        for tmp_chain in chains.values():
            # in the old schema, these chains are productive by definition
            tmp_chain["productive"] = True
            # Don't add empty chains!
            if not all(_is_na2(x) for x in tmp_chain.values()):
                tmp_ir_cell.add_chain(tmp_chain)

        try:
            tmp_ir_cell.add_serialized_chains(row["extra_chains"])
        except KeyError:
            pass
        cells.append(tmp_ir_cell)

    return cells
