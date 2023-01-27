"""Code to support older version of the scipy data structure. """

from copy import copy
import awkward as ak
from .. import __version__
from functools import wraps
import itertools
from anndata import AnnData
from typing import Callable, List, cast
from ._datastructures import AirrCell
from ._util import _IOLogger
from ..util import _is_na2
from packaging import version


def _check_schema_pre_v0_7(adata: AnnData):
    """Raise an error if AnnData is in pre scirpy v0.7 format."""
    if (
        # I would actually only use `scirpy_version` for the check, but
        # there might be cases where it gets lost (e.g. when rebuilding AnnData).
        # If a `v_call` is present, that's a safe sign that it is the AIRR schema, too
        "has_ir" in adata.obs.columns
        and (
            "IR_VJ_1_v_call" not in adata.obs.columns
            and "IR_VDJ_1_v_call" not in adata.obs.columns
        )
        and "scirpy_version" not in adata.uns
    ):
        raise ValueError(
            "It seems your anndata object is of a very old format used by scirpy < v0.7 "
            "which is not supported anymore. You might be best off reading in your data from scratch. "
            "If you absolutely want to, you can use scirpy < v0.13 to read that format, "
            "convert it to a legacy format, and convert it again using the most recent version of scirpy. "
        )


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
        version.parse(adata.uns.get("scirpy_version", "v0.0.0"))
        >= version.parse("0.13.0")
    ):
        raise ValueError(
            "Your AnnData object seems already up-to-date with scirpy v0.13"
        )
    # Raise error if very old schema
    _check_schema_pre_v0_7(adata)

    airr_cells = _obs_schema_to_airr_cells(adata)
    tmp_adata = from_airr_cells(airr_cells)
    adata.obsm["airr"] = tmp_adata.obsm["airr"]
    adata.obsm["chain_indices"] = tmp_adata.obsm["chain_indices"]
    adata.uns["scirpy_version"] = __version__
    adata.obs = tmp_adata.obs


def _check_anndata_upgrade_schema(adata):
    """Check if `adata` uses the latest scirpy schema.

    Raises ValueError if it doesn't"""
    if not any((isinstance(x, ak.Array) for x in adata.obsm.values())):
        # First check for very old version. We don't support it at all anymore.
        _check_schema_pre_v0_7(adata)
        # otherwise suggest to use `upgrade_schema`
        raise ValueError(
            "Scirpy has updated the format of `adata` in v0.13. AIRR data is now stored as an"
            "awkward array in `adata.obsm['airr']`."
            "Please run `ir.io.upgrade_schema(adata) to update your AnnData object to "
            "the latest version. "
        )


def _check_upgrade_schema(check_args=(0,)) -> Callable:
    """Decorator that checks that anndata uses the latest schema"""

    def check_upgrade_schema_decorator(f):
        @wraps(f)
        def check_wrapper(*args, **kwargs):
            for i in check_args:
                _check_anndata_upgrade_schema(args[i])
            return f(*args, **kwargs)

        return check_wrapper

    return check_upgrade_schema_decorator


def _obs_schema_to_airr_cells(adata: AnnData) -> List[AirrCell]:
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
            # Don't add empty chains!
            if not all([_is_na2(x) for x in tmp_chain.values()]):
                tmp_ir_cell.add_chain(tmp_chain)

        try:
            tmp_ir_cell.add_serialized_chains(row["extra_chains"])
        except KeyError:
            pass
        cells.append(tmp_ir_cell)

    return cells
