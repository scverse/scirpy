"""Convert IrCells to AnnData and vice-versa"""
import itertools
from anndata import AnnData
from ..util import _doc_params, _is_true, _is_na2, _is_true2, _is_false2
from ._util import doc_working_model, _IOLogger, _check_upgrade_schema
from ._datastructures import AirrCell
import pandas as pd
from typing import Collection, Iterable, List, Optional
from .. import __version__
import numpy as np
from pandas.api.types import is_object_dtype


def _sanitize_anndata(adata: AnnData) -> None:
    """Sanitization and sanity checks on IR-anndata object.
    Should be executed by every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "

    # Pending updates to anndata to properly handle boolean columns.
    # For now, let's turn them into a categorical with "True/False"
    BOOLEAN_COLS = ("has_ir", "is_cell", "multi_chain", "high_confidence", "productive")

    # explicitly convert those to categoricals. All IR_ columns that are strings
    # will be converted to categoricals, too
    CATEGORICAL_COLS = ("extra_chains",)

    # Sanitize has_ir column into categorical
    # This should always be a categorical with True / False
    for col in adata.obs.columns:
        if col.endswith(BOOLEAN_COLS):
            adata.obs[col] = pd.Categorical(
                [
                    "True" if _is_true2(x) else "False" if _is_false2(x) else "None"
                    for x in adata.obs[col]
                ],
                categories=["True", "False", "None"],
            )
        elif col.endswith(CATEGORICAL_COLS) or (
            col.startswith("IR_") and is_object_dtype(adata.obs[col])
        ):
            # Turn all IR_VJ columns that are of type string or object to categoricals
            # otherwise saving anndata doesn't work.
            adata.obs[col] = pd.Categorical(adata.obs[col])

    adata.strings_to_categoricals()


@_doc_params(doc_working_model=doc_working_model)
def from_airr_cells(
    airr_cells: Iterable[AirrCell], include_fields: Optional[Collection[str]] = None
) -> AnnData:
    """\
    Convert a collection of :class:`~scirpy.io.AirrCell` objects to :class:`~anndata.AnnData`.

    This is useful for converting arbitrary data formats into
    the scirpy :ref:`data-structure`.

    {doc_working_model}

    Parameters
    ----------
    airr_cells
        A list of :class:`~scirpy.io.AirrCell` objects
    include_fields
        A list of field names that are to be transferred to `adata`. If `None`
        (the default), transfer all fields. Use this option to avoid cluttering
        of `adata.obs` by irrelevant columns.

    Returns
    -------
    :class:`~anndata.AnnData` object with :term:`IR` information in `obs`.

    """
    import awkward._v2 as ak

    obs = pd.DataFrame.from_records(iter(airr_cells)).set_index("cell_id")

    # For now, require that all chains have the same keys
    airr_keys = None
    for tmp_cell in airr_cells:
        for tmp_chain in tmp_cell.chains:
            assert airr_keys == tmp_chain.keys() or airr_keys is None
            airr_keys = tmp_chain.keys()

    # Build list of lists for awkward array
    # TODO this is inefficient, but maybe efficient enough.
    # Dimensions (obs, var, n_chains)
    cell_list = []
    for tmp_cell in airr_cells:
        tmp_var_list = [[] for _ in airr_keys]
        for tmp_chain in tmp_cell.chains:
            for tmp_chain_list, value in zip(tmp_var_list, tmp_chain.values()):
                tmp_chain_list.append(value)
        cell_list.append(tmp_var_list)

    X = ak.Array(cell_list)
    X = ak.to_regular(X, 1)

    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=airr_keys))


@_check_upgrade_schema()
def to_airr_cells(adata: AnnData) -> List[AirrCell]:
    """
    Convert an adata object with IR information back to a list of :class:`~scirpy.io.AirrCell`
    objects.

    Inverse function of :func:`from_airr_cells`.

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
        tmp_ir_cell = AirrCell(cell_id, logger=logger)

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
