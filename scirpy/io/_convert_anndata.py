"""Convert IrCells to AnnData and vice-versa"""
import itertools
from anndata import AnnData
from ..util import _doc_params, _is_true, _is_na2
from ._util import doc_working_model, _IOLogger
from ._datastructures import AirrCell
import pandas as pd
from typing import Collection, List
import numpy as np


def _sanitize_anndata(adata: AnnData) -> None:
    """Sanitization and sanity checks on IR-anndata object.
    Should be executed by every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "

    # TODO update!
    CATEGORICAL_COLS = ("locus", "v_call", "d_call", "j_call", "c_call", "multichain")

    # Sanitize has_ir column into categorical
    # This should always be a categorical with True / False
    has_ir_mask = _is_true(adata.obs["has_ir"])
    adata.obs["has_ir"] = pd.Categorical(
        ["True" if x else "False" for x in has_ir_mask]
    )

    # Turn other columns into categorical
    for col in adata.obs.columns:
        if col.endswith(CATEGORICAL_COLS):
            adata.obs[col] = pd.Categorical(adata.obs[col])

    adata._sanitize()


@_doc_params(doc_working_model=doc_working_model)
def from_ir_objs(ir_objs: Collection[AirrCell]) -> AnnData:
    """\
    Convert a collection of :class:`AirrCell` objects to an :class:`~anndata.AnnData`.

    This is useful for converting arbitrary data formats into
    the scirpy :ref:`data-structure`.

    {doc_working_model}

    Parameters
    ----------
    ir_objs


    Returns
    -------
    :class:`~anndata.AnnData` object with :term:`IR` information in `obs`.

    """
    ir_df = pd.DataFrame.from_records(
        (x.to_scirpy_record() for x in ir_objs)
    ).set_index("cell_id")
    adata = AnnData(obs=ir_df, X=np.empty([ir_df.shape[0], 0]))
    _sanitize_anndata(adata)
    return adata


def to_ir_objs(adata: AnnData) -> List[AirrCell]:
    """
    Convert an adata object with IR information back to a list of IrCells.

    Inverse function of :func:`from_ir_objs`.

    Parameters
    ----------
    adata
        annotated data matrix with :term:`IR` annotations.

    Returns
    -------
    List of IrCells
    """
    cells = []
    logger = _IOLogger()

    obs = adata.obs.copy()
    ir_cols = obs.columns[obs.columns.str.startswith("IR_")]
    for cell_id, row in obs.iterrows():
        tmp_ir_cell = AirrCell(cell_id, logger=logger, multi_chain=row["multi_chain"])
        chains = {
            (junction_type, chain_id): dict()
            for junction_type, chain_id in itertools.product(["VJ", "VDJ"], ["1", "2"])
        }
        for tmp_col in ir_cols:
            _, junction_type, chain_id, key = tmp_col.split("_", maxsplit=3)
            chains[(junction_type, chain_id)][key] = row[tmp_col]

        for tmp_chain in chains.values():
            # Don't add empty chains!
            if not all([_is_na2(x) for x in tmp_chain.values()]):
                tmp_ir_cell.add_chain(tmp_chain)
        tmp_ir_cell.add_serialized_chains(row["extra_chains"])
        cells.append(tmp_ir_cell)

    return cells


# TODO getter function to retrieve the 'extra' chains, productive or not.
# TODO getter function to retrieve adata.obs without IR columns, or IR columns only.
