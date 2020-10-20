"""Convert IrCells to AnnData and vice-versa"""
import itertools
from anndata import AnnData
from ..util import _is_na, _is_true, _doc_params
from ._common_doc import doc_working_model
from ._datastructures import IrCell, IrChain
import pandas as pd
from typing import Collection, List
import numpy as np

#: All fields available for each of the four IR chains.
IR_OBS_KEYS = [
    "locus",
    "cdr3",
    "cdr3_nt",
    "expr",
    "expr_raw",
    "v_gene",
    "d_gene",
    "j_gene",
    "c_gene",
    "junction_ins",
]

#: All cols addedby scirpy when reading in data
IR_OBS_COLS = [
    f"IR_{chain_type}_{chain_id}_{key}"
    for key, chain_type, chain_id in itertools.product(
        IR_OBS_KEYS, ["VJ", "VDJ"], ["1", "2"]
    )
] + ["has_ir", "multi_chain"]


def _sanitize_anndata(adata: AnnData) -> None:
    """Sanitization and sanity checks on IR-anndata object.
    Should be executed by every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "

    CATEGORICAL_COLS = ("locus", "v_gene", "d_gene", "j_gene", "c_gene", "multichain")

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
def from_ir_objs(ir_objs: Collection[IrCell]) -> AnnData:
    """\
    Convert a collection of :class:`IrCell` objects to an :class:`~anndata.AnnData`.

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
        (_process_ir_cell(x) for x in ir_objs), index="cell_id"
    )
    adata = AnnData(obs=ir_df, X=np.empty([ir_df.shape[0], 0]))
    _sanitize_anndata(adata)
    return adata


@_doc_params(doc_working_model=doc_working_model)
def _process_ir_cell(ir_obj: IrCell) -> dict:
    """\
    Process a IrCell object into a dictionary according
    to our working model of adaptive immune receptors.

    {doc_working_model}

    Parameters
    ----------
    ir_obj
        IrCell object to process

    Returns
    -------
    Dictionary representing one row of the final `AnnData.obs`
    data frame.
    """
    res_dict = dict()
    res_dict["cell_id"] = ir_obj.cell_id
    res_dict["multi_chain"] = ir_obj.multi_chain
    chain_dict = dict()
    for junction_type in ["VJ", "VDJ"]:
        # sorting subordinately by raw and cdr3 ensures consistency
        # between load from json and load from csv.
        tmp_chains = sorted(
            [
                x
                for x in ir_obj.chains
                if x.junction_type == junction_type and x.is_productive
            ],
            key=lambda x: (x.expr, x.expr_raw, x.cdr3),
            reverse=True,
        )
        # multichain if at least one of the receptor arms is multichain
        res_dict["multi_chain"] = res_dict["multi_chain"] | (len(tmp_chains) > 2)
        # slice to max two chains
        tmp_chains = tmp_chains[:2]
        # add None if less than two chains
        tmp_chains += [None] * (2 - len(tmp_chains))
        chain_dict[junction_type] = tmp_chains

    for key in IR_OBS_KEYS:
        for junction_type, tmp_chains in chain_dict.items():
            for i, chain in enumerate(tmp_chains):
                res_dict["IR_{}_{}_{}".format(junction_type, i + 1, key)] = (
                    getattr(chain, key) if chain is not None else None
                )

    # in some weird reasons, it can happen that a cell has been called from
    # TCR-seq but no TCR seqs have been found. `has_ir` should be equal
    # to "at least one productive chain"
    res_dict["has_ir"] = not (
        _is_na(res_dict["IR_VJ_1_cdr3"]) and _is_na(res_dict["IR_VDJ_1_cdr3"])
    )

    # if there are not chains at all, we want multi-chain to be nan
    # This is to be consistent with when turning an anndata object into ir_objs
    # and converting it back to anndata.
    if not len(ir_obj.chains):
        res_dict["multi_chain"] = np.nan

    if _is_na(res_dict["IR_VJ_1_cdr3"]):
        assert _is_na(
            res_dict["IR_VJ_2_cdr3"]
        ), "There can't be a secondary chain if there is no primary one: {}".format(
            res_dict
        )
    if _is_na(res_dict["IR_VDJ_1_cdr3"]):
        assert _is_na(
            res_dict["IR_VDJ_2_cdr3"]
        ), "There can't be a secondary chain if there is no primary one: {}".format(
            res_dict
        )

    return res_dict


def to_ir_objs(adata: AnnData) -> List[IrCell]:
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
    try:
        for cell_id, row in adata.obs.iterrows():
            tmp_ir_cell = IrCell(cell_id, multi_chain=row["multi_chain"])
            for chain_type, chain_id in itertools.product(["VJ", "VDJ"], ["1", "2"]):
                chain_dict = {
                    key: row[f"IR_{chain_type}_{chain_id}_{key}"] for key in IR_OBS_KEYS
                }
                # per definition, we currently only have productive chains in adata.
                chain_dict["is_productive"] = True
                if not _is_na(chain_dict["locus"]):
                    # if no locus/chain specified, the correponding chain
                    # does not exists and we don't want to add this.
                    # This is also the way we can represent cells without IR.
                    tmp_ir_cell.add_chain(IrChain(**chain_dict))

            cells.append(tmp_ir_cell)
    except KeyError as e:
        raise ValueError(
            f"Key {str(e)} not found in adata. Does it contain immune receptor data?"
        )

    return cells
