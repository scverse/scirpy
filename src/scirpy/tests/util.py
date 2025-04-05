from importlib.metadata import version
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from scirpy.io import AirrCell
from scirpy.util import _is_na, _is_na2


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan, "True" to True, and "False" to False.

    Modifies df inplace.
    """
    df.sort_index(axis="columns", inplace=True)
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)
        df.loc[_is_na(df[col]), col] = None
        df.loc[df[col] == "True", col] = True
        df.loc[df[col] == "False", col] = False


def _squarify(matrix: list[list] | np.ndarray):
    """Squarify a upper triangular matrix"""
    matrix = np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "only works for square matrices"
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    assert np.allclose(matrix.T, matrix)
    return matrix


def _make_adata(obs: pd.DataFrame, mudata: bool = False) -> AnnData | MuData:
    """Generate an AnnData object from a obs dataframe formatted according to the old obs-based scheam.

    This is used to convert test cases from unittests. Writing them from scratch
    would be a lot of effort. Also the awkward array format is not very ergonomic to create
    manually, so we use this instead.

    It accepts the following columns
        * IR_{VJ,VDJ}_{<airr_var>}_{1,2}, to set an arbitrary airr variable for any of the four chains
        * _multi_chain (optional), to manually set the multi chain status. Defaults to False.

    Compared to the function that converts legacy anndata objects via an intermediate step of
    creating AirrCells, this function works more directly, and can cope with minimal data that is
     * incorrect on purpose (for a test case)
     * is missing columns that are mandatory, but irrelevant for a test case
     * ensures a value ends up in the chain (VJ_1, VDJ_2, etc) the author of the test explicitly intended, instead
       of relying on the ranking of cells implemented in the AirrCell class.
    """
    # AnnData requires indices to be strings
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    # ensure that the columns are ordered, i.e. for each variable, VJ_1, VJ_2, VDJ1, ... come in the same order.
    obs.sort_index(axis=1, inplace=True)
    cols = [x for x in obs.columns if x.startswith("IR_")]
    unique_variables = {c.split("_", 3)[3] for c in cols}

    def _sanitize_value(v):
        """Nans are represented as the string `"nan"` in most test cases for historical reasons"""
        return None if _is_na(v) else v

    # determine the number of chains per cell. This is used to determine the size of the second
    # dimension of the awkward array. May be different for each cell, but has the same length for all variables
    # of a certain cell.
    has_chain = []
    for _, row in obs.iterrows():
        has_chain_dict = dict.fromkeys(["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"], False)
        for c in cols:
            # if any of the columns has that chain, we set the value to True
            _, receptor_arm, chain, var = c.split("_", 3)
            if not _is_na2(row[c]):
                has_chain_dict[f"{receptor_arm}_{chain}"] = True
        has_chain.append(has_chain_dict)

    # Now we build a list of chain dictionaries and chain indices row by row
    cell_list = []
    chain_idx_list = []
    for has_chain_dict, (_, row) in zip(has_chain, obs.iterrows(), strict=False):
        tmp_chains = []
        tmp_chain_idx: dict[str, Any] = {k: [None, None] for k in ["VJ", "VDJ"]}
        for chain, row_has_chain in has_chain_dict.items():
            receptor_arm, chain_i = chain.split("_")
            chain_i = int(chain_i) - 1
            if row_has_chain:
                tmp_chains.append({v: _sanitize_value(row.get(f"IR_{chain}_{v}", None)) for v in unique_variables})
                tmp_chain_idx[receptor_arm][chain_i] = len(tmp_chains) - 1
        tmp_chain_idx["multichain"] = row.get("_multi_chain", False)

        cell_list.append(tmp_chains)
        chain_idx_list.append(tmp_chain_idx)

    airr_data = ak.Array(_make_airr_chains_valid(cell_list))
    chain_indices = ak.Array(chain_idx_list)
    for k in ["VJ", "VDJ"]:
        # ensure chain indices are alwasy int (even when all values are None)
        chain_indices[k] = ak.values_astype(chain_indices[k], int, including_unknown=True)

    adata = AnnData(
        X=None,
        obs=obs.loc[:, ~obs.columns.isin(cols)].copy(),  # type:ignore
        obsm={"chain_indices": chain_indices, "airr": airr_data},  # type:ignore
        uns={"scirpy_version": version("scirpy")},
    )
    adata.strings_to_categoricals()
    if mudata:
        return MuData({"airr": adata})
    else:
        return adata


def _make_airr_chains_valid(tmp_airr: list[list[dict]]) -> list[list[dict]]:
    """Take a list of lists of Airr chain dictionaries, and add empty fields that are required
    as per the rearrangement standard
    """
    new_airr = []
    for row in tmp_airr:
        new_row = []
        for chain in row:
            new_chain = AirrCell.empty_chain_dict()
            new_chain.update(chain)
            new_row.append(new_chain)
        new_airr.append(new_row)
    return new_airr
