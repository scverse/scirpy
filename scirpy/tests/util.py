from ..util import _is_na
import pandas as pd
import numpy as np
from typing import List, Union, Optional, cast
from anndata import AnnData
import awkward as ak
from scirpy import __version__
from scirpy.util import _is_na2


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


def _squarify(matrix: Union[List[List], np.ndarray]):
    """Squarify a upper triangular matrix"""
    matrix = np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "only works for square matrices"
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    assert np.allclose(matrix.T, matrix)
    return matrix


def _make_adata(obs: pd.DataFrame) -> AnnData:
    """Generate an AnnData object from a obs dataframe formatted according to the old obs-based scheam.

    This is used to convert test cases from unittests. Writing them from scratch
    would be a lot of effort. Also the awkward array format is not very ergonomic to create
    manually, so we use this instead.

    Compared to the function that converts legacy anndata objects via an intermediate step of
    creating AirrCells, this function works more directly, and can cope with minimal data that is
     * incorrect on purpose (for a test case)
     * is missing columns that are mandatory, but irrelevant for a test case
     * ensures a value ends up in the chain (VJ_1, VDJ_2, etc) the author of the test explicitly intended, instead
       of relying on the ranking of cells implemented in the AirrCell class.
    """
    # AnnData requires indices to be strings
    obs.index = obs.index.astype(str)
    # ensure that the columns are ordered, i.e. for each variable, VJ_1, VJ_2, VDJ1, ... come in the same order.
    obs.sort_index(axis=1, inplace=True)
    cols = [x for x in obs.columns if x.startswith("IR_")]
    unique_variables = set(c.split("_", 3)[3] for c in cols)

    # determine the number of chains per cell. This is used to determine the size of the second
    # dimension of the awkward array. May be different for each cell, but has the same length for all variables
    # of a certain cell.
    has_chain = []
    for _, row in obs.iterrows():
        has_chain_dict = {k: False for k in ["VJ_1", "VJ_2", "VDJ_1", "VDJ_2"]}
        for c in cols:
            # if any of the columns has that chain, we set the value to True
            _, receptor_arm, chain, var = c.split("_", 3)
            if not _is_na2(row[c]):
                has_chain_dict[f"{receptor_arm}_{chain}"] = True
        has_chain.append(has_chain_dict)

    # Now we build a list of chain dictionaries and chain indices row by row
    cell_list = []
    chain_idx_list = []
    for has_chain_dict, (_, row) in zip(has_chain, obs.iterrows()):
        tmp_chains = []
        tmp_chain_idx: dict[str, Optional[int]] = {k: None for k in has_chain_dict}
        for chain, row_has_chain in has_chain_dict.items():
            if row_has_chain:
                tmp_chains.append(
                    {v: row.get("IR_{chain}_{v}", None) for v in unique_variables}
                )
                tmp_chain_idx[chain] = len(tmp_chains) - 1

        cell_list.append(tmp_chains)
        chain_idx_list.append(tmp_chain_idx)

    airr_data = ak.Array(cell_list)
    chain_indices = pd.DataFrame.from_records(chain_idx_list, index=obs.index)

    adata = AnnData(
        X=None,
        obs=obs.loc[:, ~obs.columns.isin(cols)].copy(),  # type:ignore
        obsm={"chain_indices": chain_indices, "airr": airr_data},  # type:ignore
        uns={"scirpy_version": __version__},
    )
    return adata
