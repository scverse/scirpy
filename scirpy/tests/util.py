from ..util import _is_na
import pandas as pd
import numpy as np
from typing import List, Union
from anndata import AnnData
import awkward._v2 as ak
from scirpy import __version__
from scirpy.util import _is_na2


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan.

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
    # ensure that the columns are ordered, i.e. for each variable, VJ_1, VJ_2, VDJ1, ... come in the same order.
    obs.sort_index(axis=1, inplace=True)
    cols = [x for x in obs.columns if x.startswith("IR_")]
    unique_variables = set(c.split("_", 3)[3] for c in cols)
    # map unique variables to a numeric index
    var_to_index = {v: i for i, v in enumerate(unique_variables)}

    # determine the number of chains per cell. This is used to determine the size of the third
    # dimension of the awkward array. May be different for each cell, but has the same length for all variablese
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

    cell_list = []
    chain_idx_list = []
    for i, (_, row) in enumerate(obs.iterrows()):
        tmp_cell = [[] for _ in unique_variables]
        chain_idx_row = {}
        for c in cols:
            _, receptor_arm, chain, var = c.split("_", 3)
            chain_key = f"{receptor_arm}_{chain}"
            chain_idx = len(tmp_cell[var_to_index[var]])

            if _is_na2(row[c]) and not has_chain[i][chain_key]:
                # keep nan values only if any other variable has that many chains.
                # otherwise the 'ragged' array will just be one entry shorter.
                continue
            else:
                tmp_cell[var_to_index[var]].append(None if _is_na2(row[c]) else row[c])
                try:
                    try:
                        assert chain_idx_row[chain_key] == chain_idx
                    except AssertionError:
                        print("TODO")
                except KeyError:
                    chain_idx_row[chain_key] = chain_idx
        cell_list.append(tmp_cell)
        chain_idx_list.append(chain_idx_row)

    X = ak.Array(cell_list)
    X = ak.to_regular(X, 1)

    chain_indices = pd.DataFrame.from_records(chain_idx_list)

    adata = AnnData(
        X=X,
        obs=obs.loc[:, ~obs.columns.isin(cols)].copy(),
        var=pd.DataFrame(index=unique_variables),
        uns={"scirpy_version": __version__},
    )
    adata.obsm["chain_indices"] = chain_indices.set_index(adata.obs_names)
    return adata
