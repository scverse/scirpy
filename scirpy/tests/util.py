from ..util import _is_na, _is_true, _is_false
import pandas as pd
import numpy as np
from typing import List, Union


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan.

    Modifies df inplace.
    """
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)
        df.loc[_is_na(df[col]), col] = None
        df.loc[_is_true(df[col]), col] = True
        df.loc[_is_false(df[col]), col] = False


def _squarify(matrix: Union[List[List], np.ndarray]):
    """Squarify a upper triangular matrix"""
    matrix = np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "only works for square matrices"
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    assert np.allclose(matrix.T, matrix)
    return matrix
