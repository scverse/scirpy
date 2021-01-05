from ..util import _is_na
import pandas as pd
import numpy as np
from typing import List, Union


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan.

    Modifies df inplace.
    """
    for col in df.columns:
        df.loc[_is_na(df[col]), col] = np.nan


def _squarify(matrix: Union[List[List], np.ndarray]):
    """Squarify a upper triangular matrix"""
    matrix = np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "only works for square matrices"
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    assert np.allclose(matrix.T, matrix)
    return matrix
