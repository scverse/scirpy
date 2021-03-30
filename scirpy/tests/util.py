import anndata
from ..util import _is_na, _is_true, _is_false
import pandas as pd
import numpy as np
from typing import List, Union
from tempfile import NamedTemporaryFile
import gzip


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan.

    Modifies df inplace.
    """
    return
    for col in df.columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].astype(str)
        df.loc[_is_na(df[col]), col] = None
        df.loc[_is_true(df[col]), col] = True
        df.loc[_is_false(df[col]), col] = False


def _write_h5ad_gz(adata, filename):
    """Write, then compress an anndata file. If only obs is stored, this results
    in significantly smaller filese than using the `compression` flag.
    """
    with NamedTemporaryFile() as tmpf:
        adata.write_h5ad(tmpf.name)
        with open(tmpf.name, "rb") as src, gzip.open(filename, "wb") as dst:
            dst.writelines(src)


def _read_h5ad_gz(filename):
    with NamedTemporaryFile() as tmpf:
        with open(tmpf.name, "wb") as dst, gzip.open(filename, "rb") as src:
            dst.writelines(src)
        return anndata.read_h5ad(tmpf.name)


def _squarify(matrix: Union[List[List], np.ndarray]):
    """Squarify a upper triangular matrix"""
    matrix = np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "only works for square matrices"
    i_lower = np.tril_indices(matrix.shape[0], -1)
    matrix[i_lower] = matrix.T[i_lower]
    assert np.allclose(matrix.T, matrix)
    return matrix
