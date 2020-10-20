from ..util import _is_na, _is_true, _is_false
import pandas as pd
import numpy as np


def _normalize_df_types(df: pd.DataFrame):
    """Convert all (text) representations of NA to np.nan.

    Modifies df inplace.
    """
    for col in df.columns:
        df.loc[_is_na(df[col]), col] = np.nan
