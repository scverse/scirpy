import pandas as pd


def _is_na(x):
    """Check if an object or string is NaN. 
    The function is vectorized over numpy arrays or pandas Series 
    but also works for single values. """
    return pd.isnull(x) | (x == "NaN") | (x == "nan") | (x == "None") | (x == "N/A")
