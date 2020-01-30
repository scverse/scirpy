import pandas as pd
import numpy as np


def _is_na(x):
    """Check if an object or string is NaN. 
    The function is vectorized over numpy arrays or pandas Series 
    but also works for single values. """
    return pd.isnull(x) | (x == "NaN") | (x == "nan") | (x == "None") | (x == "N/A")


def _is_true(x):
    """Evaluates true for bool(x) unless _is_false(x) evaluates true. 
    I.e. strings like "false" evaluate as False. 

    Everything that evaluates to _is_na(x) evaluates evaluate to False. 

    The function is vectorized over numpy arrays or pandas Series 
    but also works for single values.  """
    return ~_is_false(x) & ~_is_na(x)


def _is_false(x):
    """Evaluates false for bool(False) and str("false")/str("False"). 
    The function is vectorized over numpy arrays or pandas Series. 

    Everything that is NA as defined in `is_na()` evaluates to False. 
    
    but also works for single values.  """
    if hasattr(x, "astype"):
        x = x.astype(object)
    return np.bool_(
        ((x == "False") | (x == "false") | (x == "0") | ~np.bool_(x))
        & ~np.bool_(_is_na(x))
    )
