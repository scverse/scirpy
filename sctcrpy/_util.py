import pandas as pd
import numpy as np
from textwrap import dedent
from typing import Any
from anndata import AnnData
from collections import namedtuple


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


def _add_to_uns(
    adata: AnnData, tool: str, result: Any, *, parameters: dict = None, domain="sctcrpy"
) -> None:
    """Store results of a tool in `adata.uns`.
    
    Parameters
    ----------
    adata
        Annotated data matrix
    tool
        Name of the tool (=dict key of adata.uns)
    result
        Result to store 
    parameters
        Parameters the tool was ran with. If `None`, it is assumed 
        that the tools does not take parameters and the result
        is directly stored in `uns[domain][tool]`. 
        Otherwise, the parameters are converted into a named tuple
        that is used as a dict key: `uns[domain][tool][param_named_tuple] = result`. 
    domain
        top level key of `adata.uns` to store results under. 
    """
    if domain not in adata.uns:
        adata.uns[domain] = dict()

    if parameters is None:
        adata.uns[domain][tool] = result
    else:
        if tool not in adata.uns[domain]:
            adata.uns[domain][tool] = dict()
        assert isinstance(adata.uns[domain][tool], dict)
        Parameters = namedtuple("Parameters", sorted(parameters))
        param_tuple = Parameters(**parameters)
        adata.uns[domain][tool][param_tuple] = result


def _get_from_uns(adata: AnnData, tool: str, *, parameters: dict = None) -> Any:
    """Get results of a tool from `adata.uns`. 

    Parameters
    ----------
    adata
        annotated data matrix
    tool
        name of the tool
    parameters
        Parameters the tool was ran with. If `None` it is assumed 
        that the tools does not take parameters and the result is directly 
        stored in `uns[domain][tool]`. Otherwise, the parameters are converted 
        into a named tuple that is used as dict key: 
        `uns[domain][tool][param_named_tuple]`. Raises a KeyError if no such 
        entry exists. 

    Raises
    ------
    KeyError
        If no entry for the tool or for the given parameters exist. 

    Returns
    -------
    The stored result. 
    """
    if parameters is None:
        return adata.uns["sctcrpy"][tool]
    else:
        Parameters = namedtuple("Parameters", sorted(parameters))
        param_tuple = Parameters(**parameters)
        return adata.uns["sctcrpy"][tool][param_tuple]


def _doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec
