import pandas as pd
import numpy as np
from textwrap import dedent
from typing import Union
from scipy.sparse import issparse
import scipy.sparse
import warnings

# Determine the right tqdm version, see https://github.com/tqdm/tqdm/issues/1082
try:
    import ipywidgets  # type: ignore # NOQA
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm  # NOQA


def _allclose_sparse(A, B, atol=1e-8):
    """Check if two sparse matrices are almost equal.

    From https://stackoverflow.com/questions/47770906/how-to-test-if-two-sparse-arrays-are-almost-equal/47771340#47771340
    """
    if np.array_equal(A.shape, B.shape) == 0:
        return False

    r1, c1, v1 = scipy.sparse.find(A)
    r2, c2, v2 = scipy.sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)

    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol, equal_nan=True)


def _is_symmetric(M) -> bool:
    """check if matrix M is symmetric"""
    if issparse(M):
        return _allclose_sparse(M, M.T)
    else:
        return np.allclose(M, M.T, 1e-6, 1e-6, equal_nan=True)


@np.vectorize
def _is_na(x):
    """Check if an object or string is NaN.
    The function is vectorized over numpy arrays or pandas Series
    but also works for single values.

    Pandas Series are converted to numpy arrays.
    """
    return _is_na2(x)


@np.vectorize
def _is_true(x):
    """Evaluates true for bool(x) unless _is_false(x) evaluates true.
    I.e. strings like "false" evaluate as False.

    Everything that evaluates to _is_na(x) evaluates evaluate to False.

    The function is vectorized over numpy arrays or pandas Series
    but also works for single values."""
    return _is_true2(x)


@np.vectorize
def _is_false(x):
    """Evaluates false for bool(False) and str("false")/str("False").
    The function is vectorized over numpy arrays or pandas Series.

    Everything that is NA as defined in `is_na()` evaluates to False.

    but also works for single values."""
    x = np.array(x).astype(object)

    return _is_false2(x)


def _is_na2(x):
    """the non-vectorized version, to be called from a function
    that gets vectorized"""
    return pd.isnull(x) or x in ("NaN", "nan", "None", "N/A", "")


def _is_true2(x):
    """Non-vectorized helper function"""
    return not _is_false2(x) and not _is_na2(x)


def _is_false2(x):
    """Non-vectorized helper function"""
    return (x in ("False", "false", "0") or not bool(x)) and not _is_na2(x)


def _normalize_counts(
    obs: pd.DataFrame, normalize: Union[bool, str], default_col: Union[None, str] = None
) -> pd.Series:
    """
    Produces a pd.Series with group sizes that can be used to normalize
    counts in a DataFrame.

    Parameters
    ----------
    normalize
        If False, returns a scaling factor of `1`
        If True, computes the group sizes according to `default_col`
        If normalize is a colname, compute the group sizes according to the colname.
    """
    if not normalize:
        return np.ones(obs.shape[0])
    elif isinstance(normalize, str):
        normalize_col = normalize
    elif normalize is True and default_col is not None:
        normalize_col = default_col
    else:
        raise ValueError("No colname specified in either `normalize` or `default_col")

    # https://stackoverflow.com/questions/29791785/python-pandas-add-a-column-to-my-dataframe-that-counts-a-variable
    return 1 / obs.groupby(normalize_col)[normalize_col].transform("count").values


def _doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec


def _read_to_str(path):
    """Read a file into a string"""
    with open(path, "r") as f:
        return f.read()


def deprecated(message):
    """Decorator to mark a function as deprecated"""

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function and will be removed in a "
                "future version of scirpy. {}".format(func.__name__, message),
                category=FutureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


def _translate_dna_to_protein(dna_seq: str):
    """Simple function to translate DNA to AA sequence.

    Avoid heavy dependencies such as skbio or Biopython.

    Taken from https://github.com/prestevez/dna2proteins/blob/master/dna2proteins.py
    Copyright (c) 2015 Patricio Rodrigo Estévez Soto
    """
    table = {
        "ATA": "I",
        "ATC": "I",
        "ATT": "I",
        "ATG": "M",
        "ACA": "T",
        "ACC": "T",
        "ACG": "T",
        "ACT": "T",
        "AAC": "N",
        "AAT": "N",
        "AAA": "K",
        "AAG": "K",
        "AGC": "S",
        "AGT": "S",
        "AGA": "R",
        "AGG": "R",
        "CTA": "L",
        "CTC": "L",
        "CTG": "L",
        "CTT": "L",
        "CCA": "P",
        "CCC": "P",
        "CCG": "P",
        "CCT": "P",
        "CAC": "H",
        "CAT": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGA": "R",
        "CGC": "R",
        "CGG": "R",
        "CGT": "R",
        "GTA": "V",
        "GTC": "V",
        "GTG": "V",
        "GTT": "V",
        "GCA": "A",
        "GCC": "A",
        "GCG": "A",
        "GCT": "A",
        "GAC": "D",
        "GAT": "D",
        "GAA": "E",
        "GAG": "E",
        "GGA": "G",
        "GGC": "G",
        "GGG": "G",
        "GGT": "G",
        "TCA": "S",
        "TCC": "S",
        "TCG": "S",
        "TCT": "S",
        "TTC": "F",
        "TTT": "F",
        "TTA": "L",
        "TTG": "L",
        "TAC": "Y",
        "TAT": "Y",
        "TAA": "_",
        "TAG": "_",
        "TGC": "C",
        "TGT": "C",
        "TGA": "_",
        "TGG": "W",
    }
    protein = []
    end = len(dna_seq) - (len(dna_seq) % 3) - 1
    for i in range(0, end, 3):
        codon = dna_seq[i : i + 3]
        if codon in table:
            aminoacid = table[codon]
            protein.append(aminoacid)
        else:
            protein.append("N")
    return "".join(protein)
