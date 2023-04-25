import warnings
from textwrap import dedent
from typing import Any, Callable, Mapping, Optional, Sequence, Union, cast, overload

import awkward as ak
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from mudata import MuData
from scanpy import logging
from scipy.sparse import issparse
from tqdm.auto import tqdm

# reexport tqdm (here was previously a workaround for https://github.com/tqdm/tqdm/issues/1082)
__all__ = ["tqdm"]


def _doc_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec


class DataHandler:
    """\
    Transparent access to airr modality in both AnnData and MuData objects.

    Performs a plausibility check of the input data for public scirpy functions.

    Provides convenient accessors to the airr data that is stored somewhere in
    the input AnnData/MuData.

    DataHandler may be called with another DataHandler instance as `data` attribute. In that
    case all attributes are taken from the existing DataHandler instance and all keyword attributes
    are ignored.

    Parameters
    ----------
    {adata}
    {airr_mod}
    {airr_key}
    {chain_idx_key}
    """

    TYPE = Union[AnnData, MuData, "DataHandler"]

    @overload
    @staticmethod
    def default(data: None) -> None:
        ...

    @overload
    @staticmethod
    def default(data: "DataHandler.TYPE") -> "DataHandler":
        ...

    @staticmethod
    def default(data):
        """Initailize a DataHandler object with default keys. Returns None if `data` is None.
        Particularly useful for testing."""
        if data is not None:
            return DataHandler(data, "airr", "airr", "chain_indices")

    def __init__(
        self,
        data: "DataHandler.TYPE",
        airr_mod: Optional[str] = None,
        airr_key: Optional[str] = None,
        chain_idx_key: Optional[str] = None,
    ):
        if isinstance(data, DataHandler):
            self._data = data._data
            self._airr_mod = data._airr_mod
            self._airr_key = data._airr_key
            self._chain_idx_key = data._chain_idx_key
        else:
            self._data = data
            self._airr_mod = airr_mod
            self._airr_key = airr_key
            self._chain_idx_key = chain_idx_key

        # check for outdated schema
        self._check_airr_key_in_obsm()
        self._check_chain_indices()

    def _check_chain_indices(self):
        """Check if chain indices are available. Compute chain indices with default parameters
        if not available.

        Skip if no airr key has been defined.
        """
        if (
            self._airr_key is not None
            and self._chain_idx_key is not None
            and self._chain_idx_key not in self.adata.obsm
        ):
            # import here to avoid circular import
            from ..pp import index_chains

            logging.warning(
                f"No chain indices found under adata.obsm['{self._chain_idx_key}']. "
                "Running scirpy.pp.index_chains with default parameters. ",
            )

            index_chains(
                self.adata, airr_key=self._airr_key, key_added=self._chain_idx_key
            )

    @overload
    def get_obs(self, columns: str) -> pd.Series:
        ...

    @overload
    def get_obs(self, columns: Sequence[str]) -> pd.DataFrame:
        ...

    def get_obs(self, columns):
        """\
        Get one or multiple obs columns from either MuData or AIRR AnnData

        Checks if the column is available in `MuData.obs`. If it can't be found or DataHandler is initalized without MuData
        object, `AnnData.obs` is tried.

        The returned object always has the dimensions and index of MuData, even if
        only columns from AnnData are used. It is easy to subset to AnnData if required:

        .. code-block:: python

            params.get_obs([col1, col2]).reindex(params.adata.obs_names)

        Parameters
        ----------
        columns
            one or multiple columns.

        Returns
        -------
        If this is a single column passed as `str`, a :class:`~pandas.Series` will be returned,
        otherwise a :class:`~pandas.DataFrame`.
        """
        if isinstance(columns, str):
            return self._get_obs_col(columns)
        else:
            if len(columns):
                df = pd.concat({c: self._get_obs_col(c) for c in columns}, axis=1)
                assert df.index.is_unique, "Index not unique"
                return df.reindex(self.data.obs_names)
            else:
                # return empty dataframe (only index) if no columns are specified
                return self.data.obs.loc[:, []]

    def _get_obs_col(self, column: str) -> pd.Series:
        try:
            return self.mdata.obs[column]
        except (KeyError, AttributeError):
            return self.adata.obs[column]

    def set_obs(
        self, key: str, value: Union[pd.Series, Sequence[Any], np.ndarray]
    ) -> None:
        """Store results in .obs of AnnData and MuData.

        The result will be written to `mdata.obs["{airr_mod}:{key}"]` and to `adata.obs[key]`.
        """
        # index series with AnnData (in case MuData has different dimensions)
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=self.adata.obs_names)
        if isinstance(self.data, MuData):
            # write to both AnnData and MuData
            if self._airr_mod is None:
                raise ValueError(
                    "Trying to write to both AnnData and Mudata, but no `airr_mod` is specified."
                )
            mudata_key = f"{self._airr_mod}:{key}"
            adata_key = key

            self.mdata.obs[mudata_key] = value
            self.adata.obs[adata_key] = value
            logging.info(f'Stored result in `mdata.obs["{self._airr_mod}:{key}"]`.')
        else:
            self.data.obs[key] = value
            logging.info(f'Stored result in `adata.obs["{key}"]`.')

    @property
    def chain_indices(self) -> ak.Array:
        """Reference to the chain indices

        Raises an AttributeError if chain indices are not available."""
        if self._chain_idx_key is not None:
            return cast(ak.Array, self.adata.obsm[self._chain_idx_key])
        else:
            raise AttributeError("DataHandler was initialized without chain indices.")

    @property
    def airr(self) -> ak.Array:
        """reference to the awkward array with AIRR information."""
        if self._airr_key is not None:
            return cast(ak.Array, self.adata.obsm[self._airr_key])
        else:
            raise AttributeError("DataHandler was initialized wihtout airr information")

    @property
    def adata(self) -> AnnData:
        """Reference to the AnnData object of the AIRR modality."""
        if isinstance(self._data, AnnData):
            return self._data
        else:
            if self._airr_mod is not None:
                try:
                    return self._data.mod[self._airr_mod]
                except KeyError:
                    raise KeyError(
                        f"There is no AIRR modality in MuData under key '{self._airr_mod}'"
                    )
            else:
                raise AttributeError(
                    "DataHandler was initalized with MuData, but without specifying a modality"
                )

    @property
    def data(self) -> Union[MuData, AnnData]:
        """Get the outermost container. If MuData is defined, return the MuData object.
        Otherwise the AnnData object."""
        return self._data

    @property
    def mdata(self) -> MuData:
        """Reference to the MuData object.

        Raises an attribute error if only AnnData is available."""
        if isinstance(self._data, MuData):
            return self._data
        else:
            raise AttributeError("DataHandler was initalized with only AnnData")

    def strings_to_categoricals(self):
        """Convert strings to categoricals. If MuData is not defined, perform this on AnnData"""
        try:
            self.mdata.strings_to_categoricals()
        except AttributeError:
            self.adata.strings_to_categoricals()

    @staticmethod
    def inject_param_docs(
        **kwargs: str,
    ) -> Callable:
        """Inject parameter documentation into a function docstring

        Parameters
        ----------
        **kwargs
            Further, custom {keys} to replace in the docstring.
        """
        doc = {}
        doc["adata"] = dedent(
            """\
            adata
                AnnData or MuData object that contains :term:`AIRR` information. 
            """
        )
        doc["airr_mod"] = dedent(
            """\
            airr_mod
                Name of the modality with :term:`AIRR` information is stored in 
                the :class:`~mudata.MuData` object. if an :class:`~anndata.AnnData` object
                is passed to the function, this parameter is ignored.
            """
        )
        doc["airr_key"] = dedent(
            """\
            airr_key
                Key under which the :term:`AIRR` information is stored in adata.obsm as an 
                :term:`awkward array`.
            """
        )
        doc["chain_idx_key"] = dedent(
            """\
            chain_idx_key
                Key under which the chain indices are stored in adata.obsm. 
                If chain indices are not present, :func:`~scirpy.pp.index_chains` is
                run with default parameters. 
            """
        )
        doc["inplace"] = dedent(
            """\
            inplace
                If `True`, a column with the result will be stored in `obs`. Otherwise the result will be returned.  
            """
        )
        doc["key_added"] = dedent(
            """\
            key_added
                Key under which the result will be stored in `obs`, if `inplace` is `True`. When the function is running 
                on :class:`~mudata.MuData`, the result will be written to both `mdata.obs["{airr_mod}:{key_added}"]` and 
                `mdata.mod[airr_mod].obs[key_added]`. 
            """
        )
        return _doc_params(**doc, **kwargs)

    @staticmethod
    def check_schema_pre_v0_7(adata: AnnData):
        """Raise an error if AnnData is in pre scirpy v0.7 format."""
        if (
            # I would actually only use `scirpy_version` for the check, but
            # there might be cases where it gets lost (e.g. when rebuilding AnnData).
            # If a `v_call` is present, that's a safe sign that it is the AIRR schema, too
            "has_ir" in adata.obs.columns
            and (
                "IR_VJ_1_v_call" not in adata.obs.columns
                and "IR_VDJ_1_v_call" not in adata.obs.columns
            )
            and "scirpy_version" not in adata.uns
        ):
            raise ValueError(
                "It seems your anndata object is of a very old format used by scirpy < v0.7 "
                "which is not supported anymore. You might be best off reading in your data from scratch. "
                "If you absolutely want to, you can use scirpy < v0.13 to read that format, "
                "convert it to a legacy format, and convert it again using the most recent version of scirpy. "
            )

    def _check_airr_key_in_obsm(self):
        """Check if `adata` uses the latest scirpy schema.

        Raises ValueError if it doesn't.

        If `_airr_key` is not specified, we do not perform any check (could be a valid, non-scirpy anndata object)
        """
        if self._airr_key is None:
            return
        if self._airr_key not in self.adata.obsm:
            # First check for very old version. We don't support it at all anymore.
            DataHandler.check_schema_pre_v0_7(self.adata)
            if "IR_VJ_1_junction_aa" in self.adata.obs.columns:
                # otherwise suggest to use `upgrade_schema`
                raise ValueError(
                    f"No AIRR data found in adata.obsm['{self._airr_key}']. "
                    "Your AnnData object might be using an outdated schema. "
                    "Scirpy has updated the format of `adata` in v0.13. AIRR data is now stored as an "
                    "awkward array in `adata.obsm['airr']`."
                    "Please run `ir.io.upgrade_schema(adata) to update your AnnData object to "
                    "the latest version. "
                )
            else:
                raise KeyError(
                    dedent(
                        f"""\
                        No AIRR data found in adata.obsm['{self._airr_key}']. 
                        See https://scverse.org/scirpy/develop/data-structure.html for more info about the scirpy data structure.
                        """
                    )
                )


DataHandler = DataHandler.inject_param_docs()(DataHandler)


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


def _is_na2(x):
    """Check if an object or string is NaN.
    The function is vectorized over numpy arrays or pandas Series
    but also works for single values.

    Pandas Series are converted to numpy arrays.
    """
    return pd.isnull(x) or x in ("NaN", "nan", "None", "N/A", "")


_is_na = np.vectorize(_is_na2, otypes=[bool])


def _is_true2(x):
    """Evaluates true for bool(x) unless _is_false(x) evaluates true.
    I.e. strings like "false" evaluate as False.

    Everything that evaluates to _is_na(x) evaluates evaluate to False.

    The function is vectorized over numpy arrays or pandas Series
    but also works for single values."""
    return not _is_false2(x) and not _is_na2(x)


_is_true = np.vectorize(_is_true2, otypes=[bool])


def _is_false2(x):
    """Evaluates false for bool(False) and str("false")/str("False").
    The function is vectorized over numpy arrays or pandas Series.

    Everything that is NA as defined in `is_na()` evaluates to False.

    but also works for single values."""
    return (x in ("False", "false", "0") or not bool(x)) and not _is_na2(x)


_is_false = np.vectorize(
    lambda x: _is_false2(np.array(x).astype(object)), otypes=[bool]
)


def _normalize_counts(
    obs: pd.DataFrame, normalize: Union[bool, str], default_col: Union[None, str] = None
) -> np.ndarray:
    """
    Produces an array with group sizes that can be used to normalize
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
    return 1 / cast(
        np.ndarray, obs.groupby(normalize_col)[normalize_col].transform("count").values
    )


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
