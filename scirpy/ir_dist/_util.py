import numpy as np
import pandas as pd
from typing import Dict, Hashable, Sequence, Tuple, Union, Mapping
import scipy.sparse as sp
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from .._compat import Literal
import warnings
from functools import reduce
from operator import mul


def merge_coo_matrices(mats):
    """Fast sum of coo_matrices. Equivalent to builtin function `sum()`, but faster."""
    mats = list(mats)

    # special case: empty list - sum returns 0
    if not len(mats):
        return 0

    # check that shapes are consistent
    shape = mats[0].shape
    for mat in mats:
        if mat.shape != shape:
            raise ValueError("Incompatible shapes")

    # return empty matrix if one dimension is of length 0
    if reduce(mul, shape) == 0:
        return sp.coo_matrix(shape)

    data, row, col = zip(*((x.data, x.row, x.col) for x in mats))

    return sp.coo_matrix(
        (np.hstack(data), (np.hstack(row), np.hstack(col))), shape=shape
    )


def reduce_or(*args, chain_count=None):
    """Reduce two or more (sparse) masys by OR as if they were boolean:
    Take minimum, ignore 0s and nans.

    All arrays must be of a float dtype (to support nan and inf)
    """
    # ignore runtime warnings due to NAN-slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        tmp_array = np.vstack(args)
        assert np.issubdtype(tmp_array.dtype, np.floating)
        tmp_array[tmp_array == 0] = np.inf
        tmp_array = np.nanmin(tmp_array, axis=0)
        tmp_array[np.isinf(tmp_array)] = 0
        return tmp_array


def reduce_and(*args, chain_count):
    """Reduce two or more (sparse) masks by AND as if they were boolean:
    Take maximum, ignore nans.

    All arrays must be of a float dtype (to support nan and inf)

    Only entries that have the same chain count (e.g. clonotypes with both TRA_1
    and TRA_2) are comparable.
    """
    # ignore runtime warnings due to NAN-slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        tmp_array = np.vstack(args)
        assert np.issubdtype(tmp_array.dtype, np.floating)
        tmp_array[tmp_array == 0] = np.inf
        same_count_mask = np.sum(~np.isnan(tmp_array), axis=0) == chain_count
        tmp_array = np.nanmax(tmp_array, axis=0)
        tmp_array[np.isinf(tmp_array)] = 0
        tmp_array = np.multiply(tmp_array, same_count_mask)
        return tmp_array


class ReverseLookupTable:
    def __init__(self, dist_type: Literal["boolean", "numeric"], size: int):
        """Reverse lookup table holds a mask that indicates which objects
        are neighbors of an object with a given index `i`.

        It respects two types:
            * boolean -> dense boolean mask
            * numeric -> sparse integer row

        The boolean mask is the more efficient option for features with many
        neighbors per value (e.g. most of the cells may be of receptor_type TCR).
        The numeric mask is more efficient for features with few neighbors
        per value (e.g. there are likely only few neighbors for a specific CDR3
        sequence).

        Parameters
        ----------
        dist_type
            Either `boolean` or `numeric`
        size
            The size of the masks.
        """
        if dist_type not in ["boolean", "numeric"]:
            raise ValueError("invalid dist_type")
        self.dist_type = dist_type
        self.size = size
        self.lookup: Dict[Hashable, sp.coo_matrix] = dict()

    @staticmethod
    def from_dict_of_indices(
        dict_of_indices: Mapping,
        dist_type: Literal["boolean", "numeric"],
        size: int,
    ):
        """Convert a dict of indices to a ReverseLookupTable of row masks.

        Parameters
        ----------
        dict_of_indices
            Dictionary mapping each index to a list of indices.
        dist_type
            Either `boolean` or `numeric`
        size
            The size of the masks
        """
        rlt = ReverseLookupTable(dist_type, size)

        # convert into coo matrices (numeric distances) or numpy boolean arrays.
        for k, v in dict_of_indices.items():
            if rlt.is_boolean:
                tmp_array = np.zeros(shape=(1, size), dtype=bool)
                tmp_array[0, v] = True
                rlt.lookup[k] = tmp_array
            else:
                rlt.lookup[k] = sp.coo_matrix(
                    (
                        np.ones(len(v), dtype=np.uint8),
                        (np.zeros(len(v), dtype=int), v),
                    ),
                    shape=(1, size),
                )
        return rlt

    @property
    def is_boolean(self):
        return self.dist_type == "boolean"

    def empty(self):
        """Create an empty row with same dimensions as those stored
        in the lookup. Respects the distance type"""
        if self.is_boolean:
            return np.zeros((1, self.size), dtype=bool)
        else:
            return sp.coo_matrix((1, self.size))

    def __getitem__(self, i):
        """Get mask for index `i`"""
        return self.lookup.get(i, self.empty())


class DoubleLookupNeighborFinder:
    def __init__(self, feature_table: pd.DataFrame):
        """
        A datastructure to efficiently retrieve distances based on different features.

        The datastructure essentially consists of
            * a feature table, with objects in rows (=clonotypes) and features in
              columns (=sequence features, v-genes, etc. )
            * a set of sparse distance matrices, one for each feature. A distance matrix
              can be re-used (e.g. a VJ-sequence distance matrix can be reused for
              both the primary and secondary immune receptor. Distance matrices
              are added via `add_distance_matrix`
            * a set of lookup tables, one for each feature. Lookup tables can be
              constructed via `add_lookup_table`.

        For each object `o`, all neighbors `N` with a distance != 0 can be retrieved
        in `O(|N|))` via double-lookup. The forward lookup-step retrieves
        the row/col index in the distance matrix for the entry associated with `o`.
        From the spare matrix we directly obtain the indices in the distance matrix
        with distance != 0. The reverse lookup-step retrieves all neighbors `N` from
        the indices.

        Parameters
        ----------
        feature_table
            A data frame with features in columns. Rows must be unique.
            In our case, rows are clonotypes, and features can be CDR3 sequences,
            v genes, etc.
        """
        self.feature_table = feature_table

        # n_feature x n_feature sparse, symmetric distance matrices
        self.distance_matrices: Dict[str, sp.csr_matrix] = dict()
        # mapping feature_label -> feature_index with len = n_feature
        self.distance_matrix_labels: Dict[str, dict] = dict()
        # tuples (dist_mat, forward, reverse)
        # dist_mat: name of associated distance matrix
        # forward: clonotype -> feature_index lookups
        # reverse: feature_index -> clonotype lookups
        self.lookups: Dict[str, Tuple[str, np.ndarray, ReverseLookupTable]] = dict()

    @property
    def n_rows(self):
        return self.feature_table.shape[0]

    def lookup(
        self,
        object_id: int,
        forward_lookup_table: str,
        reverse_lookup_table: Union[str, None] = None,
    ) -> Union[coo_matrix, np.ndarray]:
        """Get ids of neighboring objects from a lookup table.

        Performs the following lookup:

            clonotype_id -> dist_mat -> neighboring features -> neighboring object.

        "nan"s are not looked up via the distance matrix, they return a row of zeros
        instead.

        Parameters
        ----------
        object_id
            The row index of the feature_table.
        forward_lookup_table
            The unique identifier of a lookup table previously added via
            `add_lookup_table`.
        reverse_lookup_table
            The unique identifier of the lookup table used for the reverse lookup.
            If not provided will use the same lookup table for forward and reverse
            lookup. This is useful to calculate distances across features from
            different columns of the feature table.
        """
        distance_matrix_name, forward, reverse = self.lookups[forward_lookup_table]

        if reverse_lookup_table is not None:
            distance_matrix_name_reverse, _, reverse = self.lookups[
                reverse_lookup_table
            ]
            if distance_matrix_name != distance_matrix_name_reverse:
                raise ValueError(
                    "Forward and reverse lookup tablese must be defined "
                    "on the same distance matrices."
                )

        distance_matrix = self.distance_matrices[distance_matrix_name]
        idx_in_dist_mat = forward[object_id]
        if np.isnan(idx_in_dist_mat):
            return reverse.empty()
        else:
            # get distances from the distance matrix...
            row = distance_matrix[idx_in_dist_mat, :]

            if reverse.is_boolean:
                assert (
                    len(row.indices) == 1  # type: ignore
                ), "Boolean reverse lookup only works for identity distance matrices."
                return reverse[row.indices[0]]  # type: ignore
            else:
                # ... and get column indices directly from sparse row
                # sum concatenates coo matrices
                return merge_coo_matrices(
                    (
                        reverse[i] * multiplier
                        for i, multiplier in zip(row.indices, row.data)  # type: ignore
                    )
                )  # type: ignore

    def add_distance_matrix(
        self, name: str, distance_matrix: sp.csr_matrix, labels: Sequence
    ):
        """Add a distance matrix.

        Parameters
        ----------
        name
            Unique identifier of the distance matrix
        distance_matrix
            sparse distance matrix `D` in CSR format
        labels
            array with row/column names of the distance matrix.
            `len(array) == D.shape[0] == D.shape[1]`
        """
        if not (len(labels) == distance_matrix.shape[0] == distance_matrix.shape[1]):
            raise ValueError("Dimension mismatch!")
        if not isinstance(distance_matrix, csr_matrix):
            raise TypeError("Distance matrix must be sparse and in CSR format. ")

        # The class relies on zeros not being explicitly stored during reverse lookup.
        distance_matrix.eliminate_zeros()
        self.distance_matrices[name] = distance_matrix
        self.distance_matrix_labels[name] = {k: i for i, k in enumerate(labels)}
        # The label "nan" does not have an index in the matrix
        self.distance_matrix_labels[name]["nan"] = np.nan

    def add_lookup_table(
        self,
        name: str,
        feature_col: str,
        distance_matrix: str,
        *,
        dist_type: Literal["boolean", "numeric"] = "numeric",
    ):
        """Build a pair of forward- and reverse-lookup tables.

        Parameters
        ----------
        feature_col
            a column name in `self.feature_table`
        distance_matrix
            name of a distance matrix previously added via `add_distance_matrix`
        name
            unique identifier of the lookup table"""
        forward = self._build_forward_lookup_table(feature_col, distance_matrix)
        reverse = self._build_reverse_lookup_table(
            feature_col, distance_matrix, dist_type=dist_type
        )
        self.lookups[name] = (distance_matrix, forward, reverse)

    def _build_forward_lookup_table(
        self, feature_col: str, distance_matrix: str
    ) -> np.ndarray:
        """Create a lookup array that maps each clonotype to the respective
        index in the feature distance matrix.
        """
        return np.array(
            [
                self.distance_matrix_labels[distance_matrix][k]
                for k in self.feature_table[feature_col]
            ]
        )

    def _build_reverse_lookup_table(
        self,
        feature_col: str,
        distance_matrix: str,
        *,
        dist_type: Literal["boolean", "numberic"],
    ) -> ReverseLookupTable:
        """Create a reverse-lookup dict that maps each (numeric) index
        of a feature distance matrix to a numeric or boolean mask.
        If the dist_type is numeric, will use a sparse numeric matrix.
        If the dist_type is boolean, use a dense boolean.
        """
        tmp_reverse_lookup = dict()
        tmp_index_lookup = self.distance_matrix_labels[distance_matrix]

        # Build reverse lookup
        for i, k in enumerate(self.feature_table[feature_col]):
            tmp_key = tmp_index_lookup[k]
            if np.isnan(tmp_key):
                continue
            try:
                tmp_reverse_lookup[tmp_key].append(i)
            except KeyError:
                tmp_reverse_lookup[tmp_key] = [i]

        return ReverseLookupTable.from_dict_of_indices(
            tmp_reverse_lookup, dist_type, self.n_rows
        )
