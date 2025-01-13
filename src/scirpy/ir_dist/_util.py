import warnings
from collections.abc import Hashable, Mapping, Sequence
from functools import reduce
from operator import mul
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix


def merge_coo_matrices(mats: Sequence[coo_matrix], shape=None) -> coo_matrix:
    """Fast sum of coo_matrices. Equivalent to builtin function `sum()`, but faster.

    The only execption is that while `sum` procudes `0` on an empty list, this function
    returns an empty coo matrix with shape `(0, 0)`. This makes downstream operations
    more consistent.

    This function makes use of the fact that COO matrices can have multiple entries
    for the same coordinates. When converting them to a different sparse matrix type
    they will be summed up internally.

    Parameters
    ----------
    mats
        Sequence of COO matrices
    shape
        Expected output shape. If None will infer from the input
        sequence. Will raise an error if one of the matrices is not consistent with
        the shape.
    """
    mats = list(mats)

    # check that shapes are consistent. matrices with shape 0 are ignored.
    for mat in mats:
        if mat.shape == (0, 0):
            continue
        elif shape is None:
            shape = mat.shape
        else:
            if mat.shape != shape:
                raise ValueError("Incompatible shapes")

    # sum is 0 if all inputs are 0 or the shape is 0 in one dimension
    if not len(mats) or shape is None or reduce(mul, shape) == 0:
        return coo_matrix((0, 0) if shape is None else shape)

    data, row, col = zip(*((x.data, x.row, x.col) for x in mats), strict=False)

    return sp.coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=shape)


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
        self.lookup: dict[Hashable, sp.coo_matrix] = {}

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
        in the lookup. Respects the distance type
        """
        if self.is_boolean:
            return np.zeros((1, self.size), dtype=bool)
        else:
            return sp.coo_matrix((1, self.size))

    def __getitem__(self, i):
        """Get mask for index `i`"""
        return self.lookup.get(i, self.empty())


class DoubleLookupNeighborFinder:
    def __init__(self, feature_table: pd.DataFrame, feature_table2: pd.DataFrame | None = None):
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
        in `O(|N|)` via double-lookup. The forward lookup-step retrieves
        the row/col index in the distance matrix for the entry associated with `o`.
        From the sparse matrix we directly obtain the indices in the distance matrix
        with distance != 0. The reverse lookup-step retrieves all neighbors `N` from
        the indices.

        Parameters
        ----------
        feature_table
            A data frame with features in columns. Rows must be unique.
            In our case, rows are clonotypes, and features can be CDR3 sequences,
            v genes, etc.
        feature_table2
            A second feature table. If omitted, computes pairwise distances
            between the rows of the first feature table
        """
        self.feature_table = feature_table
        self.feature_table2 = feature_table2 if feature_table2 is not None else feature_table

        # n_feature x n_feature sparse m x k distance matrices
        # where m is the number of unique features in feature_table and
        # k is the number if unique features in feature_table2
        self.distance_matrices: dict[str, sp.csr_matrix] = {}
        # mapping feature_label -> feature_index with len = n_feature for feature_table
        # (rows of the distance matrix)
        self.distance_matrix_labels: dict[str, dict] = {}
        # ... for feature_table2 (columns of the distance matrix)
        self.distance_matrix_labels2: dict[str, dict] = {}
        # tuples (dist_mat, forward, reverse)
        # dist_mat: name of associated distance matrix
        # forward: clonotype -> feature_index lookups
        # reverse: feature_index -> clonotype lookups
        self.lookups: dict[str, tuple[str, np.ndarray, ReverseLookupTable]] = {}

    @property
    def n_rows(self):
        return self.feature_table.shape[0]

    @property
    def n_cols(self):
        return self.feature_table2.shape[0]

    def lookup(
        self,
        object_ids: np.ndarray[int],
        forward_lookup_table_name: str,
        reverse_lookup_table_name: str | None = None,
    ) -> sp.csr_matrix:
        """
        Creates a distance matrix between objects with the given ids based on a feature distance matrix.

        To get the distance between two objects we need to look up the features of the two objects.
        The distance between those two features is then the distance between the two objects.

        To do so, we first use the `object_ids` together with the `forward_lookup_table` to look up
        the indices of the objects in the feature `distance_matrix`. Afterwards we pick the according row for each object
        out of the `distance_matrix` and construct a `rows` matrix (n_object_ids x n_features).

        "nan"s (index = -1) are not looked up in the feature `distance_matrix`, they return a row of zeros
        instead.

        Then we use the entries of the `reverse_lookup_table` to construct a `reverse_lookup_matrix` (n_features x n_object_ids).
        By multiplying the `rows` matrix with the `reverse_lookup_matrix` we get the final `object_distance_matrix` that shows
        the distances between the objects with the given `object_ids` regarding a certain feature column.

        It might not be obvious at the first sight that the matrix multiplication between `rows` and `reverse_lookup_matrix` gives
        us the desired result. But this trick allows us to use the built-in sparse matrix multiplication of `scipy.sparse`
        for enhanced performance.

        Parameters
        ----------
        object_ids
            The row indices of the feature_table.
        forward_lookup_table_name
            The unique identifier of a lookup table previously added via
            `add_lookup_table`.
        reverse_lookup_table_name
            The unique identifier of the lookup table used for the reverse lookup.
            If not provided will use the same lookup table for forward and reverse
            lookup. This is useful to calculate distances across features from
            different columns of the feature table (e.g. primary and secondary VJ chains).

        Returns
        -------
        object_distance_matrix
            A CSR matrix containing the pairwise distances between objects with the
            given `object_ids` regarding a certain feature column.
        """
        distance_matrix_name, forward_lookup_table, reverse_lookup_table = self.lookups[forward_lookup_table_name]

        if reverse_lookup_table_name is not None:
            distance_matrix_name_reverse, _, reverse_lookup_table = self.lookups[reverse_lookup_table_name]
            if distance_matrix_name != distance_matrix_name_reverse:
                raise ValueError("Forward and reverse lookup tablese must be defined on the same distance matrices.")

        distance_matrix = self.distance_matrices[distance_matrix_name]

        if np.max(distance_matrix.data) > np.iinfo(np.uint8).max:
            raise OverflowError(
                "The data values in the distance scipy.sparse.csr_matrix exceed the maximum value for uint8 (255)"
            )

        indices_in_dist_mat = forward_lookup_table[object_ids]
        indptr = np.empty(distance_matrix.indptr.shape[0] + 1, dtype=np.int64)
        indptr[:-1] = distance_matrix.indptr
        indptr[-1] = indptr[-2]
        distance_matrix_extended = sp.csr_matrix(
            (distance_matrix.data.astype(np.uint8), distance_matrix.indices, indptr),
            shape=(distance_matrix.shape[0] + 1, distance_matrix.shape[1]),
        )
        rows = distance_matrix_extended[indices_in_dist_mat, :]

        reverse_matrix_data = [np.array([], dtype=np.uint8)] * rows.shape[1]
        reverse_matrix_col = [np.array([], dtype=np.int64)] * rows.shape[1]
        nnz_array = np.zeros(rows.shape[1], dtype=np.int64)

        for key, value in reverse_lookup_table.lookup.items():
            reverse_matrix_data[key] = value.data
            reverse_matrix_col[key] = value.col
            nnz_array[key] = value.nnz

        data = np.concatenate(reverse_matrix_data)
        col = np.concatenate(reverse_matrix_col)
        indptr = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(nnz_array)])

        reverse_matrix = sp.csr_matrix((data, col, indptr), shape=(rows.shape[1], reverse_lookup_table.size))
        object_distance_matrix = rows * reverse_matrix
        return object_distance_matrix

    def add_distance_matrix(
        self,
        name: str,
        distance_matrix: sp.csr_matrix,
        labels: Sequence,
        labels2: Sequence | None = None,
    ):
        """Add a distance matrix.

        Parameters
        ----------
        name
            Unique identifier of the distance matrix
        distance_matrix
            sparse distance matrix `D` in CSR format
        labels
            array with row names of the distance matrix.
            `len(array) == D.shape[0]`
        labels2
            array with column names of the distance matrix.
            Can be omitted if the distance matrix is symmetric.
            `len(array) == D.shape[1]`.
        """
        labels2 = labels if labels2 is None else labels2
        if not len(labels) == distance_matrix.shape[0]:
            raise ValueError("Dimensions mismatch along axis 0")
        if not len(labels2) == distance_matrix.shape[1]:
            raise ValueError("Dimensions mismatch along axis 1")
        if not isinstance(distance_matrix, csr_matrix):
            raise TypeError("Distance matrix must be sparse and in CSR format. ")

        # The class relies on zeros not being explicitly stored during reverse lookup.
        distance_matrix.eliminate_zeros()
        self.distance_matrices[name] = distance_matrix
        self.distance_matrix_labels[name] = {k: i for i, k in enumerate(labels)}
        self.distance_matrix_labels2[name] = {k: i for i, k in enumerate(labels2)}
        # The label "nan" does not have an index in the matrix
        self.distance_matrix_labels[name]["nan"] = -1
        self.distance_matrix_labels2[name]["nan"] = -1

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
        unique identifier of the lookup table
        """
        forward = self._build_forward_lookup_table(feature_col, distance_matrix)
        reverse = self._build_reverse_lookup_table(feature_col, distance_matrix, dist_type=dist_type)
        self.lookups[name] = (distance_matrix, forward, reverse)

    def _build_forward_lookup_table(self, feature_col: str, distance_matrix: str) -> np.ndarray:
        """Create a lookup array that maps each clonotype to the respective
        index in the feature distance matrix.
        """
        return np.array(
            [self.distance_matrix_labels[distance_matrix][k] for k in self.feature_table[feature_col]],
        )

    def _build_reverse_lookup_table(
        self,
        feature_col: str,
        distance_matrix: str,
        *,
        dist_type: Literal["boolean", "numeric"],
    ) -> ReverseLookupTable:
        """Create a reverse-lookup dict that maps each (numeric) index
        of a feature distance matrix to a numeric or boolean mask.
        If the dist_type is numeric, will use a sparse numeric matrix.
        If the dist_type is boolean, use a dense boolean.
        """
        tmp_reverse_lookup = {}
        tmp_index_lookup = self.distance_matrix_labels2[distance_matrix]

        # Build reverse lookup
        for i, k in enumerate(self.feature_table2[feature_col]):
            tmp_key = tmp_index_lookup[k]
            if tmp_key == -1:
                continue
            try:
                tmp_reverse_lookup[tmp_key].append(i)
            except KeyError:
                tmp_reverse_lookup[tmp_key] = [i]

        return ReverseLookupTable.from_dict_of_indices(tmp_reverse_lookup, dist_type, self.n_cols)
