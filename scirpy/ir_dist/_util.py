import collections.abc as cabc
import abc
from functools import reduce
from operator import add
import numpy as np
import pandas as pd
import itertools
from typing import Dict, Sequence, Tuple, Iterable, Union, Mapping
import scipy.sparse as sp
from scipy.sparse.csr import csr_matrix
from .._compat import Literal


class SetMask(abc.ABC):
    """A sparse mask vector that supports set operations"""

    def __init__(self, data: csr_matrix):
        if not isinstance(data, csr_matrix):
            raise ValueError("data must be a csr_matrix")
        self.data = data
        pass

    def __repr__(self):
        try:
            return f"1x{len(self)} {type(self).__name__} with {self.data.nzz} elements"
        except AttributeError:
            return f"uninitalized {type(self).__name__}"

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, key):
        return self.data[0, key]

    def __eq__(self, other) -> bool:
        """Not very efficient, mainly for testing"""
        return type(self) == type(other) and np.array_equal(
            self.data.toarray(), other.data.toarray()
        )


class BoolSetMask(SetMask):
    """A SetMask of boolean type. I.e. it does not contain values, only 1/0.

    There are no guard rails regarding the actual values. If you store values
    different than 0/1 in the bool set mask strange things may happen.
    """

    @staticmethod
    def empty(size: int):
        """Create an empty setmask of a given length"""
        data = sp.csr_matrix((1, size))
        return BoolSetMask(data)

    @staticmethod
    def from_list(lst):
        data = sp.csr_matrix(lst)
        return BoolSetMask(data)

    def __or__(self, other):
        if isinstance(other, BoolSetMask):
            return BoolSetMask(self.data.maximum(other.data))
        if isinstance(other, NumberSetMask):
            # neutral element! valuese that are 1 in the BoolSetMask and 0 in the
            # NumberSetMask will stay 0.
            return other
        else:
            return NotImplemented

    def __ror__(self, other):
        return self.__or__(other)

    def __and__(self, other):
        if isinstance(other, BoolSetMask):
            return BoolSetMask(self.data.multiply(other.data))
        elif isinstance(other, NumberSetMask):
            return NumberSetMask(self.data.multiply(other.data))
        else:
            return NotImplemented

    def __rand__(self, other):
        return self.__and__(other)


class NumberSetMask(SetMask):
    """A SetMask of number type. Or retains the min, and retains the max"""

    @staticmethod
    def empty(size: int):
        """Create an empty setmask of a given length"""
        data = sp.csr_matrix((1, size))
        return NumberSetMask(data)

    @staticmethod
    def from_list(lst):
        data = sp.csr_matrix(lst)
        return NumberSetMask(data)

    def __or__(self, other):
        if isinstance(other, NumberSetMask):
            # take min (!= 0)
            tmp_max = (
                self.data.maximum(other.data)
                .multiply(self.data > 0)
                .multiply(other.data > 0)
            )
            return NumberSetMask(self.data + other.data - tmp_max)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, NumberSetMask):
            return NumberSetMask(
                self.data.maximum(other.data)
                .multiply(self.data > 0)
                .multiply(other.data > 0)
            )
        else:
            return NotImplemented


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
        self.lookups: Dict[str, Tuple[str, np.ndarray, Mapping]] = dict()

    def lookup(
        self,
        object_id: int,
        forward_lookup_table: str,
        reverse_lookup_table: Union[str, None] = None,
    ) -> SetMask:
        """Get ids of neighboring objects from a lookup table.

        Performs the following lookup:

            clonotype_id -> dist_mat -> neighboring features -> neighboring object.

        "nan"s are not looked up via the distance matrix. Instead they have
        a special entry in the lookup tables and yield only a BooleanMask that
        works as neutral element.

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
            try:
                return BoolSetMask(reverse["nan"])
            except KeyError:
                return NumberSetMask.empty(self.feature_table.shape[0])
        else:
            # get distances from the distance matrix...
            row = distance_matrix[idx_in_dist_mat, :]

            # ... and get column indices directly from sparse row
            #
            # the individual sparse masks obtained from the reverse lookup
            # table are not overlapping because each row in the feature table
            # has only one sequence. Simple sum is therefore enough.
            return NumberSetMask(
                # TODO possibly more efficient using either dense or coo_matrices here.
                # vstack + sum densifies.
                reduce(
                    add,
                    (
                        # if no entry found (e.g. because of cross-table lookup)
                        # return nothing (-> empty iterator)
                        reverse.get(i, sp.csr_matrix((1, self.feature_table.shape[0])))
                        * distance
                        for i, distance in zip(row.indices, row.data)  # type: ignore
                    ),
                )
            )

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

    def add_lookup_table(self, name: str, feature_col: str, distance_matrix: str):
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
        reverse = self._build_reverse_lookup_table(feature_col, distance_matrix)
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
        self, feature_col: str, distance_matrix: str
    ) -> Mapping[float, csr_matrix]:
        """Create a reverse-lookup dict that maps each (numeric) index
        of a feature distance matric to a list of associated numeric clonotype indices."""
        tmp_reverse_lookup = dict()
        tmp_index_lookup = self.distance_matrix_labels[distance_matrix]
        for i, k in enumerate(self.feature_table[feature_col]):
            tmp_key = tmp_index_lookup[k]
            if np.isnan(tmp_key):
                tmp_key = "nan"
            try:
                tmp_reverse_lookup[tmp_key].append(i)
            except KeyError:
                tmp_reverse_lookup[tmp_key] = [i]

        # convert all of them to sets. In particular nan needs to be a set
        # nan can be quite large and like that it doesn't have to be re-initalized
        # all over.
        return {
            k: self._list_to_sparse_row(v, self.feature_table.shape[0])
            for k, v in tmp_reverse_lookup.items()
        }

    @staticmethod
    def _list_to_sparse_row(index_list: Sequence[int], row_len: int) -> sp.csr_matrix:
        """Efficient way to convert a list of indexes to a sparse 1 x n mask"""
        sparse_row = sp.csr_matrix((1, row_len))
        sparse_row.data = np.ones(len(index_list))
        sparse_row.indices = np.array(index_list)
        sparse_row.indptr = np.array([0, len(index_list)])
        return sparse_row

    @staticmethod
    def _dict_to_sparse_row(row_dict: Mapping, row_len: int) -> sp.csr_matrix:
        """Efficient way of converting a key, value dictionary to a
        1 x n sparse row in CSR format"""
        sparse_row = sp.csr_matrix((1, row_len))

        sparse_row.data = np.fromiter(
            (x if np.isfinite(x) else 0 for x in row_dict.values()),
            int,
            len(row_dict),
        )
        sparse_row.indices = np.fromiter(row_dict.keys(), int, len(row_dict))
        sparse_row.indptr = np.array([0, len(row_dict)])

        return sparse_row
