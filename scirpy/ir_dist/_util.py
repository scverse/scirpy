import collections.abc as cabc
import abc
from functools import reduce
from operator import add
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import indices
import pandas as pd
import itertools
from typing import Dict, Sequence, Tuple, Iterable, Union, Mapping
import scipy.sparse as sp
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from .._compat import Literal


def reduce_and(*args, chain_count=None):
    """Take maximum, ignore nans"""
    # TODO stay int and test!
    tmp_array = np.vstack(args).astype(float)
    tmp_array[tmp_array == 0] = np.inf
    if chain_count is not None:
        same_count_mask = np.sum(np.isnan(tmp_array), axis=0) == chain_count
    tmp_array = np.nanmax(tmp_array, axis=0)
    tmp_array[np.isinf(tmp_array)] = 0
    tmp_array.astype(np.uint8)
    if chain_count is not None:
        tmp_array = np.multiply(tmp_array, same_count_mask)
    return tmp_array


def reduce_or(*args, chain_count=None):
    """Take minimum, ignore 0s and nans"""
    tmp_array = np.vstack(args).astype(float)
    tmp_array[tmp_array == 0] = np.inf
    tmp_array = np.nanmin(tmp_array, axis=0)
    tmp_array[np.isinf(tmp_array)] = 0
    tmp_array.astype(np.uint8)
    return tmp_array


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
            return sp.coo_matrix((1, self.n_rows))
        else:
            # get distances from the distance matrix...
            row = distance_matrix[idx_in_dist_mat, :]

            # ... and get column indices directly from sparse row
            # sum concatenates coo matrices
            return sum(
                (
                    reverse.get(i, sp.coo_matrix((1, self.n_rows))) * multiplier
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
        dist_type: Literal["bool", "number"] = "number",
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
        dist_type: Literal["bool", "number"],
    ) -> Mapping[int, Union[coo_matrix, np.ndarray]]:
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

        # convert into coo matrices (numeric distances) or numpy boolean arrays.
        for k, v in tmp_reverse_lookup.items():
            # nan will also be a boolean mask
            if dist_type == "bool":
                tmp_array = np.zeros(shape=(1, self.n_rows), dtype=bool)
                tmp_array[0, v] = True
                tmp_reverse_lookup[k] = tmp_array
            else:
                tmp_reverse_lookup[k] = sp.coo_matrix(
                    (
                        np.ones(len(v), dtype=np.uint8),
                        (np.zeros(len(v), dtype=int), v),
                    ),
                    shape=(1, self.n_rows),
                )

        return tmp_reverse_lookup
