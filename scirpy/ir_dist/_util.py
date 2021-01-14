from collections.abc import MutableMapping
import numpy as np
import pandas as pd
import itertools
from typing import Dict, Sequence, Tuple, Iterable, Union
import scipy.sparse as sp
from scipy.sparse.csr import csr_matrix


class DoubleLookupNeighborFinder:
    def __init__(self, feature_table: pd.DataFrame, *, nan_dist: float = 0):
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
        nan_dist
            Distance between two "nan" labels. Currently, a label is
            considered "nan" if it is the literal string "nan". This might change
            in the future.
        """
        self.feature_table = feature_table
        self.nan_dist = nan_dist

        # n_feature x n_feature sparse, symmetric distance matrices
        self.distance_matrices: Dict[str, sp.csr_matrix] = dict()
        # mapping feature_label -> feature_index with len = n_feature
        self.distance_matrix_labels: Dict[str, dict] = dict()
        # tuples (dist_mat, forward, reverse)
        # dist_mat: name of associated distance matrix
        # forward: clonotype -> feature_index lookups
        # reverse: feature_index -> clonotype lookups
        self.lookups: Dict[str, Tuple[str, np.ndarray, dict]] = dict()

    def lookup(
        self,
        object_id: int,
        forward_lookup_table: str,
        reverse_lookup_table: Union[str, None] = None,
    ) -> Iterable[Tuple[int, int]]:
        """Get ids of neighboring objects from a lookup table.

        Performs the following lookup:

            clonotype_id -> dist_mat -> neighboring features -> neighboring object.

        "nan"s are not looked up via the distance matrix. Instead they have
        a special entry in the lookup tables and yield the distance predefined in
        `self.nan_dist`.

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
            if self.nan_dist != 0:
                # special case for nan. Either yield predefined distance, or
                # do not yield any entries if the predefined distance is 0.
                yield from zip(reverse[np.nan], itertools.repeat(self.nan_dist))
        else:
            # get distances from the distance matrix...
            row = distance_matrix[idx_in_dist_mat, :]
            # ... and get column indices directly from sparse row
            yield from itertools.chain.from_iterable(
                # if no entry found (e.g. because of cross-table lookup)
                # return nothing (-> empty iterator)
                zip(reverse.get(i, iter(())), itertools.repeat(distance))
                for i, distance in zip(row.indices, row.data)  # type: ignore
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
    ) -> dict:
        """Create a reverse-lookup dict that maps each (numeric) index
        of a feature distance matric to a list of associated numeric clonotype indices."""
        tmp_reverse_lookup = dict()
        tmp_index_lookup = self.distance_matrix_labels[distance_matrix]
        for i, k in enumerate(self.feature_table[feature_col]):
            try:
                tmp_reverse_lookup[tmp_index_lookup[k]].append(i)
            except KeyError:
                tmp_reverse_lookup[tmp_index_lookup[k]] = [i]

        return tmp_reverse_lookup


class SetDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """A dictionary that supports set operations.

        Values are combined as follows:
        * when using `&`, the max value is retained.
        * when using `|`, the min value is retained.

        Examples:
        ---------
        >>> SetDict(a=5, b=7) | SetDict(a=2, c=8)
        SetDict(a=2, b=7, c=8)
        >>> SetDict(a=5, b=7) & SetDict(a=2, c=8)
        SetDict(a=5)
        """
        self.store = dict(*args, **kwargs)
        # self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __or__(self, other):
        if isinstance(other, set):
            raise NotImplementedError(
                "Cannot combine SetDict and set using 'or' (wouldn't know how to handle the score)"
            )
        elif isinstance(other, SetDict):
            return SetDict(
                (
                    (k, min(self.get(k, np.inf), other.get(k, np.inf)))
                    for k in (set(self.store) | set(other.store))
                )
            )
        else:
            raise NotImplementedError("Operation implemented only for SetDict. ")

    def __ror__(self, other):
        return self.__or__(other)

    def __and__(self, other):
        if isinstance(other, set):
            return SetDict(((k, self.store[k]) for k in set(self.store) & other))
        elif isinstance(other, SetDict):
            return SetDict(
                (
                    (k, self[k] + other[k] - 1)
                    for k in (set(self.store) & set(other.store))
                )
            )
        else:
            raise NotImplementedError(
                "Operation implemented only for SetDict and set. "
            )

    def __rand__(self, other):
        return self.__and__(other)

    def __repr__(self) -> str:
        return self.store.__repr__()
