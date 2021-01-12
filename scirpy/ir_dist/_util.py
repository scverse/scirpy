from collections.abc import MutableMapping
import numpy as np
import pandas as pd
import itertools
from typing import Dict, Sequence, Tuple
import scipy.sparse as sp


class DoubleLookupNeighborFinder:
    def __init__(
        self,
        feature_table: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        feature_table
            A data frame with features in columns. Rows must be unique.
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
        self.lookups: Dict[str, Tuple[str, np.ndarray, dict]] = dict()

    def lookup(self, clonotype_id, lookup_table):
        """Get ids of neighboring clonotypes given a pair of lookup tables
        and a distance matrix"""
        distance_matrix_name, forward, reverse = self.lookups[lookup_table]
        distance_matrix = self.distance_matrices[distance_matrix_name]
        row = distance_matrix[forward[clonotype_id], :]
        # get column indices directly from sparse row
        return itertools.chain.from_iterable(
            (reverse[i], distance) for i, distance in zip(row.indices, row.data)  # type: ignore
        )

    def add_distance_matrix(
        self, name: str, distance_matrix: sp.csr_matrix, labels: Sequence
    ):
        """Add a distance matrix."""
        self.distance_matrices[name] = distance_matrix
        self.distance_matrix_labels[name] = {k: i for i, k in enumerate(labels)}

    def add_lookup_table(
        self, clonotype_feature: str, distance_matrix: str, *, name: str
    ):
        forward = self._build_forward_lookup_table(clonotype_feature, distance_matrix)
        reverse = self._build_reverse_lookup_table(clonotype_feature, distance_matrix)
        self.lookups[name] = (distance_matrix, forward, reverse)

    def _build_forward_lookup_table(
        self, clonotype_feature: str, distance_matrix: str
    ) -> np.ndarray:
        """Create a lookup array that maps each clonotype to the respective
        index in the feature distance matrix.
        """
        return np.array(
            [
                self.distance_matrix_labels[distance_matrix][k]
                for k in self.feature_table[clonotype_feature]
            ]
        )

    def _build_reverse_lookup_table(
        self, clonotype_feature: str, distance_matrix: str
    ) -> dict:
        """Create a reverse-lookup dict that maps each (numeric) index
        of a feature distance matric to a list of associated numeric clonotype indices."""
        tmp_reverse_lookup = dict()
        tmp_index_lookup = self.distance_matrix_labels[distance_matrix]
        for i, k in enumerate(self.feature_table[clonotype_feature]):
            try:
                tmp_reverse_lookup[tmp_index_lookup[k]].append(i)
            except KeyError:
                tmp_reverse_lookup[tmp_index_lookup[k]] = [i]

        return tmp_reverse_lookup


class SetDict(MutableMapping):
    """A dictionary that supports set operations"""

    def __init__(self, *args, **kwargs):
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
                    (k, max(self[k], other[k]))
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
