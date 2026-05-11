import abc
import itertools
import warnings
from collections.abc import Sequence
from typing import Literal

import joblib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy.sparse
import scipy.spatial
from Levenshtein import distance as levenshtein_dist
from scanpy import logging
from scipy.sparse import coo_matrix, csr_matrix

from scirpy.util import _doc_params, _get_usable_cpus, _parallelize_with_joblib, deprecated

_doc_params_parallel_distance_calculator = """\
n_jobs
    Number of jobs to use for the pairwise distance calculation, passed to
    :class:`joblib.Parallel`. If -1, use all CPUs (only for ParallelDistanceCalculators).
    Via the :class:`joblib.parallel_config` context manager, another backend (e.g. `dask`)
    can be selected.
block_size
    Deprecated. This is now set in `calc_dist_mat`.
"""


_doc_dist_mat = """\
Calculates the full pairwise distance matrix.

.. important::
  * Distances are offset by 1 to allow efficient use of sparse matrices
    (:math:`d' = d+1`).
  * That means, a `distance > cutoff` is represented as `0`, a `distance == 0`
    is represented as `1`, a `distance == 1` is represented as `2` and so on.
  * Only returns distances `<= cutoff`. Larger distances are eliminated
    from the sparse matrix.
  * Distances are non-negative.

"""


class DistanceCalculator(abc.ABC):
    """\
    Abstract base class for a :term:`CDR3`-sequence distance calculator.

    Parameters
    ----------
    cutoff:
        Distances > cutoff will be eliminated to make efficient use of sparse matrices.
        If None, the default cutoff shall be used.

    """

    #: The sparse matrix dtype. Defaults to uint8, constraining the max distance to 255.
    DTYPE = "uint8"

    def __init__(self, cutoff: int | None):
        if cutoff > 255:
            raise ValueError("Using a cutoff > 255 is not possible due to the `uint8` dtype used")
        self.cutoff = cutoff

    @_doc_params(dist_mat=_doc_dist_mat)
    @abc.abstractmethod
    def calc_dist_mat(self, seqs: Sequence[str], seqs2: Sequence[str] | None = None) -> csr_matrix:
        """\
        Calculate pairwise distance matrix of all sequences in `seqs` and `seqs2`.

        When `seqs2` is omitted, computes the pairwise distance of `seqs` against
        itself.

        {dist_mat}

        Parameters
        ----------
        seqs
            array containing CDR3 sequences. Must not contain duplicates.
        seqs2
            second array containing CDR3 sequences. Must not contain
            duplicates either.

        Returns
        -------
        Sparse pairwise distance matrix.
        """

    @staticmethod
    def squarify(triangular_matrix: csr_matrix) -> csr_matrix:
        """Mirror a triangular matrix at the diagonal to make it a square matrix.

        The input matrix *must* be upper triangular to begin with, otherwise
        the results will be incorrect. No guard rails!
        """
        assert triangular_matrix.shape[0] == triangular_matrix.shape[1], "needs to be square matrix"
        # The matrix is already upper diagonal. Use the transpose method, see
        # https://stackoverflow.com/a/58806735/2340703.
        return triangular_matrix + triangular_matrix.T - scipy.sparse.diags(triangular_matrix.diagonal())


@_doc_params(params=_doc_params_parallel_distance_calculator)
class ParallelDistanceCalculator(DistanceCalculator):
    """
    Abstract base class for a DistanceCalculator that computes distances in parallel.

    It does so in a blockwise fashion. The function computing distances
    for a single block needs to be overriden.

    Parameters
    ----------
    {params}
    """

    def __init__(
        self,
        cutoff: int,
        *,
        n_jobs: int = -1,
        block_size: int | None = None,
    ):
        super().__init__(cutoff)
        self.n_jobs = n_jobs
        if block_size is not None:
            warnings.warn(
                "The `block_size` parameter is now set in the `calc_dist_mat` function instead of the object level. It is ignored here.",
                category=FutureWarning,
            )

    @abc.abstractmethod
    def _compute_block(
        self,
        seqs1: Sequence[str],
        seqs2: Sequence[str] | None,
        origin: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute the distances for a block of the matrix

        Parameters
        ----------
        seqs1
            array containing sequences
        seqs2
            other array containing sequences. If `None` compute the square matrix
            of `seqs1` and iterator over the upper triangle including the diagonal only.
        origin
            row, col coordinates of the origin of the block.

        Returns
        -------
        List of (distance, row, col) tuples for all elements with distance != 0.
        row, col must be the coordinates in the final matrix (they can be derived using
        `origin`). Can't be a generator because this needs to be picklable.
        """

    @staticmethod
    def _block_iter(
        seqs1: Sequence[str],
        seqs2: Sequence[str] | None = None,
        block_size: int | None = 50,
    ) -> tuple[Sequence[str], Sequence[str] | None, tuple[int, int]]:
        """Iterate over sequences in blocks.

        Parameters
        ----------
        seqs1
            array containing (unique) sequences
        seqs2
            array containing other sequences. If `None` compute
            the square matrix of `seqs1` and iterate over the upper triangle (including
            the diagonal) only.
        block_size
            side length of a block (will have `block_size ** 2` elements.)

        Yields
        ------
        seqs1
            subset of length `block_size` of seqs1
        seqs2
            subset of length `block_size` of seqs2. If seqs2 is None, this will
            be `None` if the block is on the diagonal, or a subset of seqs1 otherwise.
        origin
            (row, col) coordinates of the origin of the block.
        """
        square_mat = seqs2 is None
        if square_mat:
            seqs2 = seqs1
        for row in range(0, len(seqs1), block_size):
            start_col = row if square_mat else 0
            for col in range(start_col, len(seqs2), block_size):
                if row == col and square_mat:
                    # block on the diagonal.
                    # yield None for seqs2 to indicate that we only want the upper
                    # diagonal.
                    yield seqs1[row : row + block_size], None, (row, row)
                else:
                    yield seqs1[row : row + block_size], seqs2[col : col + block_size], (row, col)

    def calc_dist_mat(
        self, seqs: Sequence[str], seqs2: Sequence[str] | None = None, *, block_size: int | None = None
    ) -> csr_matrix:
        """Calculate the distance matrix.

        See :meth:`DistanceCalculator.calc_dist_mat`.

        Parameters
        ----------
        seqs
            array containing CDR3 sequences. Must not contain duplicates.
        seqs2
            second array containing CDR3 sequences. Must not contain
            duplicates either.
        block_size
            The width of a block that's sent to a worker. A block contains
            `block_size ** 2` elements. If `None` the block
            size is determined automatically based on the problem size.



        Returns
        -------
        Sparse pairwise distance matrix.
        """
        problem_size = len(seqs) * len(seqs2) if seqs2 is not None else len(seqs) ** 2
        # dynamicall adjust the block size such that there are ~1000 blocks within a range of 50 and 5000
        block_size = int(np.ceil(min(max(np.sqrt(problem_size / 1000), 50), 5000)))
        logging.info(f"block size set to {block_size}")

        # precompute blocks as list to have total number of blocks for progressbar
        blocks = list(self._block_iter(seqs, seqs2, block_size=block_size))

        block_results = _parallelize_with_joblib(
            (joblib.delayed(self._compute_block)(*block) for block in blocks), total=len(blocks), n_jobs=self.n_jobs
        )

        try:
            dists, rows, cols = zip(*itertools.chain(*block_results), strict=False)
        except ValueError:
            # happens when there is no match at all
            dists, rows, cols = (), (), ()

        shape = (len(seqs), len(seqs2)) if seqs2 is not None else (len(seqs), len(seqs))
        score_mat = scipy.sparse.coo_matrix((dists, (rows, cols)), dtype=self.DTYPE, shape=shape)
        score_mat.eliminate_zeros()
        score_mat = score_mat.tocsr()

        if seqs2 is None:
            score_mat = self.squarify(score_mat)

        return score_mat


class IdentityDistanceCalculator(DistanceCalculator):
    """\
    Calculates the Identity-distance between :term:`CDR3` sequences.

    The identity distance is defined as
        * `0`, if sequences are identical
        * `1`, if sequences are not identical.

    Choosing a cutoff:
        For this DistanceCalculator, per definition, the cutoff = 0.
        The `cutoff` argument is ignored.

    Parameters
    ----------
    cutoff
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices. For the IdentityDistanceCalculator this argument
        will be ignored and is always 0.
    """

    def __init__(self, cutoff: int = 0):
        cutoff = 0
        super().__init__(cutoff)

    def calc_dist_mat(self, seqs: np.ndarray, seqs2: np.ndarray = None) -> csr_matrix:
        """In this case, the offseted distance matrix is the identity matrix.

        More details: :meth:`DistanceCalculator.calc_dist_mat`
        """
        if seqs2 is None:
            # In this case, the offsetted distance matrix is the identity matrix
            return scipy.sparse.identity(len(seqs), dtype=self.DTYPE, format="csr")
        else:
            cols_by_seq = {}
            for i2, s2 in enumerate(seqs2):
                cols_by_seq.setdefault(s2, []).append(i2)

            row = []
            col = []
            for i1, s1 in enumerate(seqs):
                matching_cols = cols_by_seq.get(s1)
                if matching_cols is not None:
                    row.extend([i1] * len(matching_cols))
                    col.extend(matching_cols)

            d = np.ones(len(row), dtype=self.DTYPE)
            return coo_matrix((d, (row, col)), dtype=self.DTYPE, shape=(len(seqs), len(seqs2))).tocsr()


@_doc_params(params=_doc_params_parallel_distance_calculator)
class LevenshteinDistanceCalculator(ParallelDistanceCalculator):
    """\
    Calculates the Levenshtein edit-distance between sequences.

    The edit distance is the total number of deletion, addition and modification
    events.

    This class relies on `Python-levenshtein <https://github.com/ztane/python-Levenshtein>`_
    to calculate the distances.

    Choosing a cutoff:
        Each modification stands for a deletion, addition or modification event.
        While lacking empirical data, it seems unlikely that CDR3 sequences with more
        than two modifications still recognize the same antigen.

    Parameters
    ----------
    cutoff
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices. The default cutoff is `2`.
    {params}
    """

    def __init__(self, cutoff: int = 2, **kwargs):
        super().__init__(cutoff, **kwargs)

    def _compute_block(self, seqs1, seqs2, origin):
        origin_row, origin_col = origin
        if seqs2 is not None:
            # compute the full matrix
            coord_iterator = itertools.product(enumerate(seqs1), enumerate(seqs2))
        else:
            # compute only upper triangle in this case
            coord_iterator = itertools.combinations_with_replacement(enumerate(seqs1), r=2)

        result = []
        for (row, s1), (col, s2) in coord_iterator:
            d = levenshtein_dist(s1, s2)
            if d <= self.cutoff:
                result.append((d + 1, origin_row + row, origin_col + col))

        return result


def _substitution_to_distance_matrix(
    substitution_matrix: np.ndarray,
    alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX*",
    matrix_alphabet: str = "ARNDCQEGHILKMFPSTWYV",
    distance_cap: int | None = 4,
    distance_offset: int = 4,
) -> np.ndarray:
    """Creates a numba compatible distance matrix from a substitution matrix.

    Parameters
    ----------
    substitution_matrix:
        Amino-acid substitution matrix in the order specified by `matrix_alphabet`.
    distance_cap:
        Maximum distance assigned to a mismatch. If `None`, mismatch distances are uncapped.
    distance_offset:
        Offset from which the substitution score is subtracted.

    Returns
    -------
    distance_matrix:
        distance lookup matrix
    """
    dm = np.zeros((len(alphabet), len(alphabet)), dtype=np.int32)
    if substitution_matrix.shape != (len(matrix_alphabet), len(matrix_alphabet)):
        raise ValueError("`substitution_matrix` must be square and match `matrix_alphabet`.")
    for i, aa1 in enumerate(matrix_alphabet):
        for j, aa2 in enumerate(matrix_alphabet):
            d = 0 if aa1 == aa2 else distance_offset - substitution_matrix[i, j]
            if distance_cap is not None:
                d = min(distance_cap, d)
            dm[alphabet.index(aa1), alphabet.index(aa2)] = d
    return dm


def _seqs2mat(
    seqs: Sequence[str], alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX", max_len: None | int = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a collection of gene sequences into a
    numpy matrix of integers for fast comparison.

    Parameters
    ----------
    seqs:
        Sequence of strings

    Returns
    -------
    mat:
        matrix with gene sequences encoded as integers
    L:
        vector with length values of the gene sequences in the matrix

    Examples
    --------
    >>> seqs2mat(["CAT", "HAT"])
    array([[ 4,  0, 16],
        [ 8,  0, 16]], dtype=int8)

    Notes
    -----
    Requires all seqs to have the same length, therefore shorter sequences
    are filled up with -1 entries at the end.
    """
    if max_len is None:
        max_len = np.max([len(s) for s in seqs])
    mat = -1 * np.ones((len(seqs), max_len), dtype=np.int8)
    L = np.zeros(len(seqs), dtype=np.int8 if max_len <= np.iinfo(np.int8).max else np.int16)
    for si, s in enumerate(seqs):
        L[si] = min(len(s), max_len)
        for aai in range(max_len):
            if aai >= len(s):
                break
            try:
                mat[si, aai] = alphabet.index(s[aai])
            except ValueError:
                # Unknown symbols given value for last column/row of matrix
                mat[si, aai] = len(alphabet)
    return mat, L


class _MetricDistanceCalculator(abc.ABC):
    """
    Abstract base class for distance calculator classes that computes parwise distances between
    gene sequences in parallel based on a certain distance metric.

    The result is a (scipy) compressed sparse row distance matrix.
    Derived classes just need to implement the method _metric_mat (see method comments for more details).

    The code of this class is based on `pwseqdist <https://github.com/agartland/pwseqdist/blob/master/pwseqdist>`_.
    Reused under MIT license, Copyright (c) 2020 Andrew Fiore-Gartland.

    Parameters
    ----------
    n_jobs:
        Number of threads per process to use for the pairwise distance calculation
    n_blocks:
        Overall number of blocks given to the workers (processes)
    histogram:
        Determines whether a nearest neighbor histogram should be created
    """

    def __init__(self, n_jobs: int = -1, n_blocks: int = 1, histogram: bool = False):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_blocks = n_blocks
        self.histogram = histogram

    @abc.abstractmethod
    def _metric_mat(
        self,
        *,
        seqs: Sequence[str],
        seqs2: Sequence[str],
        is_symmetric: bool = False,
        start_column: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Abstract method that should be implemented by the derived class in a way such that it computes the pairwise distances
        for gene sequences in seqs and seqs2 based on a certain distance metric. The result should be a distance matrix
        that is returned in the form of the data, indices and intptr arrays of a (scipy) compressed sparse row matrix.

        In case that a nearest neighbour histogram should be created later, the minimum value per row is returned.

        If this function is used to compute a block of a bigger result matrix, is_symmetric and start_column
        can be used to only compute the part of the block that would be part of the upper triangular matrix of the
        result matrix.

        Parameters
        ----------
        seqs/2:
            A python sequence of strings representing gene sequences
        is_symmetric:
            Determines whether the final result matrix is symmetric, assuming that this function is
            only used to compute a block of a bigger result matrix
        start_column:
            Determines at which column the calculation should be started. This is only used if this function is
            used to compute a block of a bigger result matrix that is symmetric

        Returns
        -------
        data_rows:
            List with arrays containing the non-zero data values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        indices_rows:
            List with arrays containing the non-zero entry column indeces of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_element_counts:
            Array with integers that indicate the amount of non-zero values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_mins:
            Array containing the minimum distance per row, ignoring equal sequences and ignoring the cutoff.
            Contains None if the computation of row_mins is not implemented. Used to create a nearest neighbor
            histogram later. Is empty if the histogram should not be created.
        """
        pass

    def _make_histogram(self, row_mins: np.ndarray):
        """Subclass should override this method if the creation of a nearest neighbor histogram is implemented."""
        raise NotImplementedError("Creating a histogram is not implemented for this metric")

    def _calc_dist_mat_block(
        self,
        seqs: Sequence[str],
        seqs2: Sequence[str],
        is_symmetric: bool = False,
        start_column: int = 0,
    ) -> tuple[csr_matrix, np.ndarray]:
        """Computes a block of the final distance matrix and returns it as CSR matrix. Also computes
        the minimum distance per row, for which equal sequences and the cutoff are ignored.
        If the final result matrix that consists of all blocks together is symmetric, only the part
        of the block that would contribute to the upper triangular matrix of the final result will be computed.
        """
        if len(seqs) == 0 or len(seqs2) == 0:
            return csr_matrix((len(seqs), len(seqs2))), np.array([None])

        data_rows, indices_rows, row_element_counts, row_mins = self._metric_mat(
            seqs=seqs,
            seqs2=seqs2,
            is_symmetric=is_symmetric,
            start_column=start_column,
        )

        indptr = np.zeros(row_element_counts.shape[0] + 1)
        indptr[1:] = np.cumsum(row_element_counts)
        data, indices = np.concatenate(data_rows), np.concatenate(indices_rows)
        sparse_distance_matrix = csr_matrix((data, indices, indptr), shape=(len(seqs), len(seqs2)))
        return sparse_distance_matrix, row_mins

    def calc_dist_mat(self, seqs: Sequence[str], seqs2: Sequence[str] | None = None) -> csr_matrix:
        """Calculates the pairwise distances between two vectors of gene sequences based on the distance metric
        of the derived class and returns a CSR distance matrix. Also creates a histogram based on the minimum value
        per row of the distance matrix if histogram is set to True.
        """
        if seqs2 is None:
            seqs2 = seqs

        seqs = np.array(seqs)
        seqs2 = np.array(seqs2)
        is_symmetric = np.array_equal(seqs, seqs2)

        if self.n_blocks > 1:
            split_seqs = np.array_split(seqs, self.n_blocks)
            start_columns = np.cumsum([0] + [len(seq) for seq in split_seqs[:-1]])
            arguments = [(split_seqs[x], seqs2, is_symmetric, start_columns[x]) for x in range(self.n_blocks)]

            delayed_jobs = [joblib.delayed(self._calc_dist_mat_block)(*args) for args in arguments]
            results = joblib.Parallel(return_as="list")(delayed_jobs)

            block_matrices_csr, block_row_mins = zip(*results, strict=False)
            distance_matrix_csr = scipy.sparse.vstack(block_matrices_csr)
            row_mins = np.concatenate(block_row_mins)
        else:
            distance_matrix_csr, row_mins = self._calc_dist_mat_block(seqs, seqs2, is_symmetric)

        if is_symmetric:
            upper_triangular_distance_matrix = distance_matrix_csr
            full_distance_matrix = upper_triangular_distance_matrix.maximum(upper_triangular_distance_matrix.T)
        else:
            full_distance_matrix = distance_matrix_csr

        if self.histogram:
            self._make_histogram(row_mins)

        return full_distance_matrix


class HammingDistanceCalculator(_MetricDistanceCalculator):
    """Computes pairwise distances between gene sequences based on the "hamming" distance metric.

    Set `normalize` to True to use the normalized hamming distance metric instead of the standard hamming distance
    metric. Then the distance will be calculated as percentage of different positions relative to the sequence length
    (e.g. AAGG and AAAA -> 50 (%) normalized hamming distance). The cutoff is then also given as normalized hamming
    distance in percent.

    The code of this class is based on `pwseqdist <https://github.com/agartland/pwseqdist/blob/master/pwseqdist>`_.
    Reused under MIT license, Copyright (c) 2020 Andrew Fiore-Gartland.

    Parameters
    ----------
    cutoff:
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices.
    n_jobs:
        Number of numba parallel threads to use for the pairwise distance calculation
    n_blocks:
        Number of joblib delayed objects (blocks to compute) given to joblib.Parallel
    normalize:
        Determines whether the normalized hamming distance metric should be used instead of the standard
        hamming distance
    histogram:
        Determines whether a nearest neighbor histogram should be created
    """

    def __init__(
        self,
        n_jobs: int = -1,
        n_blocks: int = 1,
        cutoff: int = 2,
        *,
        normalize: bool = False,
        histogram: bool = False,
    ):
        super().__init__(n_jobs=n_jobs, n_blocks=n_blocks, histogram=histogram)
        self.cutoff = cutoff
        self.normalize = normalize

    def _make_histogram(self, row_mins: np.ndarray):
        if self.normalize:
            bins = np.arange(0, 101, 2)
        else:
            max_value = np.max(row_mins)
            bin_step = np.ceil(max_value / 100)
            bins = np.arange(0, max_value + 1, bin_step)

        plt.hist(row_mins, bins=bins, histtype="bar", edgecolor="black")
        plt.axvline(x=self.cutoff, color="r", linestyle="-", label="cutoff")
        plt.legend()
        plt.xlabel("Distance to nearest neighbor")
        plt.ylabel("Count")
        plt.title('Histogram of "distance-to-nearest"-distribution')
        plt.show()

    def _hamming_mat(
        self,
        *,
        seqs: Sequence[str],
        seqs2: Sequence[str],
        is_symmetric: bool = False,
        start_column: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """Computes the pairwise hamming distances for sequences in seqs and seqs2.

        This function is a wrapper and contains an inner JIT compiled numba function without parameters. The reason for this is
        that this way some of the parameters can be treated as constant by numba and this allows for a better optimization
        of the numba compiler in this specific case.

        If this function is used to compute a block of a bigger result matrix, is_symmetric and start_column
        can be used to only compute the part of the block that would be part of the upper triangular matrix of the
        result matrix.

        Parameters
        ----------
        seqs/2:
            A python sequence of strings representing gene sequences
        is_symmetric:
            Determines whether the final result matrix is symmetric, assuming that this function is
            only used to compute a block of a bigger result matrix
        start_column:
            Determines at which column the calculation should be started. This is only used if this function is
            used to compute a block of a bigger result matrix that is symmetric

        Returns
        -------
        data_rows:
            List with arrays containing the non-zero data values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        indices_rows:
            List with arrays containing the non-zero entry column indeces of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_element_counts:
            Array with integers that indicate the amount of non-zero values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_mins:
            Array containing the minimum distance per row, ignoring equal sequences and ignoring the cutoff.
            Used to create a nearest neighbor histogram later. Is empty if the histogram should not be created.
        """
        unique_characters = "".join({char for string in (*seqs, *seqs2) for char in string})
        max_seq_len = max(len(s) for s in (*seqs, *seqs2))

        seqs_mat1, seqs_L1 = _seqs2mat(seqs, alphabet=unique_characters, max_len=max_seq_len)
        seqs_mat2, seqs_L2 = _seqs2mat(seqs2, alphabet=unique_characters, max_len=max_seq_len)

        cutoff = self.cutoff
        normalize = self.normalize
        histogram = self.histogram

        if histogram:
            is_symmetric = False

        start_column *= is_symmetric

        nb.set_num_threads(_get_usable_cpus(n_jobs=self.n_jobs, use_numba=True))

        num_threads = nb.get_num_threads()

        jit_parallel = num_threads > 1

        @nb.jit(nopython=True, parallel=jit_parallel, nogil=True)
        def _nb_hamming_mat():
            assert seqs_mat1.shape[0] == seqs_L1.shape[0]
            assert seqs_mat2.shape[0] == seqs_L2.shape[0]

            num_rows = seqs_mat1.shape[0]
            num_cols = seqs_mat2.shape[0]

            data_rows = nb.typed.List()
            indices_rows = nb.typed.List()
            row_element_counts = np.zeros(num_rows)

            if histogram:
                row_mins = np.zeros(num_rows)
            else:
                row_mins = np.zeros(0)

            empty_row = np.zeros(0)
            for _ in range(0, num_rows):
                data_rows.append([empty_row])
                indices_rows.append([empty_row])

            data_row_matrix = np.empty((num_threads, num_cols))
            indices_row_matrix = np.empty((num_threads, num_cols))

            for row_index in nb.prange(num_rows):
                thread_id = nb.get_thread_id()
                row_end_index = 0
                seq1_len = seqs_L1[row_index]

                if histogram:
                    if normalize:
                        row_min = 100
                    else:
                        row_min = seq1_len

                for col_index in range(start_column + row_index * is_symmetric, num_cols):
                    distance = 1
                    seq2_len = seqs_L2[col_index]
                    if seq1_len == seq2_len:
                        for i in range(0, seq1_len):
                            distance += seqs_mat1[row_index, i] != seqs_mat2[col_index, i]

                        if normalize:
                            distance = int((distance - 1) * 100 / seq1_len + 0.5) + 1

                        if distance <= cutoff + 1:
                            data_row_matrix[thread_id, row_end_index] = distance
                            indices_row_matrix[thread_id, row_end_index] = col_index
                            row_end_index += 1

                        if histogram:
                            if distance > 1:
                                row_min = min(row_min, distance - 1)

                data_rows[row_index][0] = data_row_matrix[thread_id, 0:row_end_index].copy()
                indices_rows[row_index][0] = indices_row_matrix[thread_id, 0:row_end_index].copy()
                row_element_counts[row_index] = row_end_index
                if histogram:
                    row_mins[row_index] = row_min

            data_rows_flat = []
            indices_rows_flat = []

            for i in range(len(data_rows)):
                data_rows_flat.append(data_rows[i][0])
                indices_rows_flat.append(indices_rows[i][0])

            return data_rows_flat, indices_rows_flat, row_element_counts, row_mins

        data_rows, indices_rows, row_element_counts, row_mins = _nb_hamming_mat()

        return data_rows, indices_rows, row_element_counts, row_mins

    _metric_mat = _hamming_mat


class GPUHammingDistanceCalculator(_MetricDistanceCalculator):
    """Computes pairwise distances between gene sequences based on the "hamming" distance metric with GPU support.

    The code of this class is based on `pwseqdist <https://github.com/agartland/pwseqdist/blob/master/pwseqdist>`_.
    Reused under MIT license, Copyright (c) 2020 Andrew Fiore-Gartland.

    For performance reasons, the computation of the final result matrix is split up into several blocks. The parameter
    gpu_n_blocks determines the number of those blocks. The parameter gpu_block_width determines how much GPU memory
    is reserved for the computed result of each block in SPARSE representation.

    E.g. there is a 1000x1000 (dense represenation) not yet computed result matrix with gpu_n_blocks=10 and gpu_block_width=20.
    Then the result matrix is computed in 10 blocks of  1000x100 (dense representation). Each of these blocks needs to fit into
    a 1000x20 block in SPARSE representation once computed and this 1000x20 block needs to fit into GPU memory. So there shouldn't
    be a resulting row in a block that has more than 20 values <= cutoff.

    The parameter gpu_block_width should be chosen based on the available GPU memory. Choosing lower values for gpu_n_blocks increases
    the performance but also increases the risk of running out of reserved memory, since the result blocks that need to fit into the
    reserved GPU memory in sparse representation get bigger.

    Parameters
    ----------
    cutoff:
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices.
    gpu_n_blocks:
        Number of blocks in which the final result matrix should be computed. Each block reserves GPU memory
        in which the computed result block has to fit in sparse representation. Lower values give better performance
        but increase the risk of running out of reserved memory. This value should be chosen based on the
        estimated sparsity of the result matrix and the size of the GPU device memory.
    gpu_block_width:
        Maximum width of blocks in which the final result matrix should be computed. Each block reserves GPU memory
        in which the computed result block has to fit in sparse representation. Higher values allow for a lower
        number of result blocks (gpu_n_blocks) which increases the performance. This value should be chosen based on
        the GPU device memory.
    """

    def __init__(
        self,
        *,
        cutoff: int = 2,
        gpu_n_blocks: int = 10,
        gpu_block_width: int = 1000,
    ):
        super().__init__(n_jobs=1, n_blocks=1)
        self.cutoff = cutoff
        self.gpu_n_blocks = gpu_n_blocks
        self.gpu_block_width = gpu_block_width

    def _gpu_hamming_mat(
        self,
        *,
        seqs: Sequence[str],
        seqs2: Sequence[str],
        is_symmetric: bool = False,
        start_column: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Computes the pairwise hamming distances for sequences in seqs and seqs2 with GPU support.

        Parameters
        ----------
        seqs/2:
            A python sequence of strings representing gene sequences
        is_symmetric:
            Determines whether the final result matrix is symmetric, assuming that this function is
            only used to compute a block of a bigger result matrix
        start_column:
            Determines at which column the calculation should be started. This is only used if this function is
            used to compute a block of a bigger result matrix that is symmetric

        Returns
        -------
        data_rows:
            List with array containing the non-zero data values of the result matrix,
            needed to create the final scipy CSR result matrix later
        indices_rows:
            List with array containing the non-zero entry column indeces of the result matrix,
            needed to create the final scipy CSR result matrix later
        row_element_counts:
            Array with integers that indicate the amount of non-zero values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_mins:
            Always returns a numpy array containing None because the computation of the minimum distance per row is
            not implemented for the GPU hamming calculator yet.
        """
        import cupy as cp
        from tqdm import tqdm

        seqs_lengths = np.vectorize(len)(seqs)
        seqs_original_indices = np.argsort(seqs_lengths)
        seqs = seqs[seqs_original_indices]

        seqs2_lengths = np.vectorize(len)(seqs2)
        seqs2_original_indices = np.argsort(seqs2_lengths)
        seqs2 = seqs2[seqs2_original_indices]

        seqs_original_indices = cp.asarray(seqs_original_indices, dtype=np.int32)
        seqs2_original_indices = cp.asarray(seqs2_original_indices, dtype=np.int32)

        is_symmetric = False

        max_seq_len = max(len(s) for s in (*seqs, *seqs2))

        def _seqs2mat_fast(seqs: Sequence[str], max_len: None | int = None) -> tuple[np.ndarray, np.ndarray]:
            if max_len is None:
                max_len = np.max([len(s) for s in seqs])
            mat = -1 * np.ones((len(seqs), max_len), dtype=np.int8)
            L = np.zeros(len(seqs), dtype=np.int8 if max_len <= np.iinfo(np.int8).max else np.int16)
            for i, seq in enumerate(seqs):
                mat[i][0 : len(seq)] = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
                L[i] = len(seq)
            return mat, L

        try:
            seqs_mat1, seqs_L1 = _seqs2mat_fast(seqs, max_len=max_seq_len)
            seqs_mat2, seqs_L2 = _seqs2mat_fast(seqs2, max_len=max_seq_len)
        except UnicodeError:
            logging.info(
                "UnicodeError error occurred while converting sequences, retrying with implementation for non ascii sequences"
            )
            unique_characters = "".join(sorted({char for string in (*seqs, *seqs2) for char in string}))
            seqs_mat1, seqs_L1 = _seqs2mat(seqs, alphabet=unique_characters, max_len=max_seq_len)
            seqs_mat2, seqs_L2 = _seqs2mat(seqs2, alphabet=unique_characters, max_len=max_seq_len)

        hamming_kernel = cp.RawKernel(
            r"""
        extern "C" __global__ __launch_bounds__(256)
        void hamming_kernel(
            const char* __restrict__ seqs_mat1,
            const char* __restrict__ seqs_mat2,
            const int* __restrict__ seqs_L1,
            const int* seqs_L2,
            const int* __restrict__ seqs_original_indices,
            const int* seqs2_original_indices,
            const int cutoff,
            char* __restrict__ data,
            int* __restrict__ indices,
            int* __restrict__ row_element_counts,
            const int block_offset,
            const int seqs_mat1_rows,
            const int seqs_mat2_rows,
            const int seqs_mat1_cols,
            const int seqs_mat2_cols,
            const int data_cols,
            const int indices_cols,
            const bool is_symmetric
        ) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            if (row < seqs_mat1_rows) {
                int seqs_original_index = seqs_original_indices[row];
                int seq1_len = seqs_L1[row];
                int row_end_index = 0;

                for (int col = 0; col < seqs_mat2_rows; col++) {
                    if ((! is_symmetric ) || (col + block_offset) >= row) {
                        int seq2_len = seqs_L2[col];
                        char distance = 1;

                        if (seq1_len == seq2_len) {
                            for (int i = 0; i < seq1_len; i++) {
                                char val1 = seqs_mat1[i*seqs_mat1_rows+row];
                                char val2 = seqs_mat2[i*seqs_mat2_rows+col];

                                if(val1 != val2) {
                                    distance++;
                                }
                            }
                            if (distance <= cutoff + 1) {
                                int seqs2_original_index = seqs2_original_indices[col];
                                data[seqs_original_index * data_cols + row_end_index] = distance;
                                indices[seqs_original_index * indices_cols + row_end_index] = seqs2_original_index;
                                row_end_index++;
                            }
                        }
                    }
                }
                row_element_counts[seqs_original_index] = row_end_index;
            }
        }
        """,
            "hamming_kernel",
            options=("--maxrregcount=256",),
        )

        create_csr_kernel = cp.RawKernel(
            r"""
        extern "C" __global__
        void create_csr_kernel(
            int* data, int* indices,
            char* data_matrix, int* indices_matrix,
            int* indptr, int data_matrix_rows, int data_matrix_cols, int data_rows, int indices_matrix_cols
        ) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            int col = blockDim.y * blockIdx.y + threadIdx.y;

            if (row < data_matrix_rows && col < data_matrix_cols) {
                int row_start = indptr[row];
                int row_end = indptr[row + 1];
                int row_end_index = row_end - row_start;
                int data_index = row_start + col;

                if ((data_index < data_rows) && (col < row_end_index)) {
                    data[data_index] = data_matrix[row * data_matrix_cols + col];
                    indices[data_index] = indices_matrix[row * indices_matrix_cols + col];
                }
            }
        }
        """,
            "create_csr_kernel",
        )

        def calc_block_gpu(
            seqs_mat1, seqs_mat2_block, seqs_L1_block, seqs_L2, seqs2_original_indices_blocks, block_offset
        ):
            import cupy as cp

            d_seqs_mat1 = cp.asarray(seqs_mat1.astype(np.int8))
            d_seqs_mat2 = cp.asarray(seqs_mat2_block.astype(np.int8))
            d_seqs_L1 = cp.asarray(seqs_L1_block.astype(np.int32))
            d_seqs_L2 = cp.asarray(seqs_L2.astype(np.int32))

            # Due to performance reasons and since we expect the result matrix to be very sparse, we
            # set a maximum result width for the current block
            max_block_width = self.gpu_block_width

            d_data_matrix = cp.empty((seqs_mat1.shape[0], max_block_width), dtype=cp.int8)
            d_indices_matrix = cp.empty((seqs_mat1.shape[0], max_block_width), dtype=np.int32)
            d_row_element_counts = cp.zeros(seqs_mat1.shape[0], dtype=np.int32)

            threads_per_block = 256
            blocks_per_grid = (seqs_mat1.shape[0] + (threads_per_block - 1)) // threads_per_block

            seqs_mat1_rows, seqs_mat1_cols = seqs_mat1.shape
            seqs_mat2_rows, seqs_mat2_cols = seqs_mat2_block.shape
            d_data_matrix_cols = max_block_width
            d_indices_matrix_cols = max_block_width

            d_seqs_mat1_transposed = cp.transpose(d_seqs_mat1).copy()
            d_seqs_mat2_transposed = cp.transpose(d_seqs_mat2).copy()

            hamming_kernel(
                (blocks_per_grid,),
                (threads_per_block,),
                (
                    d_seqs_mat1_transposed,
                    d_seqs_mat2_transposed,
                    d_seqs_L1,
                    d_seqs_L2,
                    seqs_original_indices,
                    seqs2_original_indices_blocks,
                    self.cutoff,
                    d_data_matrix,
                    d_indices_matrix,
                    d_row_element_counts,
                    block_offset,
                    seqs_mat1_rows,
                    seqs_mat2_rows,
                    seqs_mat1_cols,
                    seqs_mat2_cols,
                    d_data_matrix_cols,
                    d_indices_matrix_cols,
                    is_symmetric,
                ),
            )

            row_element_counts = d_row_element_counts.get()
            row_max_len = np.max(row_element_counts)
            row_element_sum = np.sum(row_element_counts, dtype=np.int64)

            assert (
                row_max_len <= max_block_width
            ), f"""ERROR: The chosen result block width is too small to hold all result values of the current block.
            Chosen width: {max_block_width}, Necessary width: {row_max_len}."""

            assert (
                row_element_sum <= np.iinfo(np.int32).max
            ), f"""ERROR: There are too many result values to be held by the resulting CSR matrix of the current block.
            Current number: {row_element_sum}, Maximum number: {np.iinfo(np.int32).max}.
            Consider choosing a smaller cutoff to resolve this issue."""

            indptr = np.zeros(seqs_mat1.shape[0] + 1, dtype=np.int32)
            indptr[1:] = np.cumsum(row_element_counts)
            d_indptr = cp.asarray(indptr)

            n_elements = indptr[-1]
            data = np.zeros(n_elements, dtype=np.int32)
            d_data = cp.zeros_like(data)

            indices = np.zeros(n_elements, dtype=np.int32)
            d_indices = cp.zeros_like(indices)

            threads_per_block = (1, 256)
            blocks_per_grid_x = (d_data_matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (d_data_matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

            create_csr_kernel(
                (blocks_per_grid_x, blocks_per_grid_y),
                threads_per_block,
                (
                    d_data,
                    d_indices,
                    d_data_matrix,
                    d_indices_matrix,
                    d_indptr,
                    d_data_matrix.shape[0],
                    d_data_matrix.shape[1],
                    d_data.shape[0],
                    d_indices_matrix.shape[1],
                ),
            )

            data = d_data.get()
            indptr = d_indptr.get()
            indices = d_indices.get()

            res = csr_matrix((data, indices, indptr), shape=(seqs_mat1.shape[0], seqs_mat2.shape[0]))
            return res

        # Set the number of blocks for the calculation. A higher number can be more memory friendly, whereas
        # a lower number can improve the performance.
        n_blocks = self.gpu_n_blocks

        seqs_mat2_blocks = np.array_split(seqs_mat2, n_blocks)
        seqs_L2_blocks = np.array_split(seqs_L2, n_blocks)
        seqs2_original_indices_blocks = np.array_split(seqs2_original_indices, n_blocks)
        result_blocks = [None] * n_blocks

        block_offset = start_column

        logging.info(
            f"\nStart GPU calculations for {n_blocks} sparse matrix result blocks of max width {self.gpu_block_width}:"
        )

        for i in tqdm(range(0, n_blocks), desc="Processing", unit="block"):
            result_blocks[i] = calc_block_gpu(
                seqs_mat1,
                seqs_mat2_blocks[i],
                seqs_L1,
                seqs_L2_blocks[i],
                seqs2_original_indices_blocks[i],
                block_offset,
            )
            block_offset += seqs_mat2_blocks[i].shape[0]

        num_elements = 0
        for i in range(0, len(result_blocks)):
            num_elements += result_blocks[i].indptr[-1]

        assert (
            num_elements <= np.iinfo(np.int32).max
        ), f"""ERROR: The overall number of result values is too high to construct the final CSR matrix by combining
        the already calculated blocks.
        Current number: {num_elements}, Maximum number: {np.iinfo(np.int32).max}.
        Consider choosing a smaller cutoff to resolve this issue."""

        @nb.njit
        def csr_union_numba(block_data, block_indices, block_indptrs, num_rows, num_elements):
            data = np.empty(num_elements, dtype=block_data[0].dtype)
            indices = np.empty(num_elements, dtype=block_indices[0].dtype)
            indptr = np.zeros(num_rows + 1, dtype=np.int32)

            ptr = 0
            for row in range(num_rows):
                for b in range(len(block_indptrs)):
                    start = block_indptrs[b][row]
                    end = block_indptrs[b][row + 1]
                    count = end - start

                    for j in range(count):
                        data[ptr + j] = block_data[b][start + j]
                        indices[ptr + j] = block_indices[b][start + j]

                    ptr += count
                indptr[row + 1] = ptr

            return data, indices, indptr

        def csr_union(blocks):
            num_rows = blocks[0].shape[0]
            num_elements = sum(b.nnz for b in blocks)

            block_data = [b.data for b in blocks]
            block_indices = [b.indices for b in blocks]
            block_indptrs = [b.indptr for b in blocks]

            data, indices, indptr = csr_union_numba(block_data, block_indices, block_indptrs, num_rows, num_elements)

            shape = blocks[0].shape
            return csr_matrix((data, indices, indptr), shape=shape)

        result_sparse = csr_union(result_blocks)

        row_element_counts_gpu = np.diff(result_sparse.indptr)
        result_sparse.sort_indices()

        # Returns the results in a way that fits the current interface, could be improved later
        return [result_sparse.data], [result_sparse.indices], row_element_counts_gpu, np.array([None])

    _metric_mat = _gpu_hamming_mat


class TCRdistDistanceCalculator(_MetricDistanceCalculator):
    """Computes pairwise distances between TCR CDR3 sequences based on the "tcrdist" distance metric.

    The code of this class is heavily based on `pwseqdist <https://github.com/agartland/pwseqdist/blob/master/pwseqdist>`_.
    Reused under MIT license, Copyright (c) 2020 Andrew Fiore-Gartland.

    Using default weight, gap penalty, ntrim and ctrim is equivalent to the
    original distance published in :cite:`TCRdist`.

    Parameters
    ----------
    dist_weight:
        Weight applied to the mismatch distances before summing with the gap penalties
    gap_penalty:
        Distance penalty for the difference in the length of the two sequences
    ntrim/ctrim:
        Positions trimmed off the N-terminus (0) and C-terminus (L-1) ends of the peptide sequence. These symbols will be ignored
        in the distance calculation.
    fixed_gappos:
        If True, insert gaps at a fixed position after the cysteine residue statring the CDR3 (typically position 6).
        If False, find the "optimal" position for inserting the gaps to make up the difference in length
    cutoff:
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices.
    n_jobs:
        Number of numba parallel threads to use for the pairwise distance calculation
    n_blocks:
        Number of joblib delayed objects (blocks to compute) given to joblib.Parallel
    histogram:
        Determines whether a nearest neighbor histogram should be created
    base_matrix:
        Amino acid substitution matrix used by TCRdist. `"blosum62"` uses the original
        BLOSUM62 substitution matrix, while `"tcrblosum"` uses TCRBLOSUM substitution
        matrices (:cite:`TCRBLOSUM`). Depending on `chain_type`, either the TCRBLOSUM
        alpha- or beta-chain matrix is used.
    distance_cap:
        Maximum distance assigned to a mismatch after converting substitution scores to distances.
        The default value, `"default"`, keeps the original behavior: BLOSUM62 uses a cap of `4`,
        while TCRBLOSUM distances are uncapped. Set to an integer to choose a cap explicitly, or
        `None` for uncapped distances.
    chain_type:
        Required when `base_matrix="tcrblosum"`. `"VJ"` selects the alpha-chain matrix
        and `"VDJ"` selects the beta-chain matrix. When called via `ir_dist`, this value
        is set automatically and should not be provided.
    """

    parasail_aa_alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
    parasail_aa_alphabet_with_unknown = "ARNDCQEGHILKMFPSTWYVBZX*"
    # fmt: off
    matrix_alphabet = "ARNDCQEGHILKMFPSTWYV"
    blosum62_substitution_matrix = np.array(
        [
            # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
            [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
            [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
            [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
            [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
            [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
            [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
            [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
            [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
            [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
            [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
            [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
            [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
            [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
            [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
            [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
            [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
            [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
            [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
        ],
        dtype=np.int32,
    )
    tcrblosum_alpha_substitution_matrix = np.array(
        [
            # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
            [ 2, -1, -1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1,  0,  0, -1,  0, -1,  0],  # A
            [-1,  1,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # R
            [-1,  0,  1,  0,  0,  0,  0,  0,  0, -1, -2,  1,  0,  0,  0,  0,  0,  0,  0, -2],  # N
            [-1,  0,  0,  1, -5,  0,  0,  0,  0, -1, -2,  0,  0,  0,  0,  0,  0,  0,  0, -1],  # D
            [ 0,  1,  0, -5,  2, -4, -4,  0, -2, -5,  0, -5, -4, -4, -4,  0, -6, -2, -5,  0],  # C
            [ 0,  0,  0,  0, -4,  2,  0,  0,  0, -1, -2,  1,  0,  0,  0,  0,  0,  0,  0, -2],  # Q
            [ 0,  0,  0,  0, -4,  0,  1,  0,  1, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # E
            [ 0,  0,  0,  0,  0,  0,  0,  1,  0, -2, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # G
            [ 0,  0,  0,  0, -2,  0,  1,  0,  2,  0,  0, -1,  0,  0,  1,  0,  0,  1,  0,  0],  # H
            [-1,  0, -1, -1, -5, -1, -1, -2,  0,  3,  0, -1,  0,  0,  0,  0,  1, -1,  0,  0],  # I
            [-1, -1, -2, -2,  0, -2,  0, -1,  0,  0,  2, -4,  0,  1,  0, -1, -1, -1,  0,  0],  # L
            [-1,  0,  1,  0, -5,  1, -1, -1, -1, -1, -4,  3,  0, -3,  0, -2, -1, -2, -4, -3],  # K
            [-1,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0],  # M
            [-1,  0,  0,  0, -4,  0,  0,  0,  0,  0,  1, -3,  0,  1,  0,  0,  0,  0,  0,  0],  # F
            [ 0,  0,  0,  0, -4,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # P
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -2,  0,  0,  0,  1,  0,  0,  0, -1],  # S
            [-1,  0,  0,  0, -6,  0,  0,  0,  0,  1, -1, -1,  0,  0,  0,  0,  1,  0,  0,  0],  # T
            [ 0,  0,  0,  0, -2,  0,  0,  0,  1, -1, -1, -2,  0,  0,  0,  0,  0,  2,  0, -1],  # W
            [-1,  0,  0,  0, -5,  0,  0,  0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  1, -1],  # Y
            [ 0, -1, -2, -1,  0, -2,  0,  0,  0,  0,  0, -3,  0,  0,  0, -1,  0, -1, -1,  1],  # V
        ],
        dtype=np.int32,
    )
    tcrblosum_beta_substitution_matrix = np.array(
        [
            # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
            [ 0,  0,  0,  0, -5,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # A
            [ 0,  2,  0,  0, -4, -1, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # R
            [ 0,  0,  1,  1, -4,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1,  0,  0,  0,  0],  # N
            [ 0,  0,  1,  1, -4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0],  # D
            [-5, -4, -4, -4,  2, -6, -5,  0, -3, -3, -5, -2, -1, -5, -4,  0, -5, -2, -5, -4],  # C
            [ 0, -1,  0,  0, -6,  2, -1, -1, -1,  0,  1, -1,  0, -2, -1, -2, -1,  0,  0, -1],  # Q
            [-1, -1,  0,  0, -5, -1,  2,  0, -1,  0, -1,  1,  0, -2,  0, -2,  1,  0, -1,  0],  # E
            [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # G
            [ 0,  0,  0,  0, -3, -1, -1,  0,  2,  0,  0, -1,  0,  2,  0, -1,  0,  0,  1,  0],  # H
            [ 0,  0,  0,  0, -3,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0],  # I
            [ 0,  0, -1,  0, -5,  1, -1,  0,  0,  0,  1,  0,  0,  0,  0, -1,  0,  0,  0,  0],  # L
            [ 0,  0,  0,  0, -2, -1,  1,  0, -1,  0,  0,  1,  0, -1,  0,  0,  0,  0, -1,  0],  # K
            [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  0, -1,  0],  # M
            [-1, -1, -1, -1, -5, -2, -2, -1,  2,  0,  0, -1,  0,  2,  0, -2,  0,  0,  2, -1],  # F
            [ 0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0, -1,  0],  # P
            [ 0,  0, -1, -1,  0, -2, -2,  0, -1,  0, -1,  0,  0, -2, -1,  1,  0,  0, -2,  0],  # S
            [ 0,  0,  0,  0, -5, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # T
            [ 0,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # W
            [-1, -1,  0,  0, -5,  0, -1, -1,  1,  0,  0, -1, -1,  2, -1, -2,  0,  0,  2, -1],  # Y
            [ 0,  0,  0,  0, -4, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],  # V
        ],
        dtype=np.int32,
    )
    # fmt: on

    def __init__(
        self,
        cutoff: int = 20,
        *,
        dist_weight: int = 3,
        gap_penalty: int = 4,
        ntrim: int = 3,
        ctrim: int = 2,
        fixed_gappos: bool = True,
        n_jobs: int = -1,
        n_blocks: int = 1,
        histogram: bool = False,
        base_matrix: Literal["blosum62", "tcrblosum"] = "blosum62",
        distance_cap: int | None | Literal["default"] = "default",
        chain_type: Literal["VJ", "VDJ"] | None = None,
    ):
        self.dist_weight = dist_weight
        self.gap_penalty = gap_penalty
        self.ntrim = ntrim
        self.ctrim = ctrim
        self.fixed_gappos = fixed_gappos
        self.cutoff = cutoff
        self.histogram = histogram
        if distance_cap != "default" and distance_cap is not None and not isinstance(distance_cap, int):
            raise ValueError("`distance_cap` must be non-negative, `None`, or 'default'.")
        if isinstance(distance_cap, int) and distance_cap < 0:
            raise ValueError("`distance_cap` must be non-negative, `None`, or 'default'.")

        if base_matrix == "blosum62":
            matrix_distance_cap = 4 if distance_cap == "default" else distance_cap
            self.tcr_nb_distance_matrix = _substitution_to_distance_matrix(
                self.blosum62_substitution_matrix,
                self.parasail_aa_alphabet_with_unknown,
                self.matrix_alphabet,
                distance_cap=matrix_distance_cap,
                distance_offset=4,
            )

        elif base_matrix == "tcrblosum":
            if chain_type == "VJ":
                tcrdist_substitution_matrix = self.tcrblosum_alpha_substitution_matrix
            elif chain_type == "VDJ":
                tcrdist_substitution_matrix = self.tcrblosum_beta_substitution_matrix
            else:
                raise ValueError("`chain_type` must be 'VJ' or 'VDJ' when `base_matrix='tcrblosum'`.")

            off_diagonal = ~np.eye(len(self.matrix_alphabet), dtype=bool)
            max_score = int(
                max(
                    np.max(self.tcrblosum_alpha_substitution_matrix[off_diagonal]),
                    np.max(self.tcrblosum_beta_substitution_matrix[off_diagonal]),
                )
            )

            matrix_distance_cap = None if distance_cap == "default" else distance_cap
            self.tcr_nb_distance_matrix = _substitution_to_distance_matrix(
                tcrdist_substitution_matrix,
                self.parasail_aa_alphabet_with_unknown,
                self.matrix_alphabet,
                distance_cap=matrix_distance_cap,
                distance_offset=max_score + 1,
            )

        else:
            raise ValueError(f"Unknown `base_matrix`: {base_matrix!r}")

        super().__init__(n_jobs=n_jobs, n_blocks=n_blocks, histogram=histogram)

    def _tcrdist_mat(
        self,
        *,
        seqs: Sequence[str],
        seqs2: Sequence[str],
        is_symmetric: bool = False,
        start_column: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Computes the pairwise TCRdist distances for sequences in seqs and seqs2.

        This function is a wrapper and contains an inner JIT compiled numba function without parameters. The reason for this is
        that this way some of the parameters can be treated as constant by numba and this allows for a better optimization
        of the numba compiler in this specific case.

        If this function is used to compute a block of a bigger result matrix, is_symmetric and start_column
        can be used to only compute the part of the block that would be part of the upper triangular matrix of the
        result matrix.

        Note: to use with non-CDR3 sequences set ntrim and ctrim to 0.

        Parameters
        ----------
        seqs/2:
            A python sequence of strings representing gene sequences
        is_symmetric:
            Determines whether the final result matrix is symmetric, assuming that this function is
            only used to compute a block of a bigger result matrix
        start_column:
            Determines at which column the calculation should be started. This is only used if this function is
            used to compute a block of a bigger result matrix that is symmetric

        Returns
        -------
        data_rows:
            List with arrays containing the non-zero data values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        indices_rows:
            List with arrays containing the non-zero entry column indeces of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_element_counts:
            Array with integers that indicate the amount of non-zero values of the result matrix per row,
            needed to create the final scipy CSR result matrix later
        row_mins:
            Always returns a numpy array containing None because the computation of the minimum distance per row is
            not implemented for the tcrdist calculator yet.
        """
        max_seq_len = max(len(s) for s in (*seqs, *seqs2))

        seqs_mat1, seqs_L1 = _seqs2mat(seqs, max_len=max_seq_len)
        seqs_mat2, seqs_L2 = _seqs2mat(seqs2, max_len=max_seq_len)

        cutoff = self.cutoff
        dist_weight = self.dist_weight
        gap_penalty = self.gap_penalty
        ntrim = self.ntrim
        ctrim = self.ctrim
        fixed_gappos = self.fixed_gappos

        dist_mat_weighted = self.tcr_nb_distance_matrix * dist_weight
        start_column *= is_symmetric

        nb.set_num_threads(_get_usable_cpus(n_jobs=self.n_jobs, use_numba=True))

        num_threads = nb.get_num_threads()

        jit_parallel = num_threads > 1

        @nb.jit(nopython=True, parallel=jit_parallel, nogil=True)
        def _nb_tcrdist_mat():
            assert seqs_mat1.shape[0] == seqs_L1.shape[0]
            assert seqs_mat2.shape[0] == seqs_L2.shape[0]

            num_rows = seqs_mat1.shape[0]
            num_cols = seqs_mat2.shape[0]

            data_rows = nb.typed.List()
            indices_rows = nb.typed.List()
            row_element_counts = np.zeros(num_rows)

            empty_row = np.zeros(0)
            for _ in range(0, num_rows):
                data_rows.append([empty_row])
                indices_rows.append([empty_row])

            data_row_matrix = np.empty((num_threads, num_cols))
            indices_row_matrix = np.empty((num_threads, num_cols))

            for row_index in nb.prange(num_rows):
                thread_id = nb.get_thread_id()
                row_end_index = 0
                seq1_len = seqs_L1[row_index]

                for col_index in range(start_column + row_index * is_symmetric, num_cols):
                    distance = 1
                    seq2_len = seqs_L2[col_index]

                    if seq1_len == seq2_len:
                        for i in range(ntrim, seq1_len - ctrim):
                            distance += dist_mat_weighted[seqs_mat1[row_index, i], seqs_mat2[col_index, i]]

                    else:
                        short_len = min(seq1_len, seq2_len)
                        len_diff = abs(seq1_len - seq2_len)
                        if fixed_gappos:
                            min_gappos = min(6, 3 + (short_len - 5) // 2)
                            max_gappos = min_gappos
                        else:
                            min_gappos = 5
                            max_gappos = short_len - 1 - 4
                            while min_gappos > max_gappos:
                                min_gappos -= 1
                                max_gappos += 1
                        min_dist = -1

                        for gappos in range(min_gappos, max_gappos + 1):
                            tmp_dist = 0

                            remainder = short_len - gappos
                            for n_i in range(ntrim, gappos):
                                tmp_dist += dist_mat_weighted[seqs_mat1[row_index, n_i], seqs_mat2[col_index, n_i]]

                            for c_i in range(ctrim, remainder):
                                tmp_dist += dist_mat_weighted[
                                    seqs_mat1[row_index, seq1_len - 1 - c_i], seqs_mat2[col_index, seq2_len - 1 - c_i]
                                ]

                            if tmp_dist < min_dist or min_dist == -1:
                                min_dist = tmp_dist

                            if min_dist == 0:
                                break

                        distance = min_dist + len_diff * gap_penalty + 1

                    if distance <= cutoff + 1:
                        data_row_matrix[thread_id, row_end_index] = distance
                        indices_row_matrix[thread_id, row_end_index] = col_index
                        row_end_index += 1

                data_rows[row_index][0] = data_row_matrix[thread_id, 0:row_end_index].copy()
                indices_rows[row_index][0] = indices_row_matrix[thread_id, 0:row_end_index].copy()
                row_element_counts[row_index] = row_end_index

            data_rows_flat = []
            indices_rows_flat = []

            for i in range(len(data_rows)):
                data_rows_flat.append(data_rows[i][0])
                indices_rows_flat.append(indices_rows[i][0])

            return data_rows_flat, indices_rows_flat, row_element_counts

        data_rows, indices_rows, row_element_counts = _nb_tcrdist_mat()
        return data_rows, indices_rows, row_element_counts, np.array([None])

    _metric_mat = _tcrdist_mat


@_doc_params(params=_doc_params_parallel_distance_calculator)
class AlignmentDistanceCalculator(ParallelDistanceCalculator):
    """\
    Calculates distance between sequences based on pairwise sequence alignment.

    The distance between two sequences is defined as :math:`S_{{1,2}}^{{max}} - S_{{1,2}}`,
    where :math:`S_{{1,2}}` is the alignment score of sequences 1 and 2 and
    :math:`S_{{1,2}}^{{max}}` is the max. achievable alignment score of sequences 1 and 2.
    :math:`S_{{1,2}}^{{max}}` is defined as :math:`\\min(S_{{1,1}}, S_{{2,2}})`.

    The use of alignment-based distances is heavily inspired by :cite:`TCRdist`.

    High-performance sequence alignments are calculated leveraging
    the `parasail library <https://github.com/jeffdaily/parasail-python>`_ (:cite:`Daily2016`).

    Choosing a cutoff:
        Alignment distances need to be viewed in the light of the substitution matrix.
        The alignment distance is the difference between the actual alignment
        score and the max. achievable alignment score. For instance, a mutation
        from *Leucine* (`L`) to *Isoleucine* (`I`) results in a BLOSUM62 score of `2`.
        An `L` aligned with `L` achieves a score of `4`. The distance is, therefore, `2`.

        On the other hand, a single *Tryptophane* (`W`) mutating into, e.g.
        *Proline* (`P`) already results in a distance of `15`.

        We are still lacking empirical data up to which distance a CDR3 sequence still
        is likely to recognize the same antigen, but reasonable cutoffs are `<15`.

    Parameters
    ----------
    cutoff
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices. The default cutoff is `10`.
    {params}
    subst_mat
        Name of parasail substitution matrix
    gap_open
        Gap open penalty
    gap_extend
        Gap extend penatly
    """

    @deprecated(
        """\
        FastAlignmentDistanceCalculator achieves (depending on the settings) identical results
        at a higher speed.
        """
    )
    def __init__(
        self,
        cutoff: int = 10,
        *,
        n_jobs: int = -1,
        block_size: int | None = None,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 11,
    ):
        super().__init__(cutoff, n_jobs=n_jobs, block_size=block_size)
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def _compute_block(self, seqs1, seqs2, origin):
        try:
            import parasail
        except ImportError:
            raise ImportError(
                "Using the alignment distance requires the installation of `parasail`. "
                "You can install it with `pip install parasail`."
            ) from None

        subst_mat = parasail.Matrix(self.subst_mat)
        origin_row, origin_col = origin

        square_matrix = seqs2 is None
        if square_matrix:
            seqs2 = seqs1

        self_alignment_scores1 = self._self_alignment_scores(seqs1)
        if square_matrix:
            self_alignment_scores2 = self_alignment_scores1
        else:
            self_alignment_scores2 = self._self_alignment_scores(seqs2)

        result = []
        for row, s1 in enumerate(seqs1):
            col_start = row if square_matrix else 0
            for col, s2 in enumerate(seqs2[col_start:], start=col_start):
                profile = parasail.profile_create_16(s1, subst_mat)
                r = parasail.nw_scan_profile_16(profile, s2, self.gap_open, self.gap_extend)
                max_score = np.min([self_alignment_scores1[row], self_alignment_scores2[col]])
                d = max_score - r.score
                if d <= self.cutoff:
                    result.append((d + 1, origin_row + row, origin_col + col))

        return result

    def _self_alignment_scores(self, seqs: Sequence) -> dict:
        """Calculate self-alignments. We need them as reference values
        to turn scores into dists
        """
        try:
            import parasail
        except ImportError:
            raise ImportError(
                "Using the alignment distance requires the installation of `parasail`. "
                "You can install it with `pip install parasail`."
            ) from None

        return np.fromiter(
            (
                parasail.nw_scan_16(
                    s,
                    s,
                    self.gap_open,
                    self.gap_extend,
                    parasail.Matrix(self.subst_mat),
                ).score
                for s in seqs
            ),
            dtype=int,
            count=len(seqs),
        )


@_doc_params(params=_doc_params_parallel_distance_calculator)
class FastAlignmentDistanceCalculator(ParallelDistanceCalculator):
    """\
    Calculates distance between sequences based on pairwise sequence alignment.

    The distance between two sequences is defined as :math:`S_{{1,2}}^{{max}} - S_{{1,2}}`,
    where :math:`S_{{1,2}}` is the alignment score of sequences 1 and 2 and
    :math:`S_{{1,2}}^{{max}}` is the max. achievable alignment score of sequences 1 and 2.
    :math:`S_{{1,2}}^{{max}}` is defined as :math:`\\min(S_{{1,1}}, S_{{2,2}})`.

    The use of alignment-based distances is heavily inspired by :cite:`TCRdist`.

    High-performance sequence alignments are calculated leveraging
    the `parasail library <https://github.com/jeffdaily/parasail-python>`_ (:cite:`Daily2016`).

    To speed up the computation, we pre-filter sequence pairs based on
        a) differences in sequence length
        b) the number of different characters, based on an estimate of the mismatch penalty (`estimated_penalty`).

    The filtering based on `estimated_penalty` is a *heuristic* and *may lead to false negatives*, i.e. sequence
    pairs that are actually below the distance cutoff, but are removed during pre-filtering. Sensible values for
    `estimated_penalty` are depending on the substitution matrix, but higher values lead to a higher false negative rate.

    We provide default values for BLOSUM and PAM matrices. The provided default values were obtained by testing different
    alues on the Wu dataset (:cite:`Wu2020`) and selecting those that provided a reasonable balance between
    speedup and loss. Loss stayed well under 10% in all our test cases with the default values, and speedup
    increases with the number of cells.

    While the length-based filter is always active, the filter based on different characters can be disabled by
    setting the estimated penalty to zero. Using length-based filtering only, there won't be any false negatives
    *unless with a substitution matrix in which a substitution results in a higher score than the corresponding match.*
    Using the length-based filter only results in a substancially reduced speedup compared to combining it with
    the `estimated_penalty` heuristic.

    Choosing a cutoff:
        Alignment distances need to be viewed in the light of the substitution matrix.
        The alignment distance is the difference between the actual alignment
        score and the max. achievable alignment score. For instance, a mutation
        from *Leucine* (`L`) to *Isoleucine* (`I`) results in a BLOSUM62 score of `2`.
        An `L` aligned with `L` achieves a score of `4`. The distance is, therefore, `2`.

        On the other hand, a single *Tryptophane* (`W`) mutating into, e.g.
        *Proline* (`P`) already results in a distance of `15`.

        We are still lacking empirical data up to which distance a CDR3 sequence still
        is likely to recognize the same antigen, but reasonable cutoffs are `<15`.

    Choosing an expected penalty:
        The choice of an expected penalty is likely influenced by similar considerations as the
        other parameters. Essentially, this can be thought of as a superficial (dis)similarity
        measure. A higher value more strongly penalizes mismatching characters and is more in line
        with looking for closely related sequence pairs, while a lower value is more forgiving
        and better suited when looking for more distantly related sequence pairs.

    Parameters
    ----------
    cutoff
        Will eleminate distances > cutoff to make efficient
        use of sparse matrices. The default cutoff is `10`.
    {params}
    subst_mat
        Name of parasail substitution matrix
    gap_open
        Gap open penalty
    gap_extend
        Gap extend penatly
    estimated_penalty
        Estimate of the average mismatch penalty
    """

    def __init__(
        self,
        cutoff: int = 10,
        *,
        n_jobs: int = -1,
        block_size: int | None = None,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 11,
        estimated_penalty: float = None,
    ):
        super().__init__(cutoff, n_jobs=n_jobs, block_size=block_size)
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend

        penalty_dict = {
            "blosum30": 4.0,
            "blosum35": 4.0,
            "blosum40": 4.0,
            "blosum45": 4.0,
            "blosum50": 4.0,
            "blosum55": 4.0,
            "blosum60": 4.0,
            "blosum62": 4.0,
            "blosum65": 4.0,
            "blosum70": 4.0,
            "blosum75": 4.0,
            "blosum80": 4.0,
            "blosum85": 4.0,
            "blosum90": 4.0,
            "pam10": 8.0,
            "pam20": 8.0,
            "pam30": 8.0,
            "pam40": 8.0,
            "pam50": 8.0,
            "pam60": 4.0,
            "pam70": 4.0,
            "pam80": 4.0,
            "pam90": 4.0,
            "pam100": 4.0,
            "pam110": 2.0,
            "pam120": 2.0,
            "pam130": 2.0,
            "pam140": 2.0,
            "pam150": 2.0,
            "pam160": 2.0,
            "pam170": 2.0,
            "pam180": 2.0,
            "pam190": 2.0,
            "pam200": 2.0,
        }

        if subst_mat not in penalty_dict.keys():
            raise Exception("Invalid substitution matrix.")

        self.estimated_penalty = estimated_penalty if estimated_penalty is not None else penalty_dict[subst_mat]

    def _compute_block(self, seqs1, seqs2, origin):
        try:
            import parasail
        except ImportError:
            raise ImportError(
                "Using the alignment distance requires the installation of `parasail`. "
                "You can install it with `pip install parasail`."
            ) from None

        subst_mat = parasail.Matrix(self.subst_mat)
        origin_row, origin_col = origin

        square_matrix = seqs2 is None
        if square_matrix:
            seqs2 = seqs1

        self_alignment_scores1 = self._self_alignment_scores(seqs1)
        if square_matrix:
            self_alignment_scores2 = self_alignment_scores1
        else:
            self_alignment_scores2 = self._self_alignment_scores(seqs2)

        max_len_diff = ((self.cutoff - self.gap_open) / self.gap_extend) + 1

        result = []
        for row, s1 in enumerate(seqs1):
            col_start = row if square_matrix else 0
            profile = parasail.profile_create_16(s1, subst_mat)
            len1 = len(s1)

            for col, s2 in enumerate(seqs2[col_start:], start=col_start):
                len_diff = abs(len1 - len(s2))
                # No need to calculate diagonal values
                if s1 == s2:
                    result.append((1, origin_row + row, origin_col + col))
                # Dismiss sequences based on length
                elif len_diff <= max_len_diff:
                    # Dismiss sequences that are too different
                    if (
                        self._num_different_characters(s1, s2, len_diff) * self.estimated_penalty
                        + len_diff * self.gap_extend
                        <= self.cutoff
                    ):
                        r = parasail.nw_scan_profile_16(profile, s2, self.gap_open, self.gap_extend)
                        max_score = np.min([self_alignment_scores1[row], self_alignment_scores2[col]])
                        d = max_score - r.score

                        if d <= self.cutoff:
                            result.append((d + 1, origin_row + row, origin_col + col))

        return result

    def _self_alignment_scores(self, seqs: Sequence) -> dict:
        """Calculate self-alignments. We need them as reference values
        to turn scores into dists
        """
        try:
            import parasail
        except ImportError:
            raise ImportError(
                "Using the alignment distance requires the installation of `parasail`. "
                "You can install it with `pip install parasail`."
            ) from None

        return np.fromiter(
            (
                parasail.nw_scan_16(
                    s,
                    s,
                    self.gap_open,
                    self.gap_extend,
                    parasail.Matrix(self.subst_mat),
                ).score
                for s in seqs
            ),
            dtype=int,
            count=len(seqs),
        )

    def _num_different_characters(self, s1, s2, len_diff):
        longer, shorter = (s1, s2) if len(s1) >= len(s2) else (s2, s1)

        for c in shorter:
            if c in longer:
                longer = longer.replace(c, "", 1)
        return len(longer) - len_diff
