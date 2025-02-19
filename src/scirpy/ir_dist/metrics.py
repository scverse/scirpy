import abc
import itertools
import warnings
from collections.abc import Sequence

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
            # actually compare the values
            def coord_generator():
                for (i1, s1), (i2, s2) in itertools.product(enumerate(seqs), enumerate(seqs2)):
                    if s1 == s2:
                        yield 1, i1, i2

            try:
                d, row, col = zip(*coord_generator(), strict=False)
            except ValueError:
                # happens when there is no match at all
                d, row, col = (), (), ()

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


def _make_numba_matrix(distance_matrix: dict, alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX*") -> np.ndarray:
    """Creates a numba compatible distance matrix from a dict of tuples.

    Parameters
    ----------
    distance_matrix:
        Keys are tuples like ('A', 'C') with values containing an integer.

    Returns
    -------
    distance_matrix:
        distance lookup matrix
    """
    dm = np.zeros((len(alphabet), len(alphabet)), dtype=np.int32)
    for (aa1, aa2), d in distance_matrix.items():
        dm[alphabet.index(aa1), alphabet.index(aa2)] = d
        dm[alphabet.index(aa2), alphabet.index(aa1)] = d
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
    L = np.zeros(len(seqs), dtype=np.int8)
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
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
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
            L = np.zeros(len(seqs), dtype=np.int8)
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

        result_sparse = result_blocks[0]
        for i in range(1, len(result_blocks)):
            result_sparse += result_blocks[i]

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
    """

    parasail_aa_alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
    parasail_aa_alphabet_with_unknown = "ARNDCQEGHILKMFPSTWYVBZX*"
    # fmt: off
    tcr_dict_distance_matrix = {('A', 'A'): 0,  ('A', 'C'): 4,  ('A', 'D'): 4,  ('A', 'E'): 4,  ('A', 'F'): 4,  ('A', 'G'): 4,  ('A', 'H'): 4,  ('A', 'I'): 4,  ('A', 'K'): 4,  ('A', 'L'): 4,  ('A', 'M'): 4,  ('A', 'N'): 4,  ('A', 'P'): 4,  ('A', 'Q'): 4,  ('A', 'R'): 4,  ('A', 'S'): 3,  ('A', 'T'): 4,  ('A', 'V'): 4,  ('A', 'W'): 4,  ('A', 'Y'): 4,  ('C', 'A'): 4,  ('C', 'C'): 0,  ('C', 'D'): 4,  ('C', 'E'): 4,  ('C', 'F'): 4,  ('C', 'G'): 4,  ('C', 'H'): 4,  ('C', 'I'): 4,  ('C', 'K'): 4,  ('C', 'L'): 4,  ('C', 'M'): 4,  ('C', 'N'): 4,  ('C', 'P'): 4,  ('C', 'Q'): 4,  ('C', 'R'): 4,  ('C', 'S'): 4,  ('C', 'T'): 4,  ('C', 'V'): 4,  ('C', 'W'): 4,  ('C', 'Y'): 4,  ('D', 'A'): 4,  ('D', 'C'): 4,  ('D', 'D'): 0,  ('D', 'E'): 2,  ('D', 'F'): 4,  ('D', 'G'): 4,  ('D', 'H'): 4,  ('D', 'I'): 4,  ('D', 'K'): 4,  ('D', 'L'): 4,  ('D', 'M'): 4,  ('D', 'N'): 3,  ('D', 'P'): 4,  ('D', 'Q'): 4,  ('D', 'R'): 4,  ('D', 'S'): 4,  ('D', 'T'): 4,  ('D', 'V'): 4,  ('D', 'W'): 4,  ('D', 'Y'): 4,  ('E', 'A'): 4,  ('E', 'C'): 4,  ('E', 'D'): 2,  ('E', 'E'): 0,  ('E', 'F'): 4,  ('E', 'G'): 4,  ('E', 'H'): 4,  ('E', 'I'): 4,  ('E', 'K'): 3,  ('E', 'L'): 4,  ('E', 'M'): 4,  ('E', 'N'): 4,  ('E', 'P'): 4,  ('E', 'Q'): 2,  ('E', 'R'): 4,  ('E', 'S'): 4,  ('E', 'T'): 4,  ('E', 'V'): 4,  ('E', 'W'): 4,  ('E', 'Y'): 4,  ('F', 'A'): 4,  ('F', 'C'): 4,  ('F', 'D'): 4,  ('F', 'E'): 4,  ('F', 'F'): 0,  ('F', 'G'): 4,  ('F', 'H'): 4,  ('F', 'I'): 4,  ('F', 'K'): 4,  ('F', 'L'): 4,  ('F', 'M'): 4,  ('F', 'N'): 4,  ('F', 'P'): 4,  ('F', 'Q'): 4,  ('F', 'R'): 4,  ('F', 'S'): 4,  ('F', 'T'): 4,  ('F', 'V'): 4,  ('F', 'W'): 3,  ('F', 'Y'): 1,  ('G', 'A'): 4,  ('G', 'C'): 4,  ('G', 'D'): 4,  ('G', 'E'): 4,  ('G', 'F'): 4,  ('G', 'G'): 0,  ('G', 'H'): 4,  ('G', 'I'): 4,  ('G', 'K'): 4,  ('G', 'L'): 4,  ('G', 'M'): 4,  ('G', 'N'): 4,  ('G', 'P'): 4,  ('G', 'Q'): 4,  ('G', 'R'): 4,  ('G', 'S'): 4,  ('G', 'T'): 4,  ('G', 'V'): 4,  ('G', 'W'): 4,  ('G', 'Y'): 4,  ('H', 'A'): 4,  ('H', 'C'): 4,  ('H', 'D'): 4,  ('H', 'E'): 4,  ('H', 'F'): 4,  ('H', 'G'): 4,  ('H', 'H'): 0,  ('H', 'I'): 4,  ('H', 'K'): 4,  ('H', 'L'): 4,  ('H', 'M'): 4,  ('H', 'N'): 3,  ('H', 'P'): 4,  ('H', 'Q'): 4,  ('H', 'R'): 4,  ('H', 'S'): 4,  ('H', 'T'): 4,  ('H', 'V'): 4,  ('H', 'W'): 4,  ('H', 'Y'): 2,  ('I', 'A'): 4,  ('I', 'C'): 4,  ('I', 'D'): 4,  ('I', 'E'): 4,  ('I', 'F'): 4,  ('I', 'G'): 4,  ('I', 'H'): 4,  ('I', 'I'): 0,  ('I', 'K'): 4,  ('I', 'L'): 2,  ('I', 'M'): 3,  ('I', 'N'): 4,  ('I', 'P'): 4,  ('I', 'Q'): 4,  ('I', 'R'): 4,  ('I', 'S'): 4,  ('I', 'T'): 4,  ('I', 'V'): 1,  ('I', 'W'): 4,  ('I', 'Y'): 4,  ('K', 'A'): 4,  ('K', 'C'): 4,  ('K', 'D'): 4,  ('K', 'E'): 3,  ('K', 'F'): 4,  ('K', 'G'): 4,  ('K', 'H'): 4,  ('K', 'I'): 4,  ('K', 'K'): 0,  ('K', 'L'): 4,  ('K', 'M'): 4,  ('K', 'N'): 4,  ('K', 'P'): 4,  ('K', 'Q'): 3,  ('K', 'R'): 2,  ('K', 'S'): 4,  ('K', 'T'): 4,  ('K', 'V'): 4,  ('K', 'W'): 4,  ('K', 'Y'): 4,  ('L', 'A'): 4,  ('L', 'C'): 4,  ('L', 'D'): 4,  ('L', 'E'): 4,  ('L', 'F'): 4,  ('L', 'G'): 4,  ('L', 'H'): 4,  ('L', 'I'): 2,  ('L', 'K'): 4,  ('L', 'L'): 0,  ('L', 'M'): 2,  ('L', 'N'): 4,  ('L', 'P'): 4,  ('L', 'Q'): 4,  ('L', 'R'): 4,  ('L', 'S'): 4,  ('L', 'T'): 4,  ('L', 'V'): 3,  ('L', 'W'): 4,  ('L', 'Y'): 4,  ('M', 'A'): 4,  ('M', 'C'): 4,  ('M', 'D'): 4,  ('M', 'E'): 4,  ('M', 'F'): 4,  ('M', 'G'): 4,  ('M', 'H'): 4,  ('M', 'I'): 3,  ('M', 'K'): 4,  ('M', 'L'): 2,  ('M', 'M'): 0,  ('M', 'N'): 4,  ('M', 'P'): 4,  ('M', 'Q'): 4,  ('M', 'R'): 4,  ('M', 'S'): 4,  ('M', 'T'): 4,  ('M', 'V'): 3,  ('M', 'W'): 4,  ('M', 'Y'): 4,  ('N', 'A'): 4,  ('N', 'C'): 4,  ('N', 'D'): 3,  ('N', 'E'): 4,  ('N', 'F'): 4,  ('N', 'G'): 4,  ('N', 'H'): 3,  ('N', 'I'): 4,  ('N', 'K'): 4,  ('N', 'L'): 4,  ('N', 'M'): 4,  ('N', 'N'): 0,  ('N', 'P'): 4,  ('N', 'Q'): 4,  ('N', 'R'): 4,  ('N', 'S'): 3,  ('N', 'T'): 4,  ('N', 'V'): 4,  ('N', 'W'): 4,  ('N', 'Y'): 4,  ('P', 'A'): 4,  ('P', 'C'): 4,  ('P', 'D'): 4,  ('P', 'E'): 4,  ('P', 'F'): 4,  ('P', 'G'): 4,  ('P', 'H'): 4,  ('P', 'I'): 4,  ('P', 'K'): 4,  ('P', 'L'): 4,  ('P', 'M'): 4,  ('P', 'N'): 4,  ('P', 'P'): 0,  ('P', 'Q'): 4,  ('P', 'R'): 4,  ('P', 'S'): 4,  ('P', 'T'): 4,  ('P', 'V'): 4,  ('P', 'W'): 4,  ('P', 'Y'): 4,  ('Q', 'A'): 4,  ('Q', 'C'): 4,  ('Q', 'D'): 4,  ('Q', 'E'): 2,  ('Q', 'F'): 4,  ('Q', 'G'): 4,  ('Q', 'H'): 4,  ('Q', 'I'): 4,  ('Q', 'K'): 3,  ('Q', 'L'): 4,  ('Q', 'M'): 4,  ('Q', 'N'): 4,  ('Q', 'P'): 4,  ('Q', 'Q'): 0,  ('Q', 'R'): 3,  ('Q', 'S'): 4,  ('Q', 'T'): 4,  ('Q', 'V'): 4,  ('Q', 'W'): 4,  ('Q', 'Y'): 4,  ('R', 'A'): 4,  ('R', 'C'): 4,  ('R', 'D'): 4,  ('R', 'E'): 4,  ('R', 'F'): 4,  ('R', 'G'): 4,  ('R', 'H'): 4,  ('R', 'I'): 4,  ('R', 'K'): 2,  ('R', 'L'): 4,  ('R', 'M'): 4,  ('R', 'N'): 4,  ('R', 'P'): 4,  ('R', 'Q'): 3,  ('R', 'R'): 0,  ('R', 'S'): 4,  ('R', 'T'): 4,  ('R', 'V'): 4,  ('R', 'W'): 4,  ('R', 'Y'): 4,  ('S', 'A'): 3,  ('S', 'C'): 4,  ('S', 'D'): 4,  ('S', 'E'): 4,  ('S', 'F'): 4,  ('S', 'G'): 4,  ('S', 'H'): 4,  ('S', 'I'): 4,  ('S', 'K'): 4,  ('S', 'L'): 4,  ('S', 'M'): 4,  ('S', 'N'): 3,  ('S', 'P'): 4,  ('S', 'Q'): 4,  ('S', 'R'): 4,  ('S', 'S'): 0,  ('S', 'T'): 3,  ('S', 'V'): 4,  ('S', 'W'): 4,  ('S', 'Y'): 4,  ('T', 'A'): 4,  ('T', 'C'): 4,  ('T', 'D'): 4,  ('T', 'E'): 4,  ('T', 'F'): 4,  ('T', 'G'): 4,  ('T', 'H'): 4,  ('T', 'I'): 4,  ('T', 'K'): 4,  ('T', 'L'): 4,  ('T', 'M'): 4,  ('T', 'N'): 4,  ('T', 'P'): 4,  ('T', 'Q'): 4,  ('T', 'R'): 4,  ('T', 'S'): 3,  ('T', 'T'): 0,  ('T', 'V'): 4,  ('T', 'W'): 4,  ('T', 'Y'): 4,  ('V', 'A'): 4,  ('V', 'C'): 4,  ('V', 'D'): 4,  ('V', 'E'): 4,  ('V', 'F'): 4,  ('V', 'G'): 4,  ('V', 'H'): 4,  ('V', 'I'): 1,  ('V', 'K'): 4,  ('V', 'L'): 3,  ('V', 'M'): 3,  ('V', 'N'): 4,  ('V', 'P'): 4,  ('V', 'Q'): 4,  ('V', 'R'): 4,  ('V', 'S'): 4,  ('V', 'T'): 4,  ('V', 'V'): 0,  ('V', 'W'): 4,  ('V', 'Y'): 4,  ('W', 'A'): 4,  ('W', 'C'): 4,  ('W', 'D'): 4,  ('W', 'E'): 4,  ('W', 'F'): 3,  ('W', 'G'): 4,  ('W', 'H'): 4,  ('W', 'I'): 4,  ('W', 'K'): 4,  ('W', 'L'): 4,  ('W', 'M'): 4,  ('W', 'N'): 4,  ('W', 'P'): 4,  ('W', 'Q'): 4,  ('W', 'R'): 4,  ('W', 'S'): 4,  ('W', 'T'): 4,  ('W', 'V'): 4,  ('W', 'W'): 0,  ('W', 'Y'): 2,  ('Y', 'A'): 4,  ('Y', 'C'): 4,  ('Y', 'D'): 4,  ('Y', 'E'): 4,  ('Y', 'F'): 1,  ('Y', 'G'): 4,  ('Y', 'H'): 2,  ('Y', 'I'): 4,  ('Y', 'K'): 4,  ('Y', 'L'): 4,  ('Y', 'M'): 4,  ('Y', 'N'): 4,  ('Y', 'P'): 4,  ('Y', 'Q'): 4,  ('Y', 'R'): 4,  ('Y', 'S'): 4,  ('Y', 'T'): 4,  ('Y', 'V'): 4,  ('Y', 'W'): 2,  ('Y', 'Y'): 0}
    # fmt: on
    tcr_nb_distance_matrix = _make_numba_matrix(tcr_dict_distance_matrix, parasail_aa_alphabet_with_unknown)

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
    ):
        self.dist_weight = dist_weight
        self.gap_penalty = gap_penalty
        self.ntrim = ntrim
        self.ctrim = ctrim
        self.fixed_gappos = fixed_gappos
        self.cutoff = cutoff
        self.histogram = histogram
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
