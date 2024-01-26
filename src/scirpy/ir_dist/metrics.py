import abc
import itertools
import warnings
from collections.abc import Sequence
from typing import Optional, Union

import joblib
import numpy as np
import scipy.sparse
import scipy.spatial
from Levenshtein import distance as levenshtein_dist
from Levenshtein import hamming as hamming_dist
from scanpy import logging
from scipy.sparse import coo_matrix, csr_matrix

from scirpy.util import _doc_params, _parallelize_with_joblib, deprecated

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

    def __init__(self, cutoff: Union[int, None]):
        if cutoff > 255:
            raise ValueError("Using a cutoff > 255 is not possible due to the `uint8` dtype used")
        self.cutoff = cutoff

    @_doc_params(dist_mat=_doc_dist_mat)
    @abc.abstractmethod
    def calc_dist_mat(self, seqs: Sequence[str], seqs2: Optional[Sequence[str]] = None) -> csr_matrix:
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
        n_jobs: Optional[int] = -1,
        block_size: Optional[int] = None,
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
        seqs2: Union[Sequence[str], None],
        origin: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute the distances for a block of the matrix

        Parameters
        ----------
        seqs1
            array containing sequences
        seqs2
            other array containing sequences. If `None` compute the square matrix
            of `seqs1` and iteratoe over the upper triangle including the diagonal only.
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
        seqs2: Optional[Sequence[str]] = None,
        block_size: Optional[int] = 50,
    ) -> tuple[Sequence[str], Union[Sequence[str], None], tuple[int, int]]:
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
        self, seqs: Sequence[str], seqs2: Optional[Sequence[str]] = None, *, block_size: Optional[int] = None
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
            dists, rows, cols = zip(*itertools.chain(*block_results))
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
                d, row, col = zip(*coord_generator())
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


@_doc_params(params=_doc_params_parallel_distance_calculator)
class HammingDistanceCalculator(ParallelDistanceCalculator):
    """\
    Calculates the Hamming distance between sequences of identical length.

    The edit distance is the total number of substitution events. Sequences
    with different lengths will be treated as though they exceeded the
    distance-cutoff, i.e. they receive a distance of `0` in the sparse distance
    matrix and will not be connected by an edge in the graph.

    This class relies on `Python-levenshtein <https://github.com/ztane/python-Levenshtein>`_
    to calculate the distances.

    Choosing a cutoff:
        Each modification stands for a substitution event.
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
            # require identical length of sequences
            if len(s1) != len(s2):
                continue
            d = hamming_dist(s1, s2)
            if d <= self.cutoff:
                result.append((d + 1, origin_row + row, origin_col + col))

        return result


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
        n_jobs: Union[int, None] = -1,
        block_size: Optional[int] = None,
        subst_mat: str = "blosum62",
        gap_open: int = 11,
        gap_extend: int = 11,
    ):
        super().__init__(cutoff, n_jobs=n_jobs, block_size=block_size)
        self.subst_mat = subst_mat
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def _compute_block(self, seqs1, seqs2, origin):
        import parasail

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
        import parasail

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
        n_jobs: Union[int, None] = None,
        block_size: Optional[int] = None,
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
        import parasail

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
        import parasail

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
