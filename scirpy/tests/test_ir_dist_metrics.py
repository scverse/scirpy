import pytest
from scirpy.ir_dist.metrics import (
    AlignmentDistanceCalculator,
    DistanceCalculator,
    IdentityDistanceCalculator,
    LevenshteinDistanceCalculator,
    HammingDistanceCalculator,
    ParallelDistanceCalculator,
)
import numpy as np
import numpy.testing as npt
import scirpy as ir
import scipy.sparse
from .util import _squarify


def test_squarify():
    npt.assert_almost_equal(
        DistanceCalculator.squarify(
            scipy.sparse.csr_matrix(
                np.array(
                    [
                        [1, 2, 3, 3],
                        [0, 1, 2, 2],
                        [0, 0, 1, 2],
                        [0, 0, 0, 1],
                    ]
                )
            )
        ).toarray(),
        np.array(
            [
                [1, 2, 3, 3],
                [2, 1, 2, 2],
                [3, 2, 1, 2],
                [3, 2, 2, 1],
            ]
        ),
    )
    npt.assert_almost_equal(
        DistanceCalculator.squarify(
            scipy.sparse.csr_matrix(
                np.array(
                    [
                        [1, 2, 0, 0],
                        [0, 1, 2, 2],
                        [0, 0, 1, 2],
                        [0, 0, 0, 1],
                    ]
                )
            )
        ).toarray(),
        np.array(
            [
                [1, 2, 0, 0],
                [2, 1, 2, 2],
                [0, 2, 1, 2],
                [0, 2, 2, 1],
            ]
        ),
    )


def test_block_iter():
    seqs1 = list("ABCDE")
    seqs2 = list("HIJKLM")
    b1 = list(ParallelDistanceCalculator._block_iter(seqs1, block_size=2))
    b2 = list(ParallelDistanceCalculator._block_iter(seqs1, seqs2, block_size=3))
    b3 = list(ParallelDistanceCalculator._block_iter(seqs1, seqs2, block_size=50))
    b4 = list(
        ParallelDistanceCalculator._block_iter(list("ABC"), list("ABC"), block_size=1)
    )
    L = list
    assert b1 == [
        (L("AB"), None, (0, 0)),
        (L("AB"), L("CD"), (0, 2)),
        (L("AB"), L("E"), (0, 4)),
        (L("CD"), None, (2, 2)),
        (L("CD"), L("E"), (2, 4)),
        (L("E"), None, (4, 4)),
    ]
    assert b2 == [
        (L("ABC"), L("HIJ"), (0, 0)),
        (L("ABC"), L("KLM"), (0, 3)),
        (L("DE"), L("HIJ"), (3, 0)),
        (L("DE"), L("KLM"), (3, 3)),
    ]
    assert b3 == [(L("ABCDE"), L("HIJKLM"), (0, 0))]
    assert b4 == [
        (L("A"), L("A"), (0, 0)),
        (L("A"), L("B"), (0, 1)),
        (L("A"), L("C"), (0, 2)),
        (L("B"), L("A"), (1, 0)),
        (L("B"), L("B"), (1, 1)),
        (L("B"), L("C"), (1, 2)),
        (L("C"), L("A"), (2, 0)),
        (L("C"), L("B"), (2, 1)),
        (L("C"), L("C"), (2, 2)),
    ]


def test_identity_dist():
    identity = IdentityDistanceCalculator()
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA"])
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


def test_identity_dist_with_two_seq_arrays():
    identity = IdentityDistanceCalculator()
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA", "SAS"], ["KKL", "ARS", "RSA"])
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (4, 3)
    assert res.nnz == 3
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    )

    # test with completely empty result matrix
    res = identity.calc_dist_mat(["ARS", "ARS", "RSA", "SAS"], ["foo", "bar", "baz"])
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (4, 3)
    assert res.nnz == 0
    npt.assert_almost_equal(
        res.toarray(),
        np.zeros((4, 3)),
    )


def test_levenshtein_compute_block():
    levenshtein1 = LevenshteinDistanceCalculator(1)
    seqs = np.array(["A", "AAA", "AA"])
    seqs2 = np.array(["AB", "BAA"])
    b1 = list(levenshtein1._compute_block(seqs, None, (10, 20)))
    b2 = list(levenshtein1._compute_block(seqs, seqs, (10, 20)))
    b3 = list(levenshtein1._compute_block(seqs, seqs2, (10, 20)))
    b4 = list(levenshtein1._compute_block(seqs2, seqs, (10, 20)))

    assert b1 == [(1, 10, 20), (2, 10, 22), (1, 11, 21), (2, 11, 22), (1, 12, 22)]
    assert b2 == [
        (1, 10, 20),
        (2, 10, 22),
        (1, 11, 21),
        (2, 11, 22),
        (2, 12, 20),
        (2, 12, 21),
        (1, 12, 22),
    ]
    assert b3 == [(2, 10, 20), (2, 11, 21), (2, 12, 20), (2, 12, 21)]
    assert b4 == [(2, 10, 20), (2, 10, 22), (2, 11, 21), (2, 11, 22)]


def test_levensthein_dist():
    levenshtein10 = LevenshteinDistanceCalculator(10, block_size=50)
    levenshtein10_2 = LevenshteinDistanceCalculator(10, block_size=2)
    levenshtein1 = LevenshteinDistanceCalculator(1, n_jobs=1, block_size=1)

    res10 = levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))
    res10_2 = levenshtein10_2.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))
    res1 = levenshtein1.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]))

    assert isinstance(res10, scipy.sparse.csr_matrix)
    assert isinstance(res10_2, scipy.sparse.csr_matrix)
    assert isinstance(res1, scipy.sparse.csr_matrix)

    npt.assert_equal(res10.toarray(), res10_2.toarray())
    npt.assert_almost_equal(
        res10.toarray(),
        np.array(
            [
                [1, 2, 3, 3],
                [2, 1, 2, 2],
                [3, 2, 1, 2],
                [3, 2, 2, 1],
            ]
        ),
    )
    npt.assert_almost_equal(
        res1.toarray(),
        np.array(
            [
                [1, 2, 0, 0],
                [2, 1, 2, 2],
                [0, 2, 1, 2],
                [0, 2, 2, 1],
            ]
        ),
    )


def test_levensthein_dist_with_two_seq_arrays():
    levenshtein10 = LevenshteinDistanceCalculator(2)
    res = levenshtein10.calc_dist_mat(
        np.array(["A", "AA", "AAA", "AAR", "ZZZZZZ"]), np.array(["RRR", "AR"])
    )
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (5, 2)
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[0, 2], [0, 2], [0, 3], [3, 2], [0, 0]]),
    )


def test_hamming_dist():
    hamming10 = HammingDistanceCalculator(2)
    res = hamming10.calc_dist_mat(
        np.array(["A", "AA", "AAA", "AAR", "ZZZZZZ"]), np.array(["RRR", "AR"])
    )
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (5, 2)
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[0, 0], [0, 2], [0, 0], [3, 0], [0, 0]]),
    )


def test_alignment_compute_block():
    aligner = AlignmentDistanceCalculator(cutoff=255)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
    seqs = ["AWAW", "VWVW", "HHHH"]

    b1 = aligner._compute_block(seqs, None, (0, 0))
    b2 = aligner10._compute_block(seqs, None, (10, 20))
    b3 = aligner10._compute_block(seqs, seqs, (10, 20))

    assert b1 == [(1, 0, 0), (9, 0, 1), (39, 0, 2), (1, 1, 1), (41, 1, 2), (1, 2, 2)]
    assert b2 == [(1, 10, 20), (9, 10, 21), (1, 11, 21), (1, 12, 22)]
    assert b3 == [(1, 10, 20), (9, 10, 21), (9, 11, 20), (1, 11, 21), (1, 12, 22)]


def test_alignment_dist():
    with pytest.raises(ValueError):
        AlignmentDistanceCalculator(3000)
    aligner = AlignmentDistanceCalculator(cutoff=255, n_jobs=1)
    aligner10 = AlignmentDistanceCalculator(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(
        res.toarray(), _squarify(np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]]))
    )

    res = aligner10.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(
        res.toarray(), _squarify(np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]]))
    )


def test_alignment_dist_with_two_seq_arrays():
    aligner = AlignmentDistanceCalculator(cutoff=10, n_jobs=1)
    res = aligner.calc_dist_mat(
        ["AAAA", "AATA", "HHHH", "WWWW"], ["WWWW", "AAAA", "ATAA"]
    )
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (4, 3)

    npt.assert_almost_equal(
        res.toarray(), np.array([[0, 1, 5], [0, 5, 10], [0, 0, 0], [1, 0, 0]])
    )


@pytest.mark.parametrize("metric", ["alignment", "identity", "hamming", "levenshtein"])
def test_sequence_dist_all_metrics(metric):
    """Smoke test, no assertions!"""
    unique_seqs = np.array(["AAA", "ARA", "AFFFFFA", "FAFAFA", "FFF"])
    seqs2 = np.array(["RRR", "FAFA", "WWWWWWW"])
    dist_mat = ir.ir_dist.sequence_dist(unique_seqs, metric=metric, cutoff=8, n_jobs=2)
    assert dist_mat.shape == (5, 5)

    dist_mat = ir.ir_dist.sequence_dist(
        unique_seqs, seqs2, metric=metric, cutoff=8, n_jobs=2
    )
    assert dist_mat.shape == (5, 3)
