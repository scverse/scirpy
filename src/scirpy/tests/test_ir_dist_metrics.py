from functools import partial

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse

import scirpy as ir
from scirpy.ir_dist.metrics import (
    AlignmentDistanceCalculator,
    DistanceCalculator,
    FastAlignmentDistanceCalculator,
    GPUHammingDistanceCalculator,
    HammingDistanceCalculator,
    IdentityDistanceCalculator,
    LevenshteinDistanceCalculator,
    ParallelDistanceCalculator,
    TCRdistDistanceCalculator,
)

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
    b4 = list(ParallelDistanceCalculator._block_iter(list("ABC"), list("ABC"), block_size=1))
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
    levenshtein10 = LevenshteinDistanceCalculator(10)
    levenshtein10_2 = LevenshteinDistanceCalculator(10)
    levenshtein1 = LevenshteinDistanceCalculator(1, n_jobs=1)

    res10 = levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]), block_size=50)
    res10_2 = levenshtein10_2.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]), block_size=2)
    res1 = levenshtein1.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR"]), block_size=1)

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
    res = levenshtein10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR", "ZZZZZZ"]), np.array(["RRR", "AR"]))
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (5, 2)
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[0, 2], [0, 2], [0, 3], [3, 2], [0, 0]]),
    )


def test_hamming_dist():
    hamming10 = HammingDistanceCalculator(cutoff=2)
    res = hamming10.calc_dist_mat(np.array(["A", "AA", "AAA", "AAR", "ZZZZZZ"]), np.array(["RRR", "AR"]))
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (5, 2)
    npt.assert_almost_equal(
        res.toarray(),
        np.array([[0, 0], [0, 2], [0, 0], [3, 0], [0, 0]]),
    )


@pytest.mark.extra
@pytest.mark.parametrize(
    "metric", [AlignmentDistanceCalculator, partial(FastAlignmentDistanceCalculator, estimated_penalty=0)]
)
def test_alignment_compute_block(metric):
    aligner = metric(cutoff=255)
    aligner10 = metric(cutoff=10)
    seqs = ["AWAW", "VWVW", "HHHH"]

    b1 = aligner._compute_block(seqs, None, (0, 0))
    b2 = aligner10._compute_block(seqs, None, (10, 20))
    b3 = aligner10._compute_block(seqs, seqs, (10, 20))

    assert b1 == [(1, 0, 0), (9, 0, 1), (39, 0, 2), (1, 1, 1), (41, 1, 2), (1, 2, 2)]
    assert b2 == [(1, 10, 20), (9, 10, 21), (1, 11, 21), (1, 12, 22)]
    assert b3 == [(1, 10, 20), (9, 10, 21), (9, 11, 20), (1, 11, 21), (1, 12, 22)]


@pytest.mark.extra
@pytest.mark.parametrize(
    "metric", [AlignmentDistanceCalculator, partial(FastAlignmentDistanceCalculator, estimated_penalty=0)]
)
def test_alignment_dist(metric):
    with pytest.raises(ValueError):
        metric(3000)
    aligner = metric(cutoff=255, n_jobs=1)
    aligner10 = metric(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(res.toarray(), _squarify(np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]])))

    res = aligner10.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(res.toarray(), _squarify(np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]])))


@pytest.mark.extra
@pytest.mark.parametrize(
    "metric", [AlignmentDistanceCalculator, partial(FastAlignmentDistanceCalculator, estimated_penalty=0)]
)
def test_alignment_dist_with_two_seq_arrays(metric):
    aligner = metric(cutoff=10, n_jobs=1)
    res = aligner.calc_dist_mat(["AAAA", "AATA", "HHHH", "WWWW"], ["WWWW", "AAAA", "ATAA"])
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (4, 3)

    npt.assert_almost_equal(res.toarray(), np.array([[0, 1, 5], [0, 5, 10], [0, 0, 0], [1, 0, 0]]))


@pytest.mark.extra
def test_fast_alignment_compute_block():
    aligner = FastAlignmentDistanceCalculator(cutoff=255)
    aligner10 = FastAlignmentDistanceCalculator(cutoff=10)
    seqs = ["AWAW", "VWVW", "HHHH"]

    b1 = aligner._compute_block(seqs, None, (0, 0))
    b2 = aligner10._compute_block(seqs, None, (10, 20))
    b3 = aligner10._compute_block(seqs, seqs, (10, 20))

    assert b1 == [(1, 0, 0), (9, 0, 1), (39, 0, 2), (1, 1, 1), (41, 1, 2), (1, 2, 2)]
    assert b2 == [(1, 10, 20), (9, 10, 21), (1, 11, 21), (1, 12, 22)]
    assert b3 == [(1, 10, 20), (9, 10, 21), (9, 11, 20), (1, 11, 21), (1, 12, 22)]


@pytest.mark.extra
def test_fast_alignment_dist():
    with pytest.raises(ValueError):
        FastAlignmentDistanceCalculator(3000)
    aligner = FastAlignmentDistanceCalculator(cutoff=255, n_jobs=1)
    aligner10 = FastAlignmentDistanceCalculator(cutoff=10)
    seqs = np.array(["AAAA", "AAHA", "HHHH"])

    res = aligner.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(res.toarray(), _squarify(np.array([[1, 7, 25], [0, 1, 19], [0, 0, 1]])))

    res = aligner10.calc_dist_mat(seqs)
    assert isinstance(res, scipy.sparse.csr_matrix)
    npt.assert_almost_equal(res.toarray(), _squarify(np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]])))


@pytest.mark.extra
def test_fast_alignment_dist_with_two_seq_arrays():
    aligner = FastAlignmentDistanceCalculator(cutoff=10, n_jobs=1)
    res = aligner.calc_dist_mat(["AAAA", "AATA", "HHHH", "WWWW"], ["WWWW", "AAAA", "ATAA"])
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == (4, 3)

    npt.assert_almost_equal(res.toarray(), np.array([[0, 1, 5], [0, 5, 10], [0, 0, 0], [1, 0, 0]]))


@pytest.mark.extra
@pytest.mark.parametrize(
    "metric", ["alignment", "fastalignment", "identity", "hamming", "normalized_hamming", "levenshtein", "tcrdist"]
)
@pytest.mark.parametrize("n_jobs", [-1, 1, 2])
def test_sequence_dist_all_metrics(metric, n_jobs):
    # Smoke test, no assertions!
    # Smoke test, no assertions!
    metrics_with_n_blocks = ["hamming", "normalized_hamming", "tcrdist"]
    n_blocks_params = [1, 2]

    unique_seqs = np.array(["AAA", "ARA", "AFFFFFA", "FAFAFA", "FFF"])
    seqs2 = np.array(["RRR", "FAFA", "WWWWWWW"])
    cutoff = 8

    if metric in metrics_with_n_blocks:
        for n_blocks in n_blocks_params:
            dist_mat = ir.ir_dist.sequence_dist(
                unique_seqs, metric=metric, cutoff=cutoff, n_jobs=n_jobs, n_blocks=n_blocks
            )
            assert dist_mat.shape == (5, 5)
            dist_mat = ir.ir_dist.sequence_dist(
                unique_seqs, seqs2, metric=metric, cutoff=cutoff, n_jobs=n_jobs, n_blocks=n_blocks
            )
            assert dist_mat.shape == (5, 3)
    else:
        dist_mat = ir.ir_dist.sequence_dist(unique_seqs, metric=metric, cutoff=cutoff, n_jobs=n_jobs)
        assert dist_mat.shape == (5, 5)
        dist_mat = ir.ir_dist.sequence_dist(unique_seqs, seqs2, metric=metric, cutoff=cutoff, n_jobs=n_jobs)
        assert dist_mat.shape == (5, 3)


@pytest.mark.parametrize(
    "test_parameters,test_input,expected_result",
    [
        # test more complex strings with unequal length and set high cutoff such that cutoff is neglected
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["WEKFAPIQCMNR", "RDAIYTCCNKSWEQ", "CWWMFGHTVRI", "GWSZNNHI"]),
            ),
            np.array([[57, 74, 65, 33], [66, 68, 65, 33], [53, 58, 49, 25]]),
        ),
        # test empty input arrays
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (np.array([]), np.array([])),
            np.empty((0, 0)),
        ),
        # test very small input sequences
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 0,
                "ctrim": 0,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (np.array(["A"]), np.array(["C"])),
            np.array([[13]]),
        ),
        # test standard parameters with simple input sequences
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 0, 21], [0, 1, 21], [21, 21, 1]]),
        ),
        # test standard parameters with simple input sequences with second sequences array set to None
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]), None),
            np.array([[1, 0, 21], [0, 1, 21], [21, 21, 1]]),
        ),
        # test with dist_weight set to 0
        (
            {
                "dist_weight": 0,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 1, 9], [1, 1, 9], [9, 9, 1]]),
        ),
        # test with dist_weight set high and cutoff set high to neglect it
        (
            {
                "dist_weight": 30,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 241, 129], [241, 1, 129], [129, 129, 1]]),
        ),
        # test with gap_penalty set to 0
        (
            {
                "dist_weight": 3,
                "gap_penalty": 0,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 0, 13], [0, 1, 13], [13, 13, 1]]),
        ),
        # test with gap_penalty set high and cutoff set high to neglect it
        (
            {
                "dist_weight": 3,
                "gap_penalty": 40,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 25, 93], [25, 1, 93], [93, 93, 1]]),
        ),
        # test with ntrim = 0 and cutoff set high to neglect it
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 0,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 25, 33], [25, 1, 33], [33, 33, 1]]),
        ),
        # test with high ntrim - trimmimg with ntrim is only possible until the beginning of the gap for sequences of unequal length
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 10,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 1, 9], [1, 1, 9], [9, 9, 1]]),
        ),
        # test with ctrim = 0 and cutoff set high to neglect it
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 0,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 25, 21], [25, 1, 21], [21, 21, 1]]),
        ),
        # test with high ctrim - trimmimg with ctrim is only possible until the end of the gap for sequences of unequal length
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 10,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 1, 21], [1, 1, 21], [21, 21, 1]]),
        ),
        # test with fixed_gappos = False and a high cutoff to neglect it
        # AAAAA added at the beginning of the usual sequences to make to difference of min_gappos and max_gappos more significant
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": False,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAAAAAAA", "AAAAAAAAARRAAAA", "AAAAAAANDAAAA"]),
                np.array(["AAAAAAAAAAAAAAA", "AAAAAAAAARRAAAA", "AAAAAAANDAAAA"]),
            ),
            np.array([[1, 25, 33], [25, 1, 33], [33, 33, 1]]),
        ),
        # test with cutoff set to 0
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 0,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        # test with cutoff set high to neglect it
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 25, 21], [25, 1, 21], [21, 21, 1]]),
        ),
        # test more complex strings with multiple cores by setting n_jobs = 2
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 1000,
                "n_jobs": 1,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["WEKFAPIQCMNR", "RDAIYTCCNKSWEQ", "CWWMFGHTVRI", "GWSZNNHI"]),
            ),
            np.array([[57, 74, 65, 33], [66, 68, 65, 33], [53, 58, 49, 25]]),
        ),
        # test with multiple cores by setting n_jobs = 4
        (
            {
                "dist_weight": 3,
                "gap_penalty": 4,
                "ntrim": 3,
                "ctrim": 2,
                "fixed_gappos": True,
                "cutoff": 20,
                "n_jobs": 4,
            },
            (
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
                np.array(["AAAAAAAAAA", "AAAARRAAAA", "AANDAAAA"]),
            ),
            np.array([[1, 0, 21], [0, 1, 21], [21, 21, 1]]),
        ),
    ],
)
def test_tcrdist(test_parameters, test_input, expected_result):
    tcrdist_calculator = TCRdistDistanceCalculator(**test_parameters)
    seq1, seq2 = test_input
    res = tcrdist_calculator.calc_dist_mat(seq1, seq2)
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == expected_result.shape
    assert np.array_equal(res.todense(), expected_result)


def test_tcrdist_reference():
    # test tcrdist against reference implementation
    from . import TESTDATA

    seqs = np.load(TESTDATA / "tcrdist_test_data/tcrdist_WU3k_seqs.npy")
    reference_result = scipy.sparse.load_npz(TESTDATA / "tcrdist_test_data/tcrdist_WU3k_csr_result.npz")

    tcrdist_calculator = TCRdistDistanceCalculator(
        dist_weight=3,
        gap_penalty=4,
        ntrim=3,
        ctrim=2,
        fixed_gappos=True,
        cutoff=15,
        n_jobs=2,
        n_blocks=2,
    )
    res = tcrdist_calculator.calc_dist_mat(seqs, seqs)

    assert np.array_equal(res.data, reference_result.data)
    assert np.array_equal(res.indices, reference_result.indices)
    assert np.array_equal(res.indptr, reference_result.indptr)


def test_hamming_reference():
    # test hamming distance against reference implementation
    from . import TESTDATA

    seqs = np.load(TESTDATA / "hamming_test_data/hamming_WU3k_seqs.npy")
    reference_result = scipy.sparse.load_npz(TESTDATA / "hamming_test_data/hamming_WU3k_csr_result.npz")

    hamming_calculator = HammingDistanceCalculator(2, 2, 2)
    res = hamming_calculator.calc_dist_mat(seqs, seqs)

    assert np.array_equal(res.data, reference_result.data)
    assert np.array_equal(res.indices, reference_result.indices)
    assert np.array_equal(res.indptr, reference_result.indptr)


def test_normalized_hamming():
    hamming_calculator = HammingDistanceCalculator(1, 1, 50, normalize=True)
    seq1 = np.array(["AAAA", "AAB", "AABB", "ABA"])
    seq2 = np.array(["ABB", "ABBB", "ABBB"])
    expected_result = np.array([[0, 0, 0], [34, 0, 0], [0, 26, 26], [34, 0, 0]])
    res = hamming_calculator.calc_dist_mat(seq1, seq2)
    assert isinstance(res, scipy.sparse.csr_matrix)
    assert res.shape == expected_result.shape
    assert np.array_equal(res.todense(), expected_result)


def test_normalized_hamming_reference():
    from . import TESTDATA

    seqs = np.load(TESTDATA / "hamming_test_data/hamming_WU3k_seqs.npy")
    reference_result = scipy.sparse.load_npz(TESTDATA / "hamming_test_data/hamming_WU3k_normalized_csr_result.npz")

    normalized_hamming_calculator = HammingDistanceCalculator(2, 2, 50, normalize=True)
    res = normalized_hamming_calculator.calc_dist_mat(seqs, seqs)

    assert np.array_equal(res.data, reference_result.data)
    assert np.array_equal(res.indices, reference_result.indices)
    assert np.array_equal(res.indptr, reference_result.indptr)


def test_hamming_histogram():
    hamming_calculator = HammingDistanceCalculator(1, 1, 100, normalize=True, histogram=True)
    seqs = np.array(["AAAA", "AA", "AABB", "ABA"])
    row_mins_expected = np.array([50, 100, 50, 100])
    _, _, _, row_mins = hamming_calculator._hamming_mat(seqs=seqs, seqs2=seqs)
    assert np.array_equal(row_mins_expected, row_mins)


def test_hamming_histogram_reference():
    from . import TESTDATA

    seqs = np.load(TESTDATA / "hamming_test_data/hamming_WU3k_seqs.npy")
    hamming_calculator = HammingDistanceCalculator(2, 2, 100, normalize=True, histogram=True)
    row_mins_ref = np.load(TESTDATA / "hamming_test_data/hamming_WU3k_histogram_result.npy")
    _, _, _, row_mins = hamming_calculator._hamming_mat(seqs=seqs, seqs2=seqs)
    assert np.array_equal(row_mins_ref, row_mins)


def test_tcrdist_histogram_not_implemented():
    # Change once histogram is implemented for tcrdist
    with pytest.raises(NotImplementedError, match=None):
        tcrdist_calculator = TCRdistDistanceCalculator(histogram=True)
        seqs = np.array(["AAAA", "AA", "AABB", "ABA"])
        _ = tcrdist_calculator.calc_dist_mat(seqs, seqs)


@pytest.mark.gpu
def test_gpu_hamming_reference():
    # test hamming distance against reference implementation
    from . import TESTDATA

    seqs = np.load(TESTDATA / "hamming_test_data/hamming_WU3k_seqs.npy")
    reference_result = scipy.sparse.load_npz(TESTDATA / "hamming_test_data/hamming_WU3k_csr_result.npz")

    gpu_hamming_calculator = GPUHammingDistanceCalculator(cutoff=2, gpu_n_blocks=5, gpu_block_width=500)
    res = gpu_hamming_calculator.calc_dist_mat(seqs, seqs)

    assert np.array_equal(res.data, reference_result.data)
    assert np.array_equal(res.indices, reference_result.indices)
    assert np.array_equal(res.indptr, reference_result.indptr)
    assert np.array_equal(res.todense(), reference_result.todense())
