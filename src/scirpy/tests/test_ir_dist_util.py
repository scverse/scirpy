"""Test ir_dist._util utility functions"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scipy.sparse as sp

from scirpy.ir_dist._util import DoubleLookupNeighborFinder, merge_coo_matrices, reduce_and, reduce_or


@pytest.fixture
def dlnf_square():
    clonotypes = pd.DataFrame().assign(
        VJ=["A", "B", "C", "D", "A", "C", "D", "G"],
        VDJ=["A", "B", "C", "D", "nan", "D", "nan", "F"],
    )
    dlnf = DoubleLookupNeighborFinder(feature_table=clonotypes)
    dlnf.add_distance_matrix(
        name="test",
        distance_matrix=sp.csr_matrix(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 1, 0],
                [0, 0, 3, 4, 0, 0],
                [0, 0, 4, 3, 0, 0],
                [0, 1, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 6],
            ]
        ),
        labels=np.array(["A", "B", "C", "D", "G", "F"]),
    )
    return dlnf


@pytest.fixture
def dlnf_rectangle():
    clonotypes = pd.DataFrame().assign(
        VJ=["A", "B", "C", "D", "A", "C", "D", "G"],
        VDJ=["A", "B", "C", "D", "nan", "D", "nan", "F"],
    )
    clonotypes2 = pd.DataFrame().assign(
        VJ=["A", "B", "B", "D", "A"],
        VDJ=["A", "B", "nan", "A", "E"],
    )
    dlnf = DoubleLookupNeighborFinder(feature_table=clonotypes, feature_table2=clonotypes2)
    dlnf.add_distance_matrix(
        name="test",
        distance_matrix=sp.csr_matrix(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 3, 0],
                [0, 1, 0, 5],
                [0, 0, 0, 0],
            ]
        ),
        labels=np.array(["A", "B", "C", "D", "G", "F"]),
        labels2=np.array(["A", "B", "D", "E"]),
    )
    return dlnf


@pytest.fixture
def dlnf(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def dlnf_with_lookup(request):
    dlnf = request.getfixturevalue(request.param)
    dlnf.add_lookup_table(feature_col="VJ", distance_matrix="test", name="VJ_test")
    dlnf.add_lookup_table(feature_col="VDJ", distance_matrix="test", name="VDJ_test")
    return dlnf


@pytest.mark.parametrize(
    "mats",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([1, 2, 3, 4], [8, 8, 8, 8], [4, 4, 4, 4]),
        ([0, 0, 0, 0]),
        ([]),
    ],
)
def test_merge_coo_matrices(mats):
    """Test that the fast sum equals the builtin sum"""
    mats = [sp.coo_matrix(m) for m in mats]
    res = merge_coo_matrices(mats)
    # empty lists stay empty list
    expected = sum(mats) if len(mats) else sp.coo_matrix((0, 0))
    try:
        npt.assert_equal(res.toarray(), expected.toarray())
    except AttributeError:
        assert res == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        ([[0, 2, 4, 0], [0, 0, 0, 5]], [0, 2, 4, 5]),
        ([[0, 2, 4, 0], [0, 1, 5, 0]], [0, 1, 4, 0]),
        ([[0, 2, 4, 0], [0, 1, 0, 5]], [0, 1, 4, 5]),
        ([[1, 1, 0, 0], [0, 1, 1, 0]], [1, 1, 1, 0]),
        ([[1, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]], [1, 1, 1, 0]),
        ([[0, 2, np.nan, 0], [0, 1, 0, 5], [7, 0, 0, 3]], [7, 1, 0, 3]),
        ([[np.nan, 2, 4, np.nan], [0, np.nan, 0, 5]], [0, 2, 4, 5]),
        ([[np.nan, 2, 4, np.nan], [np.nan, np.nan, 0, 5]], [np.nan, 2, 4, 5]),
    ],
)
def test_reduce_or(args, expected):
    args = [np.array(a, dtype=np.float16) for a in args]
    expected = np.array(expected, dtype=np.float16)
    npt.assert_equal(reduce_or(*args), expected)


@pytest.mark.parametrize(
    "args,chain_count,expected",
    [
        ([[0, 2, 4, 0], [0, 0, 0, 5]], [2, 2, 2, 2], [0, 0, 0, 0]),
        ([[0, 2, 4, 0], [0, 0, 1, 0]], [2, 2, 2, 2], [0, 0, 4, 0]),
        ([[0, 0, 1, 0], [0, 2, 4, 0]], [2, 2, 2, 2], [0, 0, 4, 0]),
        ([[0, 2, 4, 0], [0, 1, 5, 0]], [2, 2, 2, 2], [0, 2, 5, 0]),
        ([[3, 2, 4, 1], [3, 1, 5, 1]], [0, 1, 2, 3], [0, 0, 5, 0]),
        (
            [[0, 2, 4, 0], [0, 1, 5, 0], [0, 2, 4, 0], [0, 1, 7, 0]],
            [4, 4, 4, 4],
            [0, 2, 7, 0],
        ),
        (
            [[np.nan, np.nan, 4, np.nan], [0, 1, np.nan, np.nan]],
            [1, 1, 1, 0],
            [0, 1, 4, np.nan],
        ),
    ],
)
def test_reduce_and(args, chain_count, expected):
    args = [np.array(a, dtype=np.float16) for a in args]
    expected = np.array(expected, dtype=np.float16)
    chain_count = np.array(chain_count, dtype=int)
    npt.assert_equal(reduce_and(*args, chain_count=chain_count), expected)


@pytest.mark.parametrize(
    "dlnf,feature_col,name,forward_expected,reverse_expected",
    [
        (
            "dlnf_square",
            "VJ",
            "VJ_test",
            np.array([0, 1, 2, 3, 0, 2, 3, 4]),
            {
                0: [1, 0, 0, 0, 1, 0, 0, 0],
                1: [0, 1, 0, 0, 0, 0, 0, 0],
                2: [0, 0, 1, 0, 0, 1, 0, 0],
                3: [0, 0, 0, 1, 0, 0, 1, 0],
                4: [0, 0, 0, 0, 0, 0, 0, 1],
            },
        ),
        (
            "dlnf_square",
            "VDJ",
            "VDJ_test",
            np.array([0, 1, 2, 3, -1, 3, -1, 5]),
            {
                0: [1, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 0, 0, 0, 0, 0, 0],
                2: [0, 0, 1, 0, 0, 0, 0, 0],
                3: [0, 0, 0, 1, 0, 1, 0, 0],
                5: [0, 0, 0, 0, 0, 0, 0, 1],
            },
        ),
        (
            "dlnf_rectangle",
            "VJ",
            "VJ_test",
            np.array([0, 1, 2, 3, 0, 2, 3, 4]),
            {
                0: [1, 0, 0, 0, 1],
                1: [0, 1, 1, 0, 0],
                2: [0, 0, 0, 1, 0],
            },
        ),
        (
            "dlnf_rectangle",
            "VDJ",
            "VDJ_test",
            np.array([0, 1, 2, 3, -1, 3, -1, 5]),
            {
                0: [1, 0, 0, 1, 0],
                1: [0, 1, 0, 0, 0],
                3: [0, 0, 0, 0, 1],
            },
        ),
    ],
    indirect=["dlnf"],
)
def test_dlnf_lookup_table(dlnf, feature_col, name, forward_expected, reverse_expected):
    dlnf.add_lookup_table(feature_col=feature_col, distance_matrix="test", name=name)
    dist_mat, forward, reverse = dlnf.lookups[name]
    assert dist_mat == "test"
    npt.assert_array_equal(forward, forward_expected)
    assert len(reverse.lookup) == len(reverse_expected)
    for (k, v), (k_expected, v_expected) in zip(reverse.lookup.items(), reverse_expected.items(), strict=False):
        assert k == k_expected
        assert list(v.todense().A1) == v_expected


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_square"], indirect=True)
def test_dlnf_lookup(dlnf_with_lookup):
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(4, "VJ_test").todense().A1)
        == [1, 0, 0, 0, 1, 0, 0, 0]
    )
    assert list(dlnf_with_lookup.lookup(1, "VJ_test").todense().A1) == ([0, 2, 0, 0, 0, 0, 0, 1])
    assert (
        list(dlnf_with_lookup.lookup(2, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(5, "VJ_test").todense().A1)
        == [0, 0, 3, 4, 0, 3, 4, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VJ_test").todense().A1)
        == [0, 0, 4, 3, 0, 4, 3, 0]
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_rectangle"], indirect=True)
def test_dlnf_lookup_rect(dlnf_with_lookup):
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(4, "VJ_test").todense().A1)
        == [1, 0, 0, 0, 1]
    )
    assert list(dlnf_with_lookup.lookup(1, "VJ_test").todense().A1) == [0, 2, 2, 0, 0]
    assert (
        list(dlnf_with_lookup.lookup(2, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(5, "VJ_test").todense().A1)
        == [0, 0, 0, 0, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VJ_test").todense().A1)
        == [0, 0, 0, 3, 0]
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_square"], indirect=True)
def test_dlnf_lookup_nan(dlnf_with_lookup):
    assert list(dlnf_with_lookup.lookup(0, "VDJ_test").todense().A1) == ([1, 0, 0, 0, 0, 0, 0, 0])
    assert (
        list(dlnf_with_lookup.lookup(3, "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(5, "VDJ_test").todense().A1)
        == [0, 0, 4, 3, 0, 3, 0, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VDJ_test").todense().A1)
        == [0, 0, 0, 0, 0, 0, 0, 0]
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_rectangle"], indirect=True)
def test_dlnf_lookup_nan_rect(dlnf_with_lookup):
    assert list(dlnf_with_lookup.lookup(0, "VDJ_test").todense().A1) == ([1, 0, 0, 1, 0])
    assert (
        list(dlnf_with_lookup.lookup(3, "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(5, "VDJ_test").todense().A1)
        == [0, 0, 0, 0, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VDJ_test").todense().A1)
        == [0, 0, 0, 0, 0]
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_square", "dlnf_rectangle"], indirect=True)
@pytest.mark.parametrize("clonotype_id", range(8))
def test_dnlf_lookup_with_two_identical_forward_and_reverse_tables(dlnf_with_lookup, clonotype_id):
    npt.assert_array_equal(
        list(dlnf_with_lookup.lookup(clonotype_id, "VJ_test").todense().A1),
        list(dlnf_with_lookup.lookup(clonotype_id, "VJ_test", "VJ_test").todense().A1),
    )
    npt.assert_array_equal(
        list(dlnf_with_lookup.lookup(clonotype_id, "VDJ_test").todense().A1),
        list(dlnf_with_lookup.lookup(clonotype_id, "VDJ_test", "VDJ_test").todense().A1),
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_square"], indirect=True)
def test_dlnf_lookup_with_different_forward_and_reverse_tables(dlnf_with_lookup):
    # if entries don't exist the the other lookup table, should return empty iterator.
    assert list(dlnf_with_lookup.lookup(7, "VDJ_test", "VJ_test").todense().A1) == ([0, 0, 0, 0, 0, 0, 0, 0])

    assert list(dlnf_with_lookup.lookup(7, "VJ_test", "VDJ_test").todense().A1) == ([0, 1, 0, 0, 0, 0, 0, 0])
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test", "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(4, "VJ_test", "VDJ_test").todense().A1)
        == [1, 0, 0, 0, 0, 0, 0, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test", "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VJ_test", "VDJ_test").todense().A1)
        == [0, 0, 4, 3, 0, 3, 0, 0]
    )
    assert list(dlnf_with_lookup.lookup(2, "VDJ_test", "VJ_test").todense().A1) == ([0, 0, 3, 4, 0, 3, 4, 0])
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test", "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VDJ_test", "VJ_test").todense().A1)
        == [0] * 8
    )


@pytest.mark.parametrize("dlnf_with_lookup", ["dlnf_rectangle"], indirect=True)
def test_dlnf_lookup_with_different_forward_and_reverse_tables_rect(dlnf_with_lookup):
    # if entries don't exist the the other lookup table, should return empty iterator.
    assert list(dlnf_with_lookup.lookup(7, "VDJ_test", "VJ_test").todense().A1) == ([0, 0, 0, 0, 0])

    assert list(dlnf_with_lookup.lookup(7, "VJ_test", "VDJ_test").todense().A1) == ([0, 1, 0, 0, 5])
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test", "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(4, "VJ_test", "VDJ_test").todense().A1)
        == [1, 0, 0, 1, 0]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test", "VDJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VJ_test", "VDJ_test").todense().A1)
        == [0, 0, 0, 0, 0]
    )
    assert list(dlnf_with_lookup.lookup(2, "VDJ_test", "VJ_test").todense().A1) == ([0, 0, 0, 0, 0])
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test", "VJ_test").todense().A1)
        == list(dlnf_with_lookup.lookup(6, "VDJ_test", "VJ_test").todense().A1)
        == [0, 0, 0, 0, 0]
    )
