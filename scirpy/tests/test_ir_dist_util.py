"""Test ir_dist._util utility functions"""

from scirpy.ir_dist._util import SetDict, DoubleLookupNeighborFinder
import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd
import numpy.testing as npt


@pytest.fixture
def set_dict():
    return SetDict(foo=1, bar=4, baz=9)


@pytest.fixture
def dlnf():
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
def dlnf_with_lookup(dlnf):
    dlnf.add_lookup_table(feature_col="VJ", distance_matrix="test", name="VJ_test")
    dlnf.add_lookup_table(feature_col="VDJ", distance_matrix="test", name="VDJ_test")
    return dlnf


def test_set_dict_init():
    expected = {"foo": 1, "bar": 4, "baz": 9}

    assert SetDict((("foo", 1), ("bar", 4)), baz=9).store == expected
    assert SetDict({"foo": 1, "bar": 4}, baz=9).store == expected
    assert SetDict(foo=1, bar=4, baz=9).store == expected


def test_set_dict_get(set_dict):
    assert set_dict["foo"] == set_dict.get("foo") == 1
    with pytest.raises(KeyError):
        set_dict["does_not_exist"]


def test_set_dict_set(set_dict):
    set_dict["foo"] = 2
    set_dict["ipsum"] = None
    assert set_dict.store == {"foo": 2, "bar": 4, "baz": 9, "ipsum": None}


def test_set_dict_del(set_dict):
    del set_dict["foo"]
    del set_dict["baz"]
    assert set_dict == {"bar": 4}


def test_set_dict_len(set_dict):
    assert len(set_dict) == 3


def test_set_dict_iter(set_dict):
    assert list(set_dict) == ["foo", "bar", "baz"]
    assert list(set_dict.keys()) == ["foo", "bar", "baz"]
    assert list(set_dict.values()) == [1, 4, 9]
    assert list(set_dict.items()) == [("foo", 1), ("bar", 4), ("baz", 9)]


def test_set_dict_eq(set_dict):
    assert set_dict == {"foo": 1, "bar": 4, "baz": 9}
    assert set_dict != {"foo": 1, "bar": 5, "baz": 9}
    assert set_dict == set_dict
    assert set_dict == SetDict({"foo": 1, "bar": 4, "baz": 9})


def test_set_dict_id(set_dict):
    assert set_dict | set_dict == set_dict
    assert set_dict | SetDict() == set_dict
    assert SetDict() | set_dict == set_dict
    assert SetDict() | SetDict() == SetDict()
    with pytest.raises(NotImplementedError):
        assert set_dict | set() == set_dict
    with pytest.raises(NotImplementedError):
        assert set() | set_dict == set_dict

    assert set_dict & set_dict == set_dict
    assert set_dict & SetDict() == SetDict()
    assert SetDict() & set_dict == SetDict()
    assert set_dict & set() == SetDict()
    assert set() & set_dict == SetDict()
    assert SetDict() & SetDict() == SetDict()


@pytest.mark.parametrize(
    "o1,o2,expected",
    [
        (SetDict(foo=2, bar=4), SetDict(baz=5), {"foo": 2, "bar": 4, "baz": 5}),
        (SetDict(foo=2, bar=4), set(["baz"]), NotImplementedError),
        (SetDict(foo=2, bar=4), SetDict(foo=1, bar=5), {"foo": 1, "bar": 4}),
        (SetDict(foo=2, bar=4), SetDict(baz=5, foo=1), {"bar": 4, "baz": 5, "foo": 1}),
        (SetDict(foo=np.nan, bar=4), SetDict(foo=5, bar=np.nan), {"foo": 5, "bar": 4}),
    ],
)
def test_set_dict_or(o1, o2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            o1 | o2
    else:
        assert o1 | o2 == expected


@pytest.mark.parametrize(
    "o1,o2,expected",
    [
        (SetDict(foo=2, bar=4), SetDict(baz=5), {}),
        (SetDict(foo=2, bar=4), set(["baz"]), {}),
        (SetDict(foo=2, bar=4), set(["bar"]), {"bar": 4}),
        (set(["bar"]), SetDict(foo=2, bar=4), {"bar": 4}),
        (SetDict(foo=2, bar=4), SetDict(foo=1, bar=5), {"foo": 2, "bar": 5}),
        (SetDict(foo=2, bar=4), SetDict(baz=5, foo=1), {"foo": 2}),
        (SetDict(foo=np.nan, bar=4), SetDict(foo=5, bar=np.nan), {"foo": 5, "bar": 4}),
    ],
)
def test_set_dict_and(o1, o2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            o1 & o2
    else:
        assert o1 & o2 == expected


def test_dlnf_lookup_table(dlnf):
    dlnf.add_lookup_table(feature_col="VJ", distance_matrix="test", name="VJ_test")
    dist_mat, forward, reverse = dlnf.lookups["VJ_test"]
    assert dist_mat == "test"
    npt.assert_array_equal(forward, np.array([0, 1, 2, 3, 0, 2, 3, 4]))
    assert reverse == {
        0: [0, 4],
        1: [1],
        2: [2, 5],
        3: [3, 6],
        4: [7],
    }

    dlnf.add_lookup_table(feature_col="VDJ", distance_matrix="test", name="VDJ_test")
    dist_mat, forward, reverse = dlnf.lookups["VDJ_test"]
    assert dist_mat == "test"
    npt.assert_array_equal(forward, np.array([0, 1, 2, 3, np.nan, 3, np.nan, 5]))
    assert reverse == {
        0: [0],
        1: [1],
        2: [2],
        3: [3, 5],
        np.nan: [4, 6],
        5: [7],
    }


def test_dlnf_lookup(dlnf_with_lookup):
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test"))
        == list(dlnf_with_lookup.lookup(4, "VJ_test"))
        == [(0, 1), (4, 1)]
    )
    assert list(dlnf_with_lookup.lookup(1, "VJ_test")) == [(1, 2), (7, 1)]
    assert (
        list(dlnf_with_lookup.lookup(2, "VJ_test"))
        == list(dlnf_with_lookup.lookup(5, "VJ_test"))
        == [(2, 3), (5, 3), (3, 4), (6, 4)]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test"))
        == list(dlnf_with_lookup.lookup(6, "VJ_test"))
        == [(2, 4), (5, 4), (3, 3), (6, 3)]
    )


@pytest.mark.parametrize(
    "nan_dist,expected",
    [
        (0, [(4, 0), (6, 0)]),
        (42, [(4, 42), (6, 42)]),
        (np.nan, [(4, np.nan), (6, np.nan)]),
    ],
)
def test_dlnf_lookup_nan(dlnf_with_lookup, nan_dist, expected):
    dlnf_with_lookup.nan_dist = nan_dist
    assert list(dlnf_with_lookup.lookup(0, "VDJ_test")) == [(0, 1)]
    assert (
        list(dlnf_with_lookup.lookup(3, "VDJ_test"))
        == list(dlnf_with_lookup.lookup(5, "VDJ_test"))
        == [(2, 4), (3, 3), (5, 3)]
    )
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test"))
        == list(dlnf_with_lookup.lookup(6, "VDJ_test"))
        == expected
    )


@pytest.mark.parametrize("clonotype_id", range(8))
def test_dnlf_lookup_with_two_identical_forward_and_reverse_tables(
    dlnf_with_lookup, clonotype_id
):
    npt.assert_array_equal(
        list(dlnf_with_lookup.lookup(clonotype_id, "VJ_test")),
        list(dlnf_with_lookup.lookup(clonotype_id, "VJ_test", "VJ_test")),
    )
    npt.assert_array_equal(
        list(dlnf_with_lookup.lookup(clonotype_id, "VDJ_test")),
        list(dlnf_with_lookup.lookup(clonotype_id, "VDJ_test", "VDJ_test")),
    )


def test_dlnf_lookup_with_different_forward_and_reverse_tables(dlnf_with_lookup):
    # if entries don't exist the the other lookup table, should return empty iterator.
    assert list(dlnf_with_lookup.lookup(7, "VDJ_test", "VJ_test")) == []

    assert list(dlnf_with_lookup.lookup(7, "VJ_test", "VDJ_test")) == [(1, 1)]
    assert (
        list(dlnf_with_lookup.lookup(0, "VJ_test", "VDJ_test"))
        == list(dlnf_with_lookup.lookup(4, "VJ_test", "VDJ_test"))
        == [(0, 1)]
    )
    assert (
        list(dlnf_with_lookup.lookup(3, "VJ_test", "VDJ_test"))
        == list(dlnf_with_lookup.lookup(6, "VJ_test", "VDJ_test"))
        == [(2, 4), (3, 3), (5, 3)]
    )
    assert list(dlnf_with_lookup.lookup(2, "VDJ_test", "VJ_test")) == [
        (2, 3),
        (5, 3),
        (3, 4),
        (6, 4),
    ]
    assert (
        list(dlnf_with_lookup.lookup(4, "VDJ_test", "VJ_test"))
        == list(dlnf_with_lookup.lookup(6, "VDJ_test", "VJ_test"))
        == []
    )
