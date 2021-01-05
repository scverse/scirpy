"""Test ir_dist._util utility functions"""

from scirpy.ir_dist._util import SetDict, DoubleLookupNeighborFinder
import pytest


@pytest.fixture
def set_dict():
    return SetDict(foo=1, bar=4, baz=9)


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
