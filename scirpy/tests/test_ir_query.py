from scirpy.tl._ir_query import (
    _reduce_json,
    _reduce_most_frequent,
    _reduce_unique_only,
    ir_query,
    ir_query_annotate,
    ir_query_annotate_df,
)
import numpy as np
import numpy.testing as npt
import pytest
import json


def test_ir_query():
    assert False


@pytest.mark.parametrize(
    "input,expected",
    [
        ([], np.nan),
        ([np.nan], np.nan),
        (["a", "b", np.nan], "ambiguous"),
        (["a", "a"], "a"),
        (["a", "a", np.nan], "a"),
    ],
)
def test_reduce_unique_only(input, expected):
    input = np.array(input, dtype="object")
    npt.assert_equal(_reduce_unique_only(input), expected)


@pytest.mark.parametrize(
    "input,expected",
    [
        ([], np.nan),
        ([np.nan], np.nan),
        (["a", "b", np.nan], "ambiguous"),
        (["a", "a"], "a"),
        (["a", "a", "b"], "a"),
        (["a", "a", "b", "b", "c"], "ambiguous"),
        (["a", "a", "b", "c", np.nan], "a"),
    ],
)
def test_reduce_most_frequent(input, expected):
    input = np.array(input, dtype="object")
    npt.assert_equal(_reduce_most_frequent(input), expected)


@pytest.mark.parametrize(
    "input,expected",
    [
        ([], {}),
        ([np.nan], {}),
        (["a", "b", np.nan], {"a": 1, "b": 1}),
        (["a", "a"], {"a": 2}),
        (["a", "a", "b"], {"a": 2, "b": 1}),
        (["a", "a", "b", "b", "c"], {"a": 2, "b": 2, "c": 1}),
        (["a", "a", "b", "c", np.nan], {"a": 2, "b": 1, "c": 1}),
    ],
)
def test_reduce_json(input, expected):
    input = np.array(input, dtype="object")
    assert json.loads(_reduce_json(input)) == expected


def test_ir_query_annotate_df():
    assert False


def test_ir_query_annotate():
    assert False
