from scirpy.tl._ir_query import (
    _reduce_json,
    _reduce_most_frequent,
    _reduce_unique_only,
    ir_query,
    ir_query_annotate,
    ir_query_annotate_df,
)
from scirpy.pp import ir_dist
import numpy as np
import numpy.testing as npt
import pytest
import json
from .fixtures import adata_cdr3, adata_cdr3_2


@pytest.mark.parametrize("metric", ["identity", "levenshtein"])
@pytest.mark.parametrize("key1", [None, "foo"])
@pytest.mark.parametrize("key2", [None, "bar"])
def test_ir_query(adata_cdr3, adata_cdr3_2, metric, key1, key2):
    ir_dist(adata_cdr3, adata_cdr3_2, metric=metric, sequence="aa", key_added=key1)
    ir_query(
        adata_cdr3,
        adata_cdr3_2,
        sequence="aa",
        metric=metric,
        distance_key=key1,
        key_added=key2,
    )

    tmp_key2 = f"ir_query_TESTDB_aa_{metric}" if key2 is None else key2
    assert adata_cdr3.uns[tmp_key2]["distances"].shape == (5, 3)
    assert len(adata_cdr3.uns[tmp_key2]["cell_indices"]) == 5
    assert len(adata_cdr3.uns[tmp_key2]["cell_indices_reference"]) == 3


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
