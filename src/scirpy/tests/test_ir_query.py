import json

import numpy as np
import numpy.testing as npt
import pytest
from mudata import MuData

from scirpy.pp import ir_dist
from scirpy.tl._ir_query import (
    _reduce_json,
    _reduce_most_frequent,
    _reduce_unique_only,
    ir_query,
    ir_query_annotate,
    ir_query_annotate_df,
)
from scirpy.util import read_cell_indices


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
    tmp_ad = adata_cdr3.mod["airr"] if isinstance(adata_cdr3, MuData) else adata_cdr3

    cell_indices = read_cell_indices(tmp_ad.uns[tmp_key2]["cell_indices"])
    cell_indices_reference = read_cell_indices(tmp_ad.uns[tmp_key2]["cell_indices_reference"])

    assert tmp_ad.uns[tmp_key2]["distances"].shape == (4, 3)
    assert len(cell_indices) == 4
    assert len(cell_indices_reference) == 3


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


@pytest.fixture
def query_reference(adata_define_clonotype_clusters):
    query = adata_define_clonotype_clusters[["cell2", "cell3", "cell4", "cell10"], :].copy()
    reference = adata_define_clonotype_clusters.copy()
    reference.obs["some_annotation"] = reference.obs_names.str.upper()
    uns_ = reference.mod["airr"].uns if isinstance(reference, MuData) else reference.uns
    uns_["DB"] = {"name": "TESTDB"}

    return query, reference


@pytest.mark.parametrize(
    "same_v_gene,match_columns,expected",
    [
        (
            False,
            None,
            [
                ("cell2", "CELL1"),
                ("cell2", "CELL2"),
                ("cell2", "CELL5"),
                ("cell2", "CELL6"),
                ("cell2", "CELL7"),
                ("cell2", "CELL8"),
                ("cell3", "CELL3"),
                ("cell3", "CELL4"),
                ("cell4", "CELL3"),
                ("cell4", "CELL4"),
            ],
        ),
        (
            True,
            None,
            [
                ("cell2", "CELL1"),
                ("cell2", "CELL2"),
                ("cell2", "CELL5"),
                ("cell2", "CELL6"),
                ("cell2", "CELL7"),
                ("cell2", "CELL8"),
                ("cell3", "CELL3"),
                ("cell4", "CELL4"),
            ],
        ),
        (
            False,
            ["receptor_type"],
            [
                ("cell2", "CELL1"),
                ("cell2", "CELL2"),
                ("cell2", "CELL6"),
                ("cell2", "CELL7"),
                ("cell2", "CELL8"),
                ("cell3", "CELL3"),
                ("cell4", "CELL4"),
            ],
        ),
    ],
)
def test_ir_query_annotate_df(query_reference, same_v_gene, match_columns, expected):
    query, reference = query_reference
    ir_dist(query, reference, sequence="aa", metric="identity")
    ir_query(
        query,
        reference,
        sequence="aa",
        metric="identity",
        receptor_arms="VJ",
        dual_ir="primary_only",
        same_v_gene=same_v_gene,
        match_columns=match_columns,
    )

    res = ir_query_annotate_df(
        query,
        reference,
        sequence="aa",
        metric="identity",
        include_query_cols=[],
        include_ref_cols=["some_annotation"],
    )

    actual = list(res["some_annotation"].items())
    print(actual)

    assert actual == expected


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (
            "unique-only",
            [
                ("cell2", "ambiguous"),
                ("cell3", "CELL3"),
                ("cell4", "CELL4"),
                ("cell10", np.nan),
            ],
        ),
        (
            "most-frequent",
            [
                ("cell2", "ambiguous"),
                ("cell3", "CELL3"),
                ("cell4", "CELL4"),
                ("cell10", np.nan),
            ],
        ),
        (
            "json",
            [
                (
                    "cell2",
                    json.dumps({"CELL1": 1, "CELL2": 1, "CELL6": 1, "CELL7": 1, "CELL8": 1}),
                ),
                ("cell3", json.dumps({"CELL3": 1})),
                ("cell4", json.dumps({"CELL4": 1})),
                ("cell10", np.nan),
            ],
        ),
    ],
)
def test_ir_query_annotate(query_reference, strategy, expected):
    query, reference = query_reference
    ir_dist(query, reference, sequence="aa", metric="identity")
    ir_query(
        query,
        reference,
        sequence="aa",
        metric="identity",
        receptor_arms="VJ",
        dual_ir="primary_only",
        same_v_gene=True,
        match_columns=["receptor_type"],
    )

    ir_query_annotate(
        query,
        reference,
        sequence="aa",
        metric="identity",
        include_ref_cols=["some_annotation"],
        strategy=strategy,
    )

    key = "airr:some_annotation" if isinstance(query, MuData) else "some_annotation"
    actual = list(query.obs[key].items())
    print(actual)

    assert actual == expected
