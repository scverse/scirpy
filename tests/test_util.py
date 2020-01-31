from sctcrpy._util import _is_na, _is_false, _is_true
import numpy as np
import pandas as pd
import numpy.testing as npt


def test_is_na():
    assert _is_na(None)
    assert _is_na(np.nan)
    assert _is_na("nan")
    assert not _is_na(42)
    assert not _is_na("Foobar")
    assert not _is_na(dict())
    array_test = np.array(["None", "nan", None, np.nan, "foobar"])
    array_expect = np.array([True, True, True, True, False])

    npt.assert_equal(_is_na(array_test), array_expect)
    npt.assert_equal(_is_na(pd.Series(array_test)).values, array_expect)


def test_is_false():
    assert _is_false(False)
    assert _is_false(0)
    assert _is_false("")
    assert _is_false("False")
    assert _is_false("false")
    assert not _is_false(42)
    assert not _is_false(True)
    assert not _is_false("true")
    assert not _is_false("foobar")
    assert not _is_false(np.nan)
    assert not _is_false(None)
    assert not _is_false("nan")
    assert not _is_false("None")
    array_test = np.array(
        ["False", "false", 0, 1, True, False, "true", "Foobar", np.nan, "nan"],
        dtype=object,
    )
    array_test_str = array_test.astype("str")
    array_expect = np.array(
        [True, True, True, False, False, True, False, False, False, False]
    )

    npt.assert_equal(_is_false(array_test), array_expect)
    npt.assert_equal(_is_false(array_test_str), array_expect)
    npt.assert_equal(_is_false(pd.Series(array_test).values), array_expect)
    npt.assert_equal(_is_false(pd.Series(array_test_str).values), array_expect)


def test_is_true():
    assert not _is_true(False)
    assert not _is_true(0)
    assert not _is_true("")
    assert not _is_true("False")
    assert not _is_true("false")
    assert not _is_true("0")
    assert not _is_true(np.nan)
    assert not _is_true(None)
    assert not _is_true("nan")
    assert not _is_true("None")
    assert _is_true(42)
    assert _is_true(True)
    assert _is_true("true")
    assert _is_true("foobar")
    assert _is_true("True")
    array_test = np.array(
        ["False", "false", 0, 1, True, False, "true", "Foobar", np.nan, "nan"],
        dtype=object,
    )
    array_test_str = array_test.astype("str")
    array_expect = np.array(
        [False, False, False, True, True, False, True, True, False, False]
    )

    npt.assert_equal(_is_true(array_test), array_expect)
    npt.assert_equal(_is_true(array_test_str), array_expect)
    npt.assert_equal(_is_true(pd.Series(array_test).values), array_expect)
    npt.assert_equal(_is_true(pd.Series(array_test_str).values), array_expect)
