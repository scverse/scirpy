"""Tests for deprecated functions and function arguments.

Deprecations are declared using the decorators from :mod:`scverse_misc`. The tests here make sure
that a `FutureWarning` is raised when (and only when) deprecated functionality is used, and that the
deprecated functionality still behaves as documented.
"""

import warnings

import pandas.testing as pdt
import pytest

import scirpy as ir
from scirpy.ir_dist.metrics import (
    AlignmentDistanceCalculator,
    FastAlignmentDistanceCalculator,
    LevenshteinDistanceCalculator,
)

from . import TESTDATA


def test_tcr_dist_deprecated():
    with pytest.warns(FutureWarning, match="renamed to `sequence_dist`"):
        actual = ir.ir_dist.tcr_dist(["AAA", "AAB"], metric="identity", cutoff=0)
    expected = ir.ir_dist.sequence_dist(["AAA", "AAB"], metric="identity", cutoff=0)
    assert (actual != expected).nnz == 0


def test_block_size_deprecated():
    """The object-level `block_size` argument is ignored since v0.15.0"""
    with pytest.warns(FutureWarning, match="argument block_size is deprecated"):
        LevenshteinDistanceCalculator(2, block_size=50)
    with pytest.warns(FutureWarning, match="argument block_size is deprecated"):
        FastAlignmentDistanceCalculator(block_size=50)


def _assert_no_deprecation_warning(func, arg: str, *args, **kwargs):
    """Assert that calling `func` does not raise a deprecation warning about `arg`"""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", FutureWarning)
        result = func(*args, **kwargs)
    assert not [w for w in record if arg in str(w.message)]
    return result


@pytest.mark.parametrize("calculator", [LevenshteinDistanceCalculator, FastAlignmentDistanceCalculator])
def test_no_spurious_block_size_warning(calculator):
    """No warning must be raised if `block_size` is not specified"""
    _assert_no_deprecation_warning(calculator, "block_size")


def test_alignment_distance_calculator_deprecated():
    with pytest.warns(FutureWarning, match="NeedlemanWunschDistanceCalculator"):
        AlignmentDistanceCalculator()


def test_clip_at_deprecated(adata_clonotype):
    """`clip_at = N` is equivalent to `breakpoints = (1, ..., N - 1)`"""
    with pytest.warns(FutureWarning, match="argument clip_at is deprecated"):
        actual = ir.tl.clonal_expansion(adata_clonotype, clip_at=3, inplace=False)
    expected = ir.tl.clonal_expansion(adata_clonotype, breakpoints=(1, 2), inplace=False)
    pdt.assert_series_equal(actual, expected)


def test_clip_at_deprecated_plotting(adata_clonotype):
    with pytest.warns(FutureWarning, match="argument clip_at is deprecated"):
        ir.pl.clonal_expansion(adata_clonotype, groupby="group", clip_at=3)


def test_no_spurious_clip_at_warning(adata_clonotype):
    """`pl.clonal_expansion` must not warn just because it wraps `tl.clonal_expansion`"""
    _assert_no_deprecation_warning(ir.pl.clonal_expansion, "clip_at", adata_clonotype, groupby="group")


def test_include_fields_deprecated():
    with pytest.warns(FutureWarning, match="argument include_fields is deprecated"):
        ir.io.read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv", include_fields=None)


def test_use_umi_count_col_deprecated():
    with pytest.warns(FutureWarning, match="argument use_umi_count_col is deprecated"):
        ir.io.read_airr(TESTDATA / "airr/rearrangement_tra.tsv", use_umi_count_col=True)
