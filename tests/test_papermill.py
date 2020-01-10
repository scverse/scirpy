#!/usr/bin/env python3
"""
Test the pipeline results:
  * does hiding inputs and outputs work as expected?
"""

from reportsrender.papermill import render_papermill, _remove_cells
import nbformat
from pprint import pprint


def test_remove_cells():
    """Test that removing and keeping outputs works properly. """
    nb = nbformat.from_dict(
        {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {"tags": ["hide_input"]},
                    "source": "# REMOVE_CELL",
                },
                {
                    "cell_type": "markdown",
                    "metadata": {"tags": ["remove_input"]},
                    "source": "# REMOVE_CELL",
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [
                        {
                            "data": {
                                "image/png": "base64",
                                "text/plain": "INCLUDE_OUTPUT_01",
                            }
                        }
                    ],
                    "source": "# INCLUDE_INPUT_01",
                },
                {
                    "cell_type": "code",
                    "metadata": {"tags": ["remove_cell"]},
                    "outputs": [
                        {"data": {"image/png": "base64", "text/plain": "REMOVE_CELL"}}
                    ],
                    "source": "# REMOVE_CELL",
                },
                {
                    "cell_type": "code",
                    "metadata": {"tags": ["hide_output"]},
                    "outputs": [
                        {"data": {"image/png": "base64", "text/plain": "REMOVE_CELL"}}
                    ],
                    "source": "# INCLUDE_INPUT_02",
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 0,
        }
    )

    nb2 = _remove_cells(nb)

    assert nb2 == nb

    str_repr = str(nb)

    pprint(nb)

    assert "INCLUDE_OUTPUT_01" in str_repr
    assert "INCLUDE_INPUT_01" in str_repr
    assert "INCLUDE_INPUT_02" in str_repr
    assert "REMOVE_CELL" not in str_repr


def test_render_papermill(tmp_path):
    in_file = "notebooks/02_analyze_data.Rmd"
    out_file = tmp_path / "report.html"
    params = {"input_file": "notebooks/iris.tsv"}
    render_papermill(in_file, out_file, params)

    result = out_file.read_text()

    assert "ECHO_FALSE" not in result
    assert "RESULTS_HIDE" not in result
    assert "ECHO_TRUE_01" in result
    assert "ECHO_TRUE_02" in result
    assert "RESULTS_SHOW_01" in result
    assert "RESULTS_SHOW_02" in result


def test_render_papermill_ipynb(tmp_path):
    """The same should work for a ipynb input file"""
    in_file = "notebooks/02_analyze_data.ipynb"
    out_file = tmp_path / "report.html"
    params = {"input_file": "notebooks/iris.tsv"}
    render_papermill(in_file, out_file, params)

    result = out_file.read_text()

    assert "ECHO_FALSE" not in result
    assert "RESULTS_HIDE" not in result
    assert "ECHO_TRUE_01" in result
    assert "ECHO_TRUE_02" in result
    assert "RESULTS_SHOW_01" in result
    assert "RESULTS_SHOW_02" in result


def test_render_papermill_md(tmp_path):
    """... and a markdown input file (jupytext format) """
    in_file = "notebooks/02_analyze_data.md"
    out_file = tmp_path / "report.html"
    params = {"input_file": "notebooks/iris.tsv"}
    render_papermill(in_file, out_file, params)

    result = out_file.read_text()

    assert "ECHO_FALSE" not in result
    assert "RESULTS_HIDE" not in result
    assert "ECHO_TRUE_01" in result
    assert "ECHO_TRUE_02" in result
    assert "RESULTS_SHOW_01" in result
    assert "RESULTS_SHOW_02" in result
