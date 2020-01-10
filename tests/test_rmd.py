from reportsrender.rmd import render_rmd, _run_rmarkdown
from reportsrender.index import _get_title


def test_run_rmarkdown(tmp_path):
    """Test that running Rmarkdown generates a .md file. """
    in_file = "notebooks/01_generate_data.Rmd"
    md_file = _run_rmarkdown(in_file, tmp_path)
    assert md_file.endswith(".md")
    with open(md_file) as f:
        text = f.read()
        assert "library" in text


def test_render_rmd(tmp_path):
    in_file = "notebooks/01_generate_data.Rmd"
    out_file = tmp_path / "report.html"
    # params = {"input_file": "notebooks/iris.tsv"}
    render_rmd(in_file, out_file, dict())


def test_render_rmd_ipynb(tmp_path):
    in_file = "notebooks/01_generate_data.ipynb"
    out_file = tmp_path / "report.html"
    # params = {"input_file": "notebooks/iris.tsv"}
    render_rmd(in_file, out_file, dict())


def test_render_rmd_py(tmp_path):
    """Render a notebook that contains python code with
    reticulate. """
    in_file = "notebooks/02_analyze_data.Rmd"
    out_file = tmp_path / "report.html"
    params = {"input_file": "notebooks/iris.tsv"}

    render_rmd(in_file, out_file, params)

    result = out_file.read_text()

    assert "ECHO_FALSE" not in result
    assert "RESULTS_HIDE" not in result
    assert "ECHO_TRUE_01" in result
    assert "ECHO_TRUE_02" in result
    assert "RESULTS_SHOW_01" in result
    assert "RESULTS_SHOW_02" in result


def test_render_rmd_title(tmpdir):
    rmd_rmd = tmpdir.join("rmd.Rmd")
    html_rmd = tmpdir.join("rmd.html")

    rmd_rmd.write(
        "\n".join(
            [
                "---",
                "title: A Novel Approach to Finding Black Cats in Dark Rooms (Rmd)",
                "---",
                "",
                "Lorem ipsum dolor sit amet. ",
            ]
        )
    )

    render_rmd(str(rmd_rmd), str(html_rmd))

    assert (
        _get_title(str(html_rmd))
        == "A Novel Approach to Finding Black Cats in Dark Rooms (Rmd)"
    )
