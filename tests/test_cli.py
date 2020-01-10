from reportsrender.cli import main
import pytest
import docopt
import os
import shutil
from ._util import tmpwd


def test_no_args():
    with pytest.raises(docopt.DocoptExit):
        main([])


def test_render(tmp_path):
    out_file = tmp_path / "output.html"
    main(["notebooks/01_generate_data.Rmd", str(out_file)])
    result = out_file.read_text()
    assert "First notebook" in result


def test_render_options(tmp_path):
    out_file = tmp_path / "output.html"
    main(
        [
            "notebooks/02_analyze_data.Rmd",
            str(out_file),
            "--engine=papermill",
            "--cpus=2",
            "--params='input_file=notebooks/iris.tsv'",
        ]
    )
    result = out_file.read_text()

    assert (
        "<pre><code>## The 0th Fibonacci number is 0" not in result
    ), "Papermill does not ouput text in <pre><code> tags and does not start with '##'."
    assert "he 0th Fibonacci number is 0" in result


def test_index_no_args():
    with pytest.raises(docopt.DocoptExit):
        main(["index"])


@pytest.mark.parametrize("filename", ["the_index.html", "index.md"])
def test_index_paths(filename, tmpdir):
    """Test if the index subcommand works"""
    index_dir = tmpdir.mkdir("index")
    html_file1 = index_dir.join("html_file1.html")
    html_file2 = index_dir.join("html_file2.html")

    shutil.copyfile("html/01_generate_data.rmd.html", html_file1)
    shutil.copyfile("html/02_analyze_data.rmd.html", html_file2)

    index_file = index_dir.join(filename)

    with tmpwd(index_dir):
        main(
            [
                "index",
                "html_file1.html",
                "html_file2.html",
                "--index={}".format(index_file),
                "--title='A cool title'",
            ]
        )

    result = index_file.read()

    assert "html_file2.html" in result
    assert "html_file1.html" in result
    assert "A cool title" in result
    assert "The second notebook" in result
