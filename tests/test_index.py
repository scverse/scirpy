from reportsrender.index import _get_title
from reportsrender import render_rmd, render_papermill
from reportsrender.index import build_index
import shutil
import os
from ._util import tmpwd


def test_get_title(tmpdir):
    """Test that the title is correctly parsed from a HTML document. """
    html_file_1 = tmpdir.join("html1.html")
    html_file_2 = tmpdir.join("html2.html.html")
    html_file_1.write(
        """
        <html>
            <head>
                <title class="foo">testtitle</title>
            </head>
        </html>
        """
    )
    html_file_2.write("Doesn't contain a valid title...")

    assert _get_title(str(html_file_1)) == "testtitle"
    assert _get_title(str(html_file_2)) == "html2.html"


def test_build_index(tmpdir):
    """Test that the index file is correctly built. """
    out_md = tmpdir.join("index.md")
    out_md2 = tmpdir.join("index_2.md")
    out_html = tmpdir.join("index.html")
    html_files = ["html/01_generate_data.rmd.html", "html/02_analyze_data.rmd.html"]

    build_index(html_files, str(out_md))
    build_index(html_files, str(out_md2), title="My Index")
    build_index(html_files, str(out_html))

    for idx in [out_md, out_md2, out_html]:
        content = idx.read()
        assert "First notebook (in R)" in content
        assert "The second notebook" in content
        assert "01_generate_data.rmd.html" in content
        assert "02_analyze_data.rmd.html" in content

    assert "Index</h1>" in out_html.read()
    assert "html/01_generate_data.rmd.html" in out_md2.read()
    assert out_md2.read().startswith("# My Index")


def test_index_paths(tmpdir):
    """Test if the index function correctly computes the
    relative paths to the HTML files.

    Our mock file structure

    ```
        .
        ├── html2
        │   └── html_file2.html
        └── index
            ├── html_file3.html
            ├── html1
            │   └── html_file1.html
            └── index.md
    ```

    """
    index_dir = tmpdir.join("index").mkdir()
    html_file1 = index_dir.mkdir("html1").join("html_file1.html")
    html_file2 = tmpdir.mkdir("html2").join("html_file2.html")
    html_file3 = index_dir.join("html_file3.html")

    shutil.copyfile("html/01_generate_data.rmd.html", html_file1)
    shutil.copyfile("html/02_analyze_data.rmd.html", html_file2)
    shutil.copyfile("html/02_analyze_data.rmd.html", html_file3)

    index_file_rel = index_dir.join("index.rel.md")
    index_file_abs = index_dir.join("index.abs.md")

    build_index(
        [
            os.path.abspath(html_file1),
            os.path.abspath(html_file2),
            os.path.abspath(html_file3),
        ],
        os.path.abspath(index_file_abs),
    )

    with tmpwd(index_dir):
        build_index(
            ["html1/html_file1.html", "../html2/html_file2.html", "html_file3.html"],
            index_file_rel,
        )

    result_rel = index_file_rel.read()
    result_abs = index_file_abs.read()

    assert result_abs == result_rel

    assert "../html2/html_file2.html" in result_abs
    assert "html1/html_file1.html" in result_abs
