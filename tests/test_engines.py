"""
Complementing the tests that are specific to the rendering engines
we put some tests here that are executed on all engines
"""

import pytest
from reportsrender import render_papermill, render_rmd
from reportsrender.index import _get_title


ENGINES = [("rmd", render_rmd), ("papermill", render_papermill)]


@pytest.mark.parametrize("engine,render_fun", ENGINES)
def test_title_from_meta(engine, render_fun, tmpdir):
    """Test that the title is correctly obtained from the notebook metadata"""
    if engine == "papermill":
        pytest.skip(
            "for now this does not work with papermill engine due to jqm/pandoc#5905"
        )

    rmd = tmpdir.join("{}.Rmd".format(engine))
    html = tmpdir.join("{}.html".format(engine))

    rmd.write(
        "\n".join(
            [
                "---",
                "title: A Novel Approach to Finding Black Cats in Dark Rooms",
                "---",
                "",
                "Lorem ipsum dolor sit amet. ",
            ]
        )
    )

    render_fun(str(rmd), str(html))

    assert (
        _get_title(str(html)) == "A Novel Approach to Finding Black Cats in Dark Rooms"
    )


@pytest.mark.parametrize("engine,render_fun", ENGINES)
def test_title_from_filename(engine, render_fun, tmpdir):
    """Test that the title falls back to the filename, if no metadata is available"""
    rmd = tmpdir.join("{}.Rmd".format(engine))
    html = tmpdir.join("{}.html".format(engine))

    rmd.write("Lorem ipsum dolor sit amet. ")

    render_fun(str(rmd), str(html))

    assert _get_title(str(html)) == engine
