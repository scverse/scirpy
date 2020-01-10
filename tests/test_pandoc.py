from reportsrender.pandoc import run_pandoc


def test_run_pandoc(tmp_path):
    res_html = tmp_path / "res_ipynb.html"
    res_md = tmp_path / "res_md.html"
    run_pandoc("notebooks/01_generate_data.ipynb", res_html)
    run_pandoc("notebooks/01_generate_data.md", res_md)

    assert "Export data" in res_html.read_text()
    assert "Export data" in res_md.read_text()
    assert (
        '<div id="TOC"' in res_md.read_text()
    ), "table of contents exists when starting from markdown. "
    assert (
        '<div id="TOC"' in res_html.read_text()
    ), "table of contents exists when starting from ipynb. "
