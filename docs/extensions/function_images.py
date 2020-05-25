"""Images for plot functions

Adapted from scanpy (c) Philipp Angerer
"""
from pathlib import Path
from typing import List, Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options


def insert_function_images(
    app: Sphinx, what: str, name: str, obj: Any, options: Options, lines: List[str],
):
    for ext in ["png", "svg"]:
        path = app.config.api_dir / f"{name}.{ext}"
        if path.is_file():
            break

    if what != "function" or not path.is_file():
        return

    # all lines but the first will be ignored.
    # Currently, the only way out I can see would be to use raw html.
    # Sphinx doesn't handle copying images then, why we put them in the `_static` dir.
    # This, ultimately, is a hack and a cleaner solution would be welcome.
    lines[0:0] = [
        f""":raw:html:`{lines[0]}<br />`<img src="{app.config.api_rel_dir}/{path.name}" style="width: 300px" />`""",
        "",
    ]


def setup(app: Sphinx):
    # Directory to search for the images
    app.add_config_value("api_dir", Path(), "env")
    # relative path to the directory with the images from the document where they are
    # included.
    app.add_config_value("api_rel_dir", "", "env")
    app.connect("autodoc-process-docstring", insert_function_images)
