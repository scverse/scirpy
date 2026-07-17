"""Images for plot functions

Adapted from scanpy (c) Philipp Angerer
"""

import re
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options


def _strip_roles(text: str):
    """Remove all roles in the format :foo:`bar` from a text"""
    regex = re.compile(r":.*:`(.*)`")
    return regex.sub(r"\1", text)


def insert_function_images(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Options,
    lines: list[str],
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
    function_description = _strip_roles(lines[0])
    lines[0:0] = [
        f""":rawhtml:`{function_description}<br />`<img src="{app.config.api_rel_dir}/{path.name}" style="width: 300px" />`""",
        "",
    ]


def setup(app: Sphinx):
    # Directory to search for the images
    app.add_config_value("api_dir", Path(), "env")
    # relative path to the directory with the images from the document where they are
    # included.
    app.add_config_value("api_rel_dir", "", "env")
    app.connect("autodoc-process-docstring", insert_function_images)
