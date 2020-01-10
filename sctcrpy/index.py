"""
Create an index file listing all specified html files.
"""

from typing import List
import lxml.html
import re
from urllib.parse import urljoin
import os.path
from tempfile import NamedTemporaryFile
from .pandoc import run_pandoc


def _get_title(html_file):
    """Extract the title from an HTML file"""
    t = lxml.html.parse(html_file)
    title = t.find(".//title")
    if title is None:
        return re.sub("\\.html$", "", os.path.basename(html_file))
    else:
        return title.text


def build_index(html_files: List[str], output_file: str, title: str = "Index"):
    """
    Create an index file referencing all specified html files.

    Parameters
    ----------
    html_files
        List of documents to include in the index. The items will
        appear in the same order as in the list.
    output_file
        Path to output file. Can either end with `.md` or `.html`.
        In case of `.html` pandoc will be ran to convert the markdown
        file to HTML.
    title
        H1-title of the page
    """
    output_file = os.path.abspath(output_file)
    md = []
    md.append("# " + title)
    md.extend(
        [
            " * [{name}]({link})".format(
                name=_get_title(f),
                link=os.path.relpath(f, os.path.dirname(output_file)),
            )
            for f in html_files
        ]
    )
    if output_file.endswith(".html"):
        with NamedTemporaryFile("w") as tmp_md_file:
            tmp_md_file.write("\n".join(md))
            tmp_md_file.flush()
            run_pandoc(tmp_md_file.name, output_file)
    else:
        with open(output_file, "w") as md_file:
            md_file.write("\n".join(md))
