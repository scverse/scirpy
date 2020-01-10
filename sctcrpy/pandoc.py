"""Helper function for converting documents to html using pandoc. """

import pkg_resources
import os
from subprocess import check_call
from typing import Collection


DEFAULT_RES_PATH = pkg_resources.resource_filename(
    __package__, "templates/adaptive-bootstrap"
)
DEFAULT_TEMPLATE = os.path.join(DEFAULT_RES_PATH, "standalone.html")
DEFAULT_CSS = os.path.join(DEFAULT_RES_PATH, "template.css")


def run_pandoc(
    in_file: str,
    out_file: str,
    res_path: Collection[str] = None,
    template_file: str = None,
    css_file: str = None,
):
    """
    Convert to HTML using pandoc.

    Will create a standalone, self-contained html based on the specified template.

    Parameters
    ----------
    in_file
        path to input file. Can be any format supported by pandoc. The format will be inferred from
        the file extension.
    out_file
        path to output (html) file.
    res_path
        List of pandoc resource paths (pandoc will look here for asset files). If no template_file is provided
        the resource path of the default template will be appended.
    template_file
        path to the pandoc template. Per default, the `adaptive-bootstrap` template
        shipped with this package will be used.
    css_file
        path to the css file used by pandoc. Per default, the css file from the `adaptive-bootstrap` template
        shipped with this package will be used.
    """
    if res_path is None:
        res_path = list()
    if template_file is None:
        template_file = DEFAULT_TEMPLATE
        res_path.append(DEFAULT_RES_PATH)
    if css_file is None:
        css_file = DEFAULT_CSS

    cmd = """
          pandoc \
            +RTS -K512m -RTS \
            {input_file} \
            --output {output_file} \
            --to html5 \
            --email-obfuscation none \
            --self-contained \
            --mathjax \
            --variable 'mathjax-url:https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML' \
            --variable 'lightbox:true' \
            --variable 'thumbnails:true' \
            --variable 'gallery:false' \
            --variable 'cards:true' \
            --standalone \
            --section-divs \
            --table-of-contents \
            --toc-depth 2 \
            --template {html_template} \
            --css {css_file} \
            --resource-path "{res_path}" \
            --highlight-style pygments 
    """
    cmd2 = cmd.format(
        input_file=in_file,
        output_file=out_file,
        html_template=template_file,
        css_file=css_file,
        res_path=":".join(res_path),
    )
    check_call(cmd2, shell=True)
