"""reportsrender

Execute and render a jupyter/Rmarkdown notebook.
The `index` subcommand generates an index html
or markdown file that links to html documents.

Usage:
  reportsrender <notebook> <out_file> [--cpus=<cpus>] [--params=<params>] [--engine=<engine>]
  reportsrender index [--index=<index_file>] [--title=<title>] [--] <html_files>...
  reportsrender --help

Arguments and options:
  <notebook>            Input notebook to be executed. Can be any format supported by jupytext.
  <out_file>            Output HTML file.
  -h --help             Show this screen.
  --cpus=<cpus>         Number of CPUs to use for Numba/Numpy/OpenBLAS/MKL [default: 1]
  --params=<params>     space-separated list of key-value pairs that will be passed
                        to papermill/Rmarkdown.
                        E.g. "input_file=dir/foo.txt output_file=dir2/bar.html"
  --engine=<engine>     Engine to execute the notebook. [default: auto]

Arguments and options of the `index` subcommand:
  <html_files>          List of HTML files that will be included in the index. The tool
                        will generate relative links from the index file to these files.
  --index=<index_file>  Path to the index file that will be generated. Will be
                        overwritten if exists. Will auto-detect markdown (.md) and
                        HTML (.html) format based on the extension. [default: index.html]
  --title=<title>       Headline of the index. [default: Index]

Possible engines are:
  auto                  Use `rmd` engine for `*.Rmd` files, papermill otherwise.
  rmd                   Use `rmarkdown` to execute the notebook. Supports R and
                        python (through reticulate)
  papermill             Use `papermill` to execute the notebook. Works for every
                        kernel available in the jupyter installation.
"""

from docopt import docopt
from .util import _set_cpus, _parse_params
from .papermill import render_papermill
from .rmd import render_rmd
from .index import build_index
import sys


def _run_index(arguments):
    if arguments["<html_files>"] is None:
        print("Please specify at least one file to include in the index. ")
    build_index(
        html_files=arguments["<html_files>"],
        output_file=arguments["--index"],
        title=arguments["--title"],
    )


def _run_render(arguments):
    params = (
        _parse_params(arguments["--params"])
        if arguments["--params"] is not None
        else dict()
    )
    _set_cpus(arguments["--cpus"])

    engine = arguments["--engine"]
    if engine == "auto":
        engine = "rmd" if arguments["<notebook>"].endswith(".Rmd") else "papermill"

    if engine == "rmd":
        render_rmd(arguments["<notebook>"], arguments["<out_file>"], params)
    elif engine == "papermill":
        render_papermill(arguments["<notebook>"], arguments["<out_file>"], params)
    else:
        print(
            "Please specify a valid engine. See `reportsrender --help` for more details. ",
            file=sys.stderr,
        )


def main(argv=None):
    """
    Execute reportsrender.

    Parameters
    ----------
    argv: command line args. Can be overridden for testing purposes.
    """
    arguments = docopt(__doc__, argv=argv)
    if arguments["index"]:
        _run_index(arguments)
    else:
        _run_render(arguments)


if __name__ == "__main__":
    main()
