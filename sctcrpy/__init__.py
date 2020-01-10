"""Execute and render notebooks as HTML reports. """
from get_version import get_version
import sys

__version__ = get_version(__file__)
del get_version

__author__ = "Gregor Sturm"


# To avoid that all dependencies need to be installed at build
# time:
if "flit" not in sys.argv[0]:
    from .papermill import render_papermill
    from .rmd import render_rmd
    from .pandoc import run_pandoc
    from .index import build_index
