"""Python library for single-cell adaptive immune receptor repertoire (AIRR) analysis"""

from ._metadata import __version__, within_flit

if not within_flit():
    from scanpy import AnnData, read_h5ad
    from . import io
    from . import util
    from . import pp
    from . import pl
    from . import tl
    from . import datasets
    from . import ir_dist
