"""Python library for single-cell adaptive immune receptor repertoire (AIRR) analysis"""

from ._metadata import __version__, __author__, __email__, within_flit

if not within_flit():
    from scanpy import AnnData, read_h5ad
    from . import io
    from . import util
    from . import _preprocessing as pp
    from . import _tools as tl
    from . import _plotting as pl
    from . import datasets
    from . import ir_dist
