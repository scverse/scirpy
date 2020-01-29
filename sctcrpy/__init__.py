"""Python library for single-cell TCR analysis"""
from get_version import get_version

__version__ = get_version(__file__)
del get_version

__author__ = ", ".join(["Gregor Sturm", "Tamas Szabo"])

from scanpy import AnnData, read_h5ad
from .io import read_10x, read_tracer
from . import _preprocessing as pp
from . import _tools as tl
from . import _plotting as pl
