"""Python library for single-cell TCR analysis"""
from get_version import get_version

__version__ = get_version(__file__)
del get_version

__author__ = "Gregor Sturm"

from ._io import read_10x, read_tracer
from . import _preprocessing as pp
from . import _tools as tl
