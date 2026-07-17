from importlib.metadata import version

from . import datasets, get, io, ir_dist, pl, pp, tl, util

__all__ = ["datasets", "get", "io", "ir_dist", "pl", "pp", "tl", "util"]

__version__ = version("scirpy")
