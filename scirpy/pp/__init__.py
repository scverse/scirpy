from ..ir_dist import _ir_dist as ir_dist
from ._index_chains import index_chains
from ._merge_adata import merge_airr

__all__ = ["ir_dist", "index_chains", "merge_airr"]

# TODO #356 refer to corresponding docs sections
def merge_with_ir(*args, **kwargs):
    raise NotImplementedError("This function has been removed in v0.13")


def merge_airr_chains(*args, **kwargs):
    raise NotImplementedError(
        "This function has been removed in v0.13 and been replaced by ir.pp.merge_airr with slightly different behavior."
    )
