from scanpy import read_h5ad
from ._io import read_10x_vdj, read_tracer, read_airr, read_bracer, read_dandelion
from ._convert_anndata import from_ir_objs, to_ir_objs, to_dandelion
from ..util import deprecated
from ._datastructures import IrCell, IrChain


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `from_ir_objs. The old version will be removed in a future release. "
)
def from_tcr_objs(*args, **kwargs):
    return from_ir_objs(*args, **kwargs)


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `IrCell. The old version will be removed in a future release. "
)
def TcrCell(*args, **kwargs):
    return IrCell(*args, **kwargs)


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `IrChain. The old version will be removed in a future release. "
)
def TcrChain(*args, **kwargs):
    return IrChain(*args, **kwargs)
