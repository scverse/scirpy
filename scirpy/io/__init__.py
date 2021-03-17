from scanpy import read_h5ad
from ._io import read_10x_vdj, read_tracer, read_airr, read_bracer
from ._convert_anndata import from_ir_objs, to_ir_objs
from ..util import deprecated
from ._datastructures import AirrCell, AirrChain


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `from_ir_objs. The old version will be removed in a future release. "
)
def from_tcr_objs(*args, **kwargs):
    return from_ir_objs(*args, **kwargs)


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `AirrCell. The old version will be removed in a future release. "
)
def TcrCell(*args, **kwargs):
    return AirrCell(*args, **kwargs)


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `AirrChain. The old version will be removed in a future release. "
)
def TcrChain(*args, **kwargs):
    return AirrChain(*args, **kwargs)
