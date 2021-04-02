from scanpy import read_h5ad
from ._io import (
    read_10x_vdj,
    read_tracer,
    read_bracer,
    read_airr,
    write_airr,
    from_dandelion,
    to_dandelion,
    upgrade_schema,
    DEFAULT_AIRR_CELL_ATTRIBUTES,
    DEFAULT_AIRR_FIELDS,
)
from ._convert_anndata import from_airr_cells, to_airr_cells
from ..util import deprecated
from ._datastructures import AirrCell


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `from_ir_objs. The old version will be removed in a future release. "
)
def from_tcr_objs(*args, **kwargs):
    return from_airr_cells(*args, **kwargs)


@deprecated("This function has been renamed to `from_airr_cells`")
def from_ir_objs(*args, **kwargs):
    return from_airr_cells(*args, **kwargs)


@deprecated("This function has been renamed to `to_airr_cells`.")
def to_ir_objs(*args, **kwargs):
    return to_airr_cells(*args, **kwargs)


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `AirrCell. The old version will be removed in a future release. "
)
def TcrCell(*args, **kwargs):
    return AirrCell(*args, **kwargs)


def AirrChain(*args, **kwargs):
    raise RuntimeError(
        "AirrChain has been removed in v0.7. "
        "Use a AIRR-rearrangement compliant dictionary instead! "
    )


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `AirrChain. The old version will be removed in a future release. "
)
def TcrChain(*args, **kwargs):
    return AirrChain(*args, **kwargs)
