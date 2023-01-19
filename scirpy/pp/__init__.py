from ._merge_adata import merge_with_ir, merge_airr_chains
from ._index_chains import index_chains
from ..util import deprecated
from ..ir_dist import _ir_dist as ir_dist  # NOQA


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `merge_with_ir`. The old version will be removed in a future release. "
)
def merge_with_tcr(adata, adata_tcr, **kwargs):
    return merge_with_ir(adata, adata_tcr, **kwargs)
