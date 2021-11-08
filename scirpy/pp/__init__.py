from ._merge_adata import merge_with_ir, merge_airr_chains
from ..util import deprecated
from ..ir_dist import _ir_dist as ir_dist  # NOQA


@deprecated(
    "Due to added BCR support, this function has been renamed "
    "to `merge_with_ir`. The old version will be removed in a future release. "
)
def merge_with_tcr(adata, adata_tcr, **kwargs):
    return merge_with_ir(adata, adata_tcr, **kwargs)


def tcr_neighbors(*args, dual_tcr="primary_only", **kwargs):
    raise RuntimeError(
        "`tcr_neighbors` has been deprecated in v0.5.0 and "
        "replaced by `ir_dist` in v0.7.0 and its behaviour "
        "has slightly changed. Please read the new docs and update your code. "
    )


def ir_neighbors(*args, **kwargs):
    raise RuntimeError(
        "`ir_neighbors` has been replaced by `ir_dist` in v0.7.0 and its behaviour "
        "has slightly changed. Also, the functions for clonotype definition and "
        "clonotype network definition have been updated. "
        "Please read the updated documentation and update your code! "
    )
