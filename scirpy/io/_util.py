from collections import Counter
from typing import Callable
from scanpy import logging
from functools import wraps

doc_working_model = """\

.. note::
    Reading data into *Scirpy* has the following constraints:
     * Each cell can have up to four productive chains chains (:term:`Dual IR`):
       two :term:`VJ<V(D)J>` and two :term:`VDJ<V(D)J>` chains. 
     * Excess chains are ignored (those with lowest read count/:term:`UMI` count)
       and cells flagged as :term:`Multichain-cell`.
     * Non-productive chains are ignored. 
     * Chain loci must be valid :term:`IGMT locus names<Chain locus>`.
     * Excess chains, non-productive chains, chains without a CDR3 sequence, 
       or chains with invalid loci are serialized to JSON and stored in the 
       `extra_chains` column. They are not used by scirpy except when exporting 
       the `AnnData` object to AIRR format. 

    For more information, see :ref:`receptor-model`.
"""


class _IOLogger:
    """Logger wrapper that prints identical messages only once"""

    def __init__(self):
        self._warnings = Counter()

    def warning(self, message):
        if not self._warnings[message]:
            logging.warning(message)  # type: ignore

        self._warnings[message] += 1


def _check_anndata_upgrade_schema(adata):
    """Check if `adata` uses the latest scirpy schema.

    Raises ValueError if it doesn't"""
    if "has_ir" in adata.obs.columns:
        # I would actually only use `scirpy_version` for the check, but
        # there might be cases where it gets lost (e.g. when rebuilding AnnData).
        # If a `v_call` is present, that's a safe sign that it is the AIRR schema, too
        if (
            "IR_VJ_1_v_call" not in adata.obs.columns
            and "IR_VDJ_1_v_call" not in adata.obs.columns
        ) and "scirpy_version" not in adata.uns:
            raise ValueError(
                "Scirpy has updated the the format of `adata.obs` in v0.7. "
                "Please run `ir.io.upgrade_schema(adata)` to update your AnnData "
                "object to the latest version. \n"
                "If you are sure your schema is up-to-date, you can override "
                "this message by setting `adata.uns['scirpy_version'] = '0.7'`"
            )


def _check_upgrade_schema(check_args=(0,)) -> Callable:
    """Decorator that checks that anndata uses the latest schema"""

    def check_upgrade_schema_decorator(f):
        @wraps(f)
        def check_wrapper(*args, **kwargs):
            for i in check_args:
                _check_anndata_upgrade_schema(args[i])
            return f(*args, **kwargs)

        return check_wrapper

    return check_upgrade_schema_decorator
