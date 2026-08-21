"""Workaround: silence spurious ``unknown role name: external+...`` warnings.

To place ``:rtype:``, ``sphinx_autodoc_typehints`` runs a throwaway reST parse of every
docstring without intersphinx's role dispatcher installed, so ``:external+...:`` roles
warn. mudata uses one in the ``read_h5mu`` docstring, which we re-export from
`scirpy.io`. The parse is discarded and the role resolves fine in the real parse, but
the warning is untyped, so ``suppress_warnings`` cannot target it and it fails ``-W``.

Remove once fixed upstream (sphinx-autodoc-typehints 3.13.2 / sphinx 9.1).
"""

import logging
import re

SPURIOUS_EXTERNAL_ROLE = re.compile(r"unknown role name: external[:+]")


class IgnoreSpuriousExternalRole(logging.Filter):  # noqa D102
    def filter(self, record):  # noqa D102
        return not SPURIOUS_EXTERNAL_ROLE.match(record.getMessage())


def setup(app):
    for handler in logging.getLogger("sphinx").handlers:
        # Insert first: Sphinx's own WarningSuppressor is what bumps the count `-W` fails on.
        handler.filters.insert(0, IgnoreSpuriousExternalRole())
    return {"parallel_read_safe": True, "parallel_write_safe": True}
