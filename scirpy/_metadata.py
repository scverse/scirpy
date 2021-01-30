"""Metadata. Adapted from https://github.com/theislab/scanpy/pull/1374/."""
import traceback
from pathlib import Path

here = Path(__file__).parent

try:
    from setuptools_scm import get_version
    import pytoml

    proj = pytoml.loads((here.parent / "pyproject.toml").read_text())
    metadata = proj["tool"]["flit"]["metadata"]

    # The `split` removes the "local" version part (PEP440), e.g. from
    # 0.6.1.dev5+ga652c20.d20210130. Leaving the local part, results in the following
    # error message:
    #    WARNING: Built wheel for scirpy is invalid: Wheel has unexpected file name:
    #    expected '0.6.1.dev5+ga652c20.d20210130', got '0.6.1.dev5-ga652c20.d20210130'
    #
    # TODO I expect this to be a temporary issue with either flit or wheel, as
    # this has worked before.
    __version__ = get_version(
        # Allegedly, the parameters from pyproject.toml should be passed automatically.
        # However, this didn't work, so I pass them explicitly here.
        root="..",
        relative_to=__file__,
        **proj["tool"]["setuptools_scm"]
    ).split("+")[0]
    __author__ = metadata["author"]
    __email__ = metadata["author-email"]

except (ImportError, LookupError, FileNotFoundError):
    from ._compat import pkg_metadata

    metadata = pkg_metadata(here.name)
    __version__ = metadata["Version"]
    __author__ = metadata["Author"]
    __email__ = metadata["Author-email"]


def within_flit():
    for frame in traceback.extract_stack():
        if frame.name == "get_docstring_and_version_via_import":
            return True
    return False
