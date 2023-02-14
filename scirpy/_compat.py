from packaging import version

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


def pkg_metadata(package):
    from importlib.metadata import metadata as m

    return m(package)


def pkg_version(package):
    from importlib.metadata import version as v

    return version.parse(v(package))


__all__ = ["Unpack", "pkg_metadata", "pkg_version"]
