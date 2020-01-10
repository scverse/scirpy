import contextlib
import os


@contextlib.contextmanager
def tmpwd(dir):
    """Temporarily switch to a working directory"""
    curdir = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(curdir)
