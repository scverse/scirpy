"""
Helper functions for the render scripts.
"""

import os
import shlex
import shutil


def _set_cpus(n_cpus):
    """Set environment variables for numba and numpy """
    n_cpus = str(n_cpus)
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_NUM_cpus"] = n_cpus
    os.environ["NUMEXPR_NUM_cpus"] = n_cpus
    os.environ["OMP_NUM_cpus"] = n_cpus
    os.environ["NUMBA_NUM_cpus"] = n_cpus


def _parse_params(params):
    """Parse a whitespace-separated key-value list into a dictionary"""
    return dict(token.split("=") for token in shlex.split(params))


def mergefolders(root_src_dir, root_dst_dir):
    """recursively merge two folders including subfolders.

    Overwrites files in destination should they already exist.

    From: https://lukelogbook.tech/2018/01/25/merging-two-folders-in-python/
    """
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)
