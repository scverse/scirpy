import scipy.sparse
import parasail
from multiprocessing import Pool
import itertools
from anndata import AnnData
from typing import Union
import pandas as pd
import numpy as np
from .._util import _is_na


def _align_row(unique_cdr3s, i_row):
    gap_open = 4
    gap_extend = 4
    subst_mat = parasail.blosum62.copy()

    for i, j in itertools.product(range(subst_mat.size), repeat=2):
        if i == j:
            subst_mat[i, j] = 0
        else:
            subst_mat[i, j] = -(4 - subst_mat.matrix[i, j])

    target = unique_cdr3s[i_row]
    profile = parasail.profile_create_16(target, subst_mat)
    result = np.empty(len(unique_cdr3s))
    result[:] = np.nan
    for j, s2 in enumerate(unique_cdr3s[i_row + 1 :], start=i_row + 1):
        r = parasail.nw_scan_profile_16(profile, s2, gap_open, gap_extend)
        result[j] = r.score

    return result


def _make_dist_mat(adata):

    # For now, only TRA
    unique_cdr3s = np.unique(
        np.hstack(
            [
                adata.obs.loc[~_is_na(adata.obs["TRA_1_cdr3"]), "TRA_1_cdr3"].values,
                adata.obs.loc[~_is_na(adata.obs["TRA_2_cdr3"]), "TRA_2_cdr3"].values,
            ]
        )
    )

    print(unique_cdr3s)

    dist_mat = np.empty([len(unique_cdr3s)] * 2)

    p = Pool()
    rows = p.starmap(
        _align_row, zip(itertools.repeat(unique_cdr3s), range(len(unique_cdr3s)))
    )

    dist_mat = np.vstack(rows)
    assert dist_mat.shape[0] == dist_mat.shape[1]

    # mirror matrix at diagonal (https://stackoverflow.com/a/42209263/2340703)
    i_lower = np.tril_indices(dist_mat.shape[0], -1)
    dist_mat[i_lower] = dist_mat.T[i_lower]
    np.fill_diagonal(dist_mat, 0)

    assert np.allclose(dist_mat, dist_mat.T, 1e-8, 1e-8), "Matrix not symmetric"
    dist_df = pd.DataFrame(dist_mat)

    dist_df.index = dist_df.columns = unique_cdr3s

    return dist_df


def tcr_dist(adata: AnnData, *, inplace: bool = True) -> Union[None, dict]:
    """Compute the TCRdist on CDR3 sequences. 

    High-performance sequence alignment through parasail library [Daily2016]_

    Parameters
    ----------
    adata
    subst_mat
    gap_open
    gap_extend
    """

    dist_df = _make_dist_mat(adata)

    res = np.zeros((adata.shape[0], adata.shape[0]))

    cells_with_cdr3 = adata.obs.reset_index(drop=True)["TRA_1_cdr3"]
    cells_with_cdr3 = cells_with_cdr3[~_is_na(cells_with_cdr3)]

    print("done w. alignments", flush=True)

    for i, cdr3_1 in cells_with_cdr3.items():
        for j, cdr3_2 in cells_with_cdr3.items():
            res[i, j] = dist_df.loc[cdr3_1, cdr3_2]

    return res
