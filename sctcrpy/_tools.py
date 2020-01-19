import itertools
import parasail
import numpy as np
import pandas as pd


def tcr_dist(adata, subst_mat=parasail.blosum62, gap_open=8, gap_extend=1):
    """Compute the TCRdist on CDR3 sequences. 

    Currently takes into account only dominant alpha and dominant beta. 
    """
    # TODO parallelize
    for chain in ["TRA", "TRB"]:
        unique_cdr3s = unique_alpha = (
            # TODO here we have a problem again with the categorical "nan"
            adata.obs["{}_1_cdr3".format(chain)]
            .dropna()
            .unique()
        )

        dist_mat = np.empty([len(unique_cdr3s)] * 2)

        for i, s1 in enumerate(unique_cdr3s):
            profile = parasail.profile_create_16(s1, subst_mat)
            for j, s2 in enumerate(unique_cdr3s[i + 1 :]):
                r = parasail.sw_striped_profile_16(profile, s2, gap_open, gap_extend)
                dist_mat[i, j] = r.score

        dist_df = pd.DataFrame(dist_mat)
        dist_df.index = dist_df.columns = unique_cdr3s

        adata.uns["tcr_dist_alpha"] = dist_df


def alpha_diversity(frequencies, flavor="shannon"):
    """
    Alpha diversity. Could rely on skbio.math if more variants are required. 
    """
    assert flavor == "shannon", "Other types not supported yet"
    return -np.sum(frequencies * np.log2(frequencies))
