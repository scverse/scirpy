import itertools
import parasail
import numpy as np
import pandas as pd


def tcr_dist(clonotype_df, subst_mat=parasail.blosum62, gap_open=8, gap_extend=1):
    """Compute the TCRdist on CDR3 sequences. 

    Currently takes into account only dominant alpha and dominant beta. 
    """
    # TODO parallelize
    unique_alpha = clonotype_df["dominant_alpha"].dropna().unique()
    unique_beta = clonotype_df["dominant_beta"].dropna().unique()

    dist_alpha = np.empty([len(unique_alpha)] * 2)
    dist_beta = np.empty([len(unique_alpha)] * 2)

    for i, s1 in enumerate(unique_alpha):
        profile = parasail.profile_create_16(s1, subst_mat)
        for j, s2 in enumerate(unique_alpha[i + 1 :]):
            r = parasail.sw_striped_profile_16(profile, s2, gap_open, gap_extend)
            dist_alpha[i, j] = r.score

    # for i, s1 in enumerate(unique_beta):
    #     profile = parasail.profile_create_16(s1, subst_mat)
    #     for j, s2 in enumerate(unique_beta[i + 1 :]):
    #         r = parasail.sw_striped_profile_16(profile, s2, gap_open, gap_extend)
    #         dist_beta[i, j] = r.score

    df_alpha = pd.DataFrame(dist_alpha)
    df_alpha.index = unique_alpha
    df_alpha.columns = unique_alpha

    return df_alpha


def alpha_diversity(frequencies, flavor="shannon"):
    """
    Alpha diversity. Could rely on skbio.math if more variants are required. 
    """
    assert flavor == "shannon", "Other types not supported yet"
    return -np.sum(frequencies * np.log2(frequencies))
