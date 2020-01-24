import matplotlib.pyplot as plt
import numpy as np


def alpha_diversity(adata):
    """Plot the alpha diversity of clonotypes in a grouping variable

    Needs to be computed with :meth:`tl.alpha_diversity`. 
    """
    groups, diversity = zip(*adata.uns["tcr_alpha_diversity"]["diversity"].items())
    fig, ax = plt.subplots()
    x = np.arange(len(groups))
    ax.bar(x, diversity)

    ax.set_ylabel("Shannon entropy")
    ax.set_title("Alpha diversity of clonotypes")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def clonal_expansion(group=None):
    """Plot the fraction of cells in each group belonging to
    singleton, doublet or triplet clonotype. 
    """
    pass


def repertoire_overlap():
    """Heatmap showing the shared clonotypes by cells (colored by cell group)"""
    pass
