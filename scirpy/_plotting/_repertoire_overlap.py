import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance as sc_distance
from scipy.cluster import hierarchy as sc_hierarchy
from typing import Union, Tuple
from .._util import _is_na
from .. import tl
from ._styling import style_axes, DEFAULT_FIG_KWS, _init_ax
from ._base import ol_scatter


def repertoire_overlap(
    adata: AnnData,
    groupby: str,
    *,
    target_col: str = 'clonotype',
    pair_to_plot: Union[None, Tuple] = None,
    heatmap_cats: Union[None, Tuple] = None,
    dendro_only: bool = False,
    overlap_measure: str = 'jaccard',
    overlap_threshold: Union[None, float] = None,
    fraction: Union[None, str, bool] = None,
    **kwargs
) -> plt.Axes:
    """Visualizes overlap betwen a pair of samples on a scatter plot or
    all samples on a heatmap or draws a dendrogram of samples only.
    
    Parameters
    ----------
    adata
        AnnData object to work on.
    groupby
        Column with group labels (e.g. samples, tussue source, diagnosis, etc).        
    target_col
        Category that overlaps among groups (`clonotype` by default, but can
        in principle be any group or cluster)
    pair_to_plot
        A tuple of two sample names that should be plotted on an IR overlap scatterplot.
    heatmap_cats
        Column names that should be shown as category on the side of the heatmap.
    dendro_only
        In case all samples are visualized, sets if a heatmap should be shown or only a dendrogram.
    overlap_measure
        Any distance measure accepted by `scipy.spatial.distance`; by default it is `jaccard`.
    overlap_threshold
        The minimum required weight to accept presence.
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column 
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    **kwargs
        Additional arguments passed to the base plotting function.  
    
    Returns
    -------
    Axes object
    """
    
    if 'repertoire_overlap' not in adata.uns:
        tl.repertoire_overlap(adata, groupby=groupby, target_col=target_col, overlap_measure=overlap_measure, overlap_threshold=overlap_threshold, fraction=fraction)
    df = adata.uns['repertoire_overlap']['weighted']

    if pair_to_plot is None:
        linkage = adata.uns['repertoire_overlap']['linkage']
        clust_colors = ['red']*len(df.index.values)
        if dendro_only:
            ax = _init_ax()
            sc_hierarchy.dendrogram(linkage, labels=df.index, ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticks([])
        else:
            scaling_factor = 0.9 # This is a subjective value to ease color bar scaling
            distM = adata.uns['repertoire_overlap']['distance']
            distM = sc_distance.squareform(distM)
            np.fill_diagonal(distM, scaling_factor)
            distM = pd.DataFrame(distM, index=df.index, columns=df.index)
            ax = sns.clustermap(1-distM, row_colors=clust_colors, row_linkage=linkage, col_linkage=linkage)
    
    else:
        invalid_pair_warning = 'Cannot find this pair in the data table. Did you supply two valid '+groupby+' names? Current indices are: '+';'.join(df.index.values)
        try:
            o_df = df.loc[list(pair_to_plot), :].T
            valid_pairs = True
        except:
            valid_pairs = False
        if valid_pairs:
            if o_df.shape[1] == 2:
                o_df = o_df.loc[(o_df.sum(axis=1) != 0), :]
                o_df = o_df.groupby(pair_to_plot).agg('size')
                o_df = o_df.reset_index()
                o_df.columns = ('x', 'y', 'z')
                o_df['z'] -= o_df['z'].min()
                o_df['z'] /= o_df['z'].max()
                o_df['z'] += 0.05
                o_df['z'] *= 1000

                # Create text for default labels
                p_a, p_b = pair_to_plot
                default_style_kws = {"title": 'Repertoire overlap between '+p_a+'and'+p_b, "xlab": 'Conotype size in '+p_a, "ylab": 'Conotype size in '+p_b}
                if "style_kws" in kwargs:
                    default_style_kws.update(kwargs["style_kws"])
                kwargs["style_kws"] = default_style_kws
                ax = ol_scatter(o_df, **kwargs)
            else:
                print(invalid_pair_warning)
        else:
            print(invalid_pair_warning)

    return ax
