from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy as sc_hierarchy
from scipy.spatial import distance as sc_distance

from scirpy import tl
from scirpy.util import DataHandler

from .base import ol_scatter
from .styling import _get_colors


@DataHandler.inject_param_docs()
def repertoire_overlap(
    adata: DataHandler.TYPE,
    groupby: str,
    *,
    target_col: str = "clone_id",
    pair_to_plot: None | Sequence[str] = None,
    heatmap_cats: None | Sequence[str] = None,
    dendro_only: bool = False,
    overlap_measure: str = "jaccard",
    overlap_threshold: None | float = None,
    fraction: None | str | bool = None,
    added_key: str = "repertoire_overlap",
    airr_mod: str = "airr",
    **kwargs,
) -> sns.matrix.ClusterGrid | plt.Axes:
    """\
    Visualizes overlap betwen a pair of samples on a scatter plot or
    all samples on a heatmap or draws a dendrogram of samples only.

    .. warning::
        This is an experimental function that will likely change in the future.

    Parameters
    ----------
    {adata}
    groupby
        Column with group labels (e.g. samples, tissue source, diagnosis, etc).
    target_col
        Category that overlaps among groups (`clone_id` by default, but can
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
    added_key
        If the tools has already been run, the results are added to `uns` under this key.
    {airr_mod}
    **kwargs
        Additional arguments passed to the base plotting function (scatter or seaborn.clustermap, repsectively).

    Returns
    -------
    Axes object
    """
    if dendro_only is True:
        raise NotImplementedError("This functionality was removed in scirpy v0.13.")

    params = DataHandler(adata, airr_mod)
    tl.repertoire_overlap(
        adata,
        groupby=groupby,
        target_col=target_col,
        overlap_measure=overlap_measure,
        overlap_threshold=overlap_threshold,
        fraction=fraction,
        added_key=added_key,
    )
    df = params.adata.uns[added_key]["weighted"]

    if pair_to_plot is None:
        linkage = params.adata.uns[added_key]["linkage"]

        color_dicts = {}
        row_colors = pd.DataFrame(index=df.index)
        if heatmap_cats is not None:
            for lbl in heatmap_cats:
                try:
                    labels = params.get_obs([groupby, lbl]).drop_duplicates().set_index(groupby).reindex(df.index)
                except ValueError as e:
                    if "duplicate labels" in str(e):
                        raise ValueError(
                            "Cannot color by category that is not unique for the categories in `groupby`."
                        ) from None
                    else:
                        raise

                # TODO refactor get_colors
                color_dicts[lbl] = _get_colors(params, lbl)
                for _label in labels[lbl]:
                    row_colors[lbl] = labels[lbl].map(color_dicts[lbl])

        distM = params.adata.uns[added_key]["distance"]
        distM = sc_distance.squareform(distM)
        np.fill_diagonal(distM, np.nan)
        distM = pd.DataFrame(distM, index=df.index, columns=df.index)
        dd = sc_hierarchy.dendrogram(linkage, labels=df.index, no_plot=True)
        distM = distM.iloc[dd["leaves"], :]
        if heatmap_cats is None:
            ax = sns.clustermap(
                1 - distM,
                col_linkage=linkage,
                row_cluster=False,
                **kwargs,
            )
        else:
            ax = sns.clustermap(
                1 - distM,
                col_linkage=linkage,
                row_cluster=False,
                row_colors=row_colors,
                **kwargs,
            )
            lax = ax.ax_row_dendrogram
            for column, tmp_color_dict in color_dicts.items():
                for cat, color in tmp_color_dict.items():
                    lax.bar(0, 0, color=color, label=f"{column}: {cat}", linewidth=0)
            lax.legend(loc="lower right")
        b, t = ax.ax_row_dendrogram.get_ylim()
        l, r = ax.ax_row_dendrogram.get_xlim()
        ax.ax_row_dendrogram.text(l, 0.9 * t, f"1-distance ({overlap_measure})")

    else:
        invalid_pair_warning = (
            "Did you supply two valid " + groupby + " names? Current indices are: " + ";".join(df.index.values)
        )
        valid_pairs = False
        try:
            o_df = df.loc[list(pair_to_plot), :].T
            valid_pairs = True
        except KeyError:
            pass
        if valid_pairs:
            if o_df.shape[1] == 2:
                o_df = o_df.loc[(o_df.sum(axis=1) != 0), :]
                o_df = o_df.groupby(pair_to_plot, observed=True).agg("size")
                o_df = o_df.reset_index()
                o_df.columns = ("x", "y", "z")
                o_df["z"] -= o_df["z"].min()
                o_df["z"] /= o_df["z"].max()
                o_df["z"] += 0.05
                o_df["z"] *= 1000

                # Create text for default labels
                p_a, p_b = pair_to_plot
                default_style_kws = {
                    "title": "Repertoire overlap between " + p_a + " and " + p_b,
                    "xlab": "Clonotype size in " + p_a,
                    "ylab": "Clonotype size in " + p_b,
                }
                if "style_kws" in kwargs:
                    default_style_kws.update(kwargs["style_kws"])
                kwargs["style_kws"] = default_style_kws
                ax = ol_scatter(o_df, **kwargs)
            else:
                raise ValueError("Wrong number of members. A pair is exactly two items! " + invalid_pair_warning)
        else:
            raise ValueError(invalid_pair_warning)

    return ax
