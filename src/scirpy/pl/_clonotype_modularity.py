from collections.abc import Sequence
from importlib.metadata import version

import numpy as np
from adjustText import adjust_text
from matplotlib import patheffects
from packaging.version import Version

from scirpy.util import DataHandler

from ._clonotypes import _plot_size_legend
from .styling import _init_ax


def _rand_jitter(arr, jitter=None):
    """Add a random jitter to data"""
    if jitter is None:
        return arr
    stdev = jitter * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


@DataHandler.inject_param_docs()
def clonotype_modularity(
    adata: DataHandler.TYPE,
    ax=None,
    target_col="clonotype_modularity",
    jitter: float = 0.01,
    panel_size: tuple[float, float] = (6, 4),
    base_size: float = 2,
    size_power: float | None = 1,
    show_labels: bool = True,
    labels_quantile_cutoff: tuple[float, float] = (0.9, 0.9),
    labels: Sequence[str] = None,
    label_fontsize: int | None = 10,
    label_fontweight: str | float = 300,
    label_fontoutline: int = 0,
    label_adjusttext: bool = True,
    show_size_legend: bool = True,
    legend_width: float = 2,
    fig_kws: dict | None = None,
    airr_mod: str = "airr",
):
    """\
    Plots the :term:`Clonotype modularity` score against the associated log10 p-value.

    Parameters
    ----------
    {adata}
    ax
        Add the plot to a predefined Axes object.
    target_col
        Column in `adata.obs` containing the clonotype modularity score and
        key in `adata.uns` containing the dictionary with parameters.
        Will look for p-values or FDRs in `adata.obs["{{target_col}}_pvalue"]` or
        `adata.obs["{{target_col}}_fdr"]`.
    jitter
        Add random jitter along the x axis to avoid overlapping point.
        Samples from `N(0, jitter * (max(arr) - min(arr)))`
    base_size
        Size of a point representing 1 cell.
    size_power
        Point sizes are raised to the power of this value.
    show_labels
        Whether to show text labels for the clonotypes with highest clonotype modularity.
    labels_quantile_cutoff
        Label clonotypes with exceeding the given quantile cutoff. Only unique
        values are considered for calculating the quantiles (avoiding thousands of
        singleton clonotypes with modularity 0). The cutoff is specified as a tuple
        `(cutoff_for_modularity, cutoff_for_pvalue)`.
    labels
        Explicitly pass a list of clonotypes to label.
        Overrides `labels_quantile_cutoff`.
    label_fontsize
        Fontsize for the clonotype labels
    label_fontweight
        Fontweight for the clonotype labels
    label_fontoutline
        Size of the fontoutline added to the clonotype labels. Set to `None` to disable.
    label_adjusttext
        Whether to "repel" labels such that they don't overlap using the `adjustText`
        library. This option significantly increases the runtime.
    show_size_legend
        Whether to show a legend for dot sizes on the right margin
    legend_width
        Width of the legend column in inches. Only applies if `show_size_legend` is True
    fig_kws
        Parameters passed to the :func:`matplotlib.pyplot.figure` call
        if no `ax` is specified.
    {airr_mod}

    Returns
    -------
    A list of axis objects
    """
    params = DataHandler(adata, airr_mod)

    # Doesn't need param handler, we only access attributes of MuData or a all-in-one AnnData.
    if ax is None:
        fig_kws = {} if fig_kws is None else fig_kws
        fig_width = panel_size[0] if not show_size_legend else panel_size[0] + legend_width + 0.5
        fig_kws.update({"figsize": (fig_width, panel_size[1])})
        ax = _init_ax(fig_kws)

    size_legend_ax = None
    if show_size_legend:
        fig = ax.get_figure()
        fig_width, fig_height = fig.get_size_inches()
        hspace = 0.5
        ax.axis("off")
        gs = ax.get_subplotspec().subgridspec(
            2,  # rows = [spacer, size_legend]
            2,  # cols = [main panel, legend panel]
            width_ratios=[
                (fig_width - hspace - legend_width) / fig_width,
                legend_width / fig_width,
            ],
            height_ratios=[3, 1],
        )
        size_legend_ax = fig.add_subplot(gs[1, 1])
        ax = fig.add_subplot(gs[:, 0])

    modularity_params = params.data.uns[target_col]
    clonotype_col = modularity_params["target_col"]

    if modularity_params["fdr_correction"]:
        pvalue_col = f"{target_col}_fdr"
        pvalue_type = "FDR"
    else:
        pvalue_col = f"{target_col}_pvalue"
        pvalue_type = "p-value"

    score_df = (
        params.get_obs([clonotype_col, target_col, pvalue_col])
        .groupby([clonotype_col, target_col, pvalue_col], observed=True)
        .size()
        .reset_index(name="clonotype_size")
        .assign(log_p=lambda x: -np.log10(x[pvalue_col]))
    )

    score_df["xs"] = _rand_jitter(score_df[target_col].values, jitter=jitter)
    score_df["ys"] = score_df["log_p"].values
    score_df["sizes"] = score_df["clonotype_size"] ** size_power * base_size
    ax.scatter(
        score_df["xs"],
        score_df["ys"],
        marker=".",
        s=score_df["sizes"],
        alpha=0.5,
        edgecolors="black",
    )
    ax.set_ylim(-0.1 * max(score_df["xs"]), 1.2 * max(score_df["ys"]))
    ax.set_xlabel("modularity score")
    ax.set_ylabel(f"-log10({pvalue_type})")

    if show_size_legend:
        _plot_size_legend(
            size_legend_ax,
            sizes=score_df["clonotype_size"].values,
            size_power=size_power,
            base_size=base_size,
        )

    if show_labels:
        if labels is None:
            qm = np.quantile(score_df[target_col].unique(), labels_quantile_cutoff[0])
            qp = np.quantile(score_df["log_p"].unique(), labels_quantile_cutoff[1])
            labels = score_df[clonotype_col][(score_df[target_col] >= qm) & (score_df["log_p"] >= qp)]

        path_effect = (
            [patheffects.withStroke(linewidth=label_fontoutline, foreground="w")]
            if label_fontoutline is not None
            else None
        )

        label_objs = []
        for _, r in score_df.loc[score_df[clonotype_col].isin(labels), :].iterrows():
            label_objs.append(
                ax.text(
                    r["xs"],
                    r["ys"],
                    r[clonotype_col],
                    weight=label_fontweight,
                    fontsize=label_fontsize,
                    path_effects=path_effect,
                )
            )

        if label_adjusttext:
            kwargs = {}
            # incompatible API between <1.0 and >=1.0. I'd like to pin 1.0, but it's not available from
            # conda-forge and there are some issue (https://github.com/Phlya/adjustText/issues/166)
            if Version(version("adjustText")) >= Version("1.0"):
                kwargs["force_static"] = (0.4, 0.4)
            else:
                kwargs["force_points"] = (0.4, 0.4)

            adjust_text(
                label_objs,
                score_df["xs"].values,
                score_df["ys"].values,
                arrowprops={"arrowstyle": "-", "color": "k", "lw": 0.5},
                force_text=(0.3, 0.3),
                **kwargs,
            )

    return ax
