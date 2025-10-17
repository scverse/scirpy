from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from scirpy import tl
from scirpy.util import DataHandler

from .base import volcano


@DataHandler.inject_param_docs()
def clonotype_imbalance(
    adata: DataHandler.TYPE,
    replicate_col: str,
    groupby: str,
    case_label: str,
    *,
    control_label: None | str = None,
    target_col: str = "clone_id",
    additional_hue: None | str | bool = None,
    top_n: int = 10,
    fraction: None | str | bool = None,
    inplace: bool = True,
    plot_type: Literal["volcano", "box", "bar", "strip"] = "box",
    key_added: str = "clonotype_imbalance",
    xlab: str = "log2FoldChange",
    ylab: str = "-log10(p-value)",
    title: str = "Volcano plot",
    airr_mod: str = "airr",
    **kwargs,
) -> plt.Axes:
    """\
    Aims to find clonotypes that are the most enriched or depleted in a category.

    Uses Fischer's exact test to rank clonotypes.
    Depends on execution of clonotype_overlap.
    Adds two dataframes (pval and logFC for clonotypes;
    abundance of clonotypes per sample) to `uns`

    .. warning::
        This is an experimental function that will likely change in the future.

    Parameters
    ----------
    {adata}
    replicate_col
        Column with batch or sample labels.
    groupby
        The column containing categories that we want to compare and find imbalance between
    case_label
        The label in `groupby` column that we want to compare.
    control_label
        The label in `groupby` column that we use as a baseline for comparison. If not set
        (None by default), all labels that are not equal to `case_label` make up the baseline.
    target_col
        The clusters (clonotypes by default) that are imbalanced.
    additional_hue
        An additional grouping factor. If the `case_label` was tumor for example, this could
        help make a distinction between imbalance in lung and colorectal tumors.
    top_n
        The number of top clonotypes to be visualized.
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    plot_type
        Whether a volcano plot of statistics or a box/bar/strip plot of frequencies
        should be shown.
    key_added
        If the tools has already been run, the results are added to `uns` under this key.
    {airr_mod}
    **kwargs
        Additional arguments passed to the base plotting function.

    Returns
    -------
    Axes object
    """
    params = DataHandler(adata, airr_mod)
    if key_added not in params.adata.uns:
        sc.logging.warning(
            "Clonotype imbalance not found. Running `ir.tl.clonotype_imbalance` and storing under {key_added}"
        )

        tl.clonotype_imbalance(
            params,
            replicate_col=replicate_col,
            groupby=groupby,
            case_label=case_label,
            control_label=control_label,
            target_col=target_col,
            additional_hue=additional_hue,
            fraction=fraction,
            key_added=key_added,
        )

    df = params.adata.uns[key_added]["pvalues"]

    if plot_type == "volcano":
        df = df.loc[:, ["logFC", "logpValue"]]
        default_style_kws = {"title": title, "xlab": xlab, "ylab": ylab}
        if "style_kws" in kwargs:
            default_style_kws.update(kwargs["style_kws"])
        kwargs["style_kws"] = default_style_kws
        return volcano(df, **kwargs)

    else:
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.7
        if "dodge" not in kwargs:
            kwargs["dodge"] = True

        tclt_kws = {
            "x": target_col,
            "y": "Normalized abundance",
            "hue": groupby,
        }

        df = df.sort_values(by="pValue")
        df = df.head(n=top_n)

        tclt_df = adata.uns[key_added]["abundance"]
        tclt_df = tclt_df.loc[tclt_df[target_col].isin(df[target_col]), :]

        if additional_hue is None:
            tclt_df = tclt_df.pivot_table(
                index=[groupby, replicate_col],
                columns=target_col,
                values="Normalized abundance",
                fill_value=0,
            ).reset_index()
            tclt_kws["data"] = pd.melt(
                tclt_df,
                id_vars=[groupby, replicate_col],
                value_name="Normalized abundance",
            )  # Melt is undoing pivot, but after pivot, we also get the zero freqs
            if plot_type == "box":
                ax = sns.boxplot(**tclt_kws)
            else:
                if plot_type == "bar":
                    ax = sns.barplot(**tclt_kws)
                else:
                    tclt_kws.update(kwargs)
                    ax = sns.stripplot(**tclt_kws)

        else:
            tclt_df = tclt_df.pivot_table(
                index=[additional_hue, groupby, replicate_col],
                columns=target_col,
                values="Normalized abundance",
                fill_value=0,
            ).reset_index()
            tclt_kws["data"] = pd.melt(
                tclt_df,
                id_vars=[additional_hue, groupby, replicate_col],
                value_name="Normalized abundance",
            )
            kwargs["kind"] = plot_type
            kwargs["col"] = additional_hue
            tclt_kws.update(kwargs)
            if plot_type in ["box", "bar"]:
                tclt_kws.pop("alpha")
            ax = sns.catplot(**tclt_kws)
        return ax
