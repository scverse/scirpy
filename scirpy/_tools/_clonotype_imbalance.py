from anndata import AnnData
from typing import Union, Tuple, List, Sequence
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd
import scanpy as sc
from ._repertoire_overlap import repertoire_overlap
from ..io._util import _check_upgrade_schema
from ..util import deprecated


@_check_upgrade_schema()
@deprecated(
    "Consider using `tl.clonotype_modularity` instead. If `clonotype_modularity` "
    "does not cover your use-case, please create an issue on GitHub to let us know "
    "such that we can take it into account! (https://github.com/icbi-lab/scirpy/issues)"
)
def clonotype_imbalance(
    adata: AnnData,
    replicate_col: str,
    groupby: str,
    case_label: str,
    *,
    control_label: Union[None, str] = None,
    target_col: str = "clone_id",
    additional_hue: Union[None, str, bool] = None,
    fraction: Union[None, str, bool] = None,
    inplace: bool = True,
    overlap_key: Union[None, str] = None,
    key_added: str = "clonotype_imbalance",
) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Aims to find clonotypes that are the most enriched or depleted in a category.

    Uses Fischer's exact test to rank clonotypes.
    Depends on execution of :func:`scirpy.tl.repertoire_overlap`.
    Adds two dataframes (abundance of clonotypes per sample; pval and logFC for clonotypes) to `uns`

    .. warning::

        This is an experimental function and will likely change in the future.

    Parameters
    ----------
    adata
        AnnData object to work on.
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
    fraction
        If `True`, compute fractions of abundances relative to the `groupby` column
        rather than reporting abosolute numbers. Alternatively, a column
        name can be provided according to that the values will be normalized or an iterable
        providing cell weights directly. Setting it to `False` or `None` assigns equal weight
        to all cells.
    inplace
        Whether results should be added to `uns` or returned directly.
    overlap_key
        Under what key should the repertoire overlap results be looked up in `uns`.
        By default it is None to ensure that the overlap tool is executed with the right parameters.
    key_added
        Results will be added to `uns` under this key.


    Returns
    -------
    Two dataframes: abundance of clonotypes per sample; pval and logFC for clonotypes.
    """

    # Retrieve clonotype presence matrix
    if overlap_key is None:
        sc.logging.warning(
            "Clonotype imbalance calculation depends on repertoire overlap. We could not detect any"
            " previous runs of repertoire_overlap, so the tool is running now..."
        )
        clonotype_presence, dM, lM = repertoire_overlap(
            adata,
            groupby=replicate_col,
            target_col=target_col,
            fraction=fraction,
            inplace=False,
        )
    else:
        try:
            clonotype_presence = adata.uns[overlap_key]["weighted"]
        except KeyError:
            raise KeyError(
                "Clonotype imbalance calculation depends on repertoire overlap, but the key"
                " you specified does not belong to a previous run of that tool."
            )

    global_minimum = clonotype_presence.min().min() / clonotype_presence.shape[0]
    global_minimum = global_minimum * 0.01

    # Create a series of case-control groups for comparison
    case_control_groups = _create_case_control_groups(
        adata.obs, replicate_col, groupby, additional_hue, case_label, control_label
    )

    #  Compare groups with Fischer's test
    clt_freq, clt_stats = [], []
    if control_label is None:
        control_label = "Background"
    for hue, cases, controls, ncase, ncontrol in case_control_groups:
        if hue is None:
            hue = "All"
        tdf1 = clonotype_presence.loc[
            cases,
        ]
        tdf2 = clonotype_presence.loc[
            controls,
        ]
        suspects = set(
            tdf1.loc[:, tdf1.sum() > 0].columns.values.tolist()
            + tdf2.loc[:, tdf2.sum() > 0].columns.values.tolist()
        )
        for suspect in suspects:
            p, logfoldchange, rel_case_sizes, rel_control_sizes = _calculate_imbalance(
                tdf1[suspect], tdf2[suspect], ncase, ncontrol, global_minimum
            )
            clt_stats.append([suspect, p, -np.log10(p), logfoldchange])
            clt_freq = _extend_clt_freq(
                clt_freq,
                suspect,
                hue,
                case_label,
                control_label,
                rel_case_sizes,
                rel_control_sizes,
            )

    # Convert records to data frames
    clt_freq = pd.DataFrame.from_records(
        clt_freq,
        columns=[
            target_col,
            additional_hue,
            groupby,
            replicate_col,
            "Normalized abundance",
        ],
    )
    clt_stats = pd.DataFrame.from_records(
        clt_stats, columns=[target_col, "pValue", "logpValue", "logFC"]
    )
    clt_stats = clt_stats.sort_values(by="pValue")

    if inplace:

        # Store calculated data
        adata.uns[key_added] = {"abundance": clt_freq, "pvalues": clt_stats}
        return

    else:
        return clt_freq, clt_stats


def _create_case_control_groups(
    df: pd.DataFrame,
    replicate_col: str,
    groupby: str,
    additional_hue: Union[None, str, bool],
    case_label: str,
    control_label: Union[None, str],
) -> List:
    """Creates groups for comparison.

    Parameters
    ----------
    df
        A pandas dataframe with all the columns we will use for grouping.
    replicate_col
        Column with batch or sample labels.
    groupby
        The column containing categories that we want to compare and find imbalance between
    additional_hue
        An additional grouping factor. If the `case_label` was tumor for example, this could
        help make a distinction between imbalance in lung and colorectal tumors.
    case_label
        The label in `groupby` column that we want to compare.
    control_label
        The label in `groupby` column that we use as a baseline for comparison. If not set
        (None by default), all labels that are not equal to `case_label` make up the baseline.
    target_col
        The clusters (clonotypes by default) that are imbalanced.

    Returns
    -------
    A list, where each item consist of a hue, list of cases, list of controls,
    number of cases, number of controls.
    """

    case_control_groups = []
    group_cols = [groupby, replicate_col]
    if additional_hue is None:
        hues = [None]
    else:
        group_cols.append(additional_hue)
        hues = df[additional_hue].unique()
    df = df.groupby(group_cols, observed=True).agg("size").reset_index()

    for hue in hues:
        if hue is None:
            tdf = df
        else:
            tdf = df.loc[df[additional_hue] == hue, :]
        cases = tdf.loc[df[groupby] == case_label, :]
        ncase = cases[0]
        cases = cases[replicate_col]
        if control_label is None:
            controls = tdf.loc[df[groupby] != case_label, :]
        else:
            controls = tdf.loc[df[groupby] == control_label, :]
        ncontrol = controls[0]
        controls = controls[replicate_col]
        case_control_groups.append([hue, cases, controls, ncase, ncontrol])

    return case_control_groups


def _calculate_imbalance(
    case_sizes: Union[np.ndarray, pd.Series],
    control_sizes: Union[np.ndarray, pd.Series],
    ncase: Sequence,
    ncontrol: Sequence,
    global_minimum: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Calculate statistics for the probability of an imbalance in the contingency table
    among two groups.

    Parameters
    ----------
    case_sizes
        An iterable of sizes (conts or normalized counts) in a given group for
        each replicates.
    control_sizes
        An iterable of sizes (conts or normalized counts) in the control group for
        each replicates.
    ncase
        Total size (all counts or sum of normalized counts) of a given group for
        each replicates.
    ncontrol
        Total size (all counts or sum of normalized counts) of the control group
        for each replicates.
    global_minimum
        Virtual residual value to avoid zero divisions. Typically, it is 1% of the
        minimum of the whole clonotype abundance table without zero values.

    Returns
    -------
    The p-value of a Fischers exact test and a logFoldChange of the case frequency
    compared to the control frequency and the relative sizes for case and control groups.
    """

    rel_case_sizes = case_sizes / np.array(ncase)
    rel_control_sizes = control_sizes / np.array(ncontrol)
    case_mean_freq = np.mean((case_sizes) / np.array(ncase))
    case_presence = case_sizes.sum()
    case_absence = ncase.sum() - case_presence
    if case_absence < 0:
        case_absence = 0
    control_mean_freq = np.mean((control_sizes) / np.array(ncontrol))
    control_presence = control_sizes.sum()
    control_absence = ncontrol.sum() - control_presence
    if control_absence < 0:
        control_absence = 0
    oddsratio, p = fisher_exact(
        [[case_presence, control_presence], [case_absence, control_absence]]
    )
    logfoldchange = np.log2(
        (case_mean_freq + global_minimum) / (control_mean_freq + global_minimum)
    )
    return p, logfoldchange, rel_case_sizes, rel_control_sizes


def _extend_clt_freq(
    clt_freq: List,
    suspect: str,
    hue: str,
    case_label: str,
    control_label: str,
    rel_case_sizes: pd.Series,
    rel_control_sizes: pd.Series,
) -> List:
    """Adds case and control frequencies to a summary list resembling the long data format.

    Parameters
    ----------
    clt_freq
        A list of records to be extended.
    suspect
        Label for the clonotype tested.
    hue
        label for hue (subgrouping factor).
    case_label
        Label for cases.
    control_label
        Label for controls.
    rel_cases_sizes
        Pandas series of case frequencies that should be added to thelist one-by-one.
    rel_control_sizes
        Pandas series of control frequencies that should be added to thelist one-by-one.

    Returns
    -------
    The extended list, where each item is a tuple of the tested clonotype, hue label,
    group label, replicate name, group size (frequency).
    """

    for e in rel_case_sizes.index.values:
        clt_freq.append((suspect, hue, case_label, e, rel_case_sizes.loc[e].mean()))
    for e in rel_control_sizes.index.values:
        clt_freq.append(
            (suspect, hue, control_label, e, rel_control_sizes.loc[e].mean())
        )
    return clt_freq
