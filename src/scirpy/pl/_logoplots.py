from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logomaker import Logo, alignment_to_matrix

from scirpy.get import airr as get_airr
from scirpy.util import DataHandler

from .styling import _init_ax


@DataHandler.inject_param_docs()
def logoplot_cdr3_motif(
    adata: DataHandler.TYPE,
    *,
    chains: Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"] | Sequence[Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"]] = "VDJ_1",
    airr_mod: str = "airr",
    airr_key: str = "airr",
    chain_idx_key: str = "chain_indices",
    cdr3_col: str = "junction_aa",
    to_type: Literal["information", "counts", "probability", "weight"] = "information",
    pseudocount: float = 0,
    background: np.ndarray | pd.DataFrame = None,
    center_weights: bool = False,
    font_name: str = "sans",
    color_scheme: str = "chemistry",
    vpad: float = 0.05,
    width: float = 0.9,
    ax: plt.Axes | None = None,
    fig_kws: dict | None = None,
    **kwargs,
) -> Logo | list[Logo]:
    """
    Generates logoplots of CDR3 sequences

    This is a user friendly wrapper function around the logomaker python package.
    Enables the analysis of potential amino acid motifs by displaying logo plots.
    Subsetting of AnnData/MuData has to be performed manually beforehand (or while calling).
    If multiple CDR3 sequence lengths are present, separate logo plots are generated for each length.

    Parameters
    ----------
    {adata}
    chains
        One or up to two chains from which to use CDR3 sequences i.e. primary and/or secondary VJ/VDJ chains. Mixing VJ and VDJ chains will likely not lead to a meaningful result.
    {airr_mod}
    {airr_key}
    {chain_idx_key}
    cdr3_col
        Key inside awkward array to retrieve junction information (should be in aa)
    to_type
        Choose one of matrix types as defined by logomaker:
        * `"information"`
        * `"counts"`
        * `"probability"`
        * `"weight"`
    pseudocount
        Pseudocount to use when converting from counts to probabilities
    background
        Background probabilities. Both arrays with the same length as ouput or df with same shape as ouput are permitted.
    center_weights
        Whether to subtract the mean of each row, but only if to_type == `weight`
    font_name
        customize the font face. You can list all available fonts with `logomaker.list_font_names()`.
    color_scheme
        customize the color scheme. You can list all available color schemes with `logomaker.list_color_schemes()`.
    vpad
        The whitespace to leave above and below each character within that character's bounding box.
    width
        x coordinate span of each character
    ax
        Add the plot to a predefined Axes object.
        Can only be used when all selected sequences have the same CDR3 length.
    fig_kws
        Parameters passed to the :func:`matplotlib.pyplot.figure` call
        if no `ax` is specified.
    **kwargs
        Additional arguments passed to `logomaker.Logo()` for comprehensive customization.
        For a full list of parameters please refer to `logomaker documentation <https://logomaker.readthedocs.io/en/latest/implementation.html#logo-class>`_

    Returns
    -------
    Returns one object of class :class:`logomaker.Logo` if all selected sequences have the
    same length. If multiple sequence lengths are present, returns a list of
    :class:`logomaker.Logo` objects, one per length.
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)

    if isinstance(chains, str):
        chains = [chains]

    # sequences need to be aligned for each logo plot, so we split by sequence length
    airr_df = get_airr(params, [cdr3_col], chains)
    sequence_list = []
    for chain in chains:
        for sequence in airr_df[chain + "_" + cdr3_col]:
            if not pd.isnull(sequence):
                sequence_list.append(sequence)
    sequences_by_length: dict[int, list[str]] = {}
    for sequence in sequence_list:
        sequences_by_length.setdefault(len(sequence), []).append(sequence)
    n_lengths = len(sequences_by_length)

    def _make_logo(seq_list: list[str], ax_: plt.Axes, title: str) -> Logo:
        motif = alignment_to_matrix(
            seq_list, to_type=to_type, pseudocount=pseudocount, background=background, center_weights=center_weights
        )
        cdr3_logo = Logo(
            motif,
            color_scheme=color_scheme,
            vpad=vpad,
            width=width,
            font_name=font_name,
            ax=ax_,
            **kwargs,
        )
        cdr3_logo.style_xticks(anchor=0, spacing=1, rotation=45)
        cdr3_logo.ax.set_ylabel(f"{to_type}")
        cdr3_logo.ax.grid(False)
        cdr3_logo.ax.set_xlim([-1, len(motif)])
        cdr3_logo.ax.set_title(title)
        return cdr3_logo

    chain_label = "/".join(chains)
    if n_lengths == 1:
        if ax is None:
            fig_kws = {} if fig_kws is None else fig_kws
            if "figsize" not in fig_kws:
                fig_kws["figsize"] = (6, 2)
            ax = _init_ax(fig_kws)
        only_sequences = next(iter(sequences_by_length.values()))
        return _make_logo(only_sequences, ax, chain_label)

    if ax is not None:
        raise ValueError(
            "Selected sequences contain multiple CDR3 lengths, but a single `ax` was provided. "
            "Please omit `ax` to auto-generate one subplot per length."
        )

    fig_kws = {} if fig_kws is None else fig_kws
    if "figsize" not in fig_kws:
        fig_kws["figsize"] = (6, 2 * n_lengths)

    _, axes = plt.subplots(n_lengths, 1, squeeze=False, **fig_kws)
    logos = []
    for ax_, sequence_length in zip(axes.ravel(), sorted(sequences_by_length), strict=False):
        logos.append(
            _make_logo(
                sequences_by_length[sequence_length],
                ax_,
                f"{chain_label} (CDR3 length {sequence_length})",
            )
        )
    return logos
