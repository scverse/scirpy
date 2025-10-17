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
) -> Logo:
    """
    Generates logoplots of CDR3 sequences

    This is a user friendly wrapper function around the logomaker python package.
    Enables the analysis of potential amino acid motifs by displaying logo plots.
    Subsetting of AnnData/MuData has to be performed manually beforehand (or while calling) and only cells with equal cdr3 sequence lengths are permitted.

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
    fig_kws
        Parameters passed to the :func:`matplotlib.pyplot.figure` call
        if no `ax` is specified.
    **kwargs
        Additional arguments passed to `logomaker.Logo()` for comprehensive customization.
        For a full list of parameters please refer to `logomaker documentation <https://logomaker.readthedocs.io/en/latest/implementation.html#logo-class>`_

    Returns
    -------
    Returns a object of class logomaker.Logo (see here for more information https://logomaker.readthedocs.io/en/latest/implementation.html#matrix-functions)
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)

    if isinstance(chains, str):
        chains = [chains]

    if ax is None:
        fig_kws = {} if fig_kws is None else fig_kws
        if "figsize" not in fig_kws:
            fig_kws["figsize"] = (6, 2)
        ax = _init_ax(fig_kws)

    # make sure that sequences are prealigned i.e. they need to have the the same length
    airr_df = get_airr(params, [cdr3_col], chains)
    sequence_list = []
    for chain in chains:
        for sequence in airr_df[chain + "_" + cdr3_col]:
            if sequence is not None:
                sequence_list.append(sequence)

    motif = alignment_to_matrix(
        sequence_list, to_type=to_type, pseudocount=pseudocount, background=background, center_weights=center_weights
    )
    cdr3_logo = Logo(motif, color_scheme=color_scheme, vpad=vpad, width=width, font_name=font_name, ax=ax, **kwargs)

    cdr3_logo.style_xticks(anchor=0, spacing=1, rotation=45)
    cdr3_logo.ax.set_ylabel(f"{to_type}")
    cdr3_logo.ax.grid(False)
    cdr3_logo.ax.set_xlim([-1, len(motif)])
    cdr3_logo.ax.set_title("/".join(chains))
    return cdr3_logo
