from ..util import _is_na, _is_true, deprecated
from anndata import AnnData
from typing import Union, Sequence, Tuple
import numpy as np
from scanpy import logging
from ..io._util import _check_upgrade_schema


@deprecated("Use `tl.chain_qc` instead.")
def chain_pairing(
    adata: AnnData, *, inplace: bool = True, key_added: str = "chain_pairing"
) -> Union[None, np.ndarray]:
    """Categorize cells based on how many TRA and TRB chains they have.

    Parameters
    ----------
    adata
        Annotated data matrix
    inplace
        If True, adds a column to adata.obs
    key_added
        Column name to add to 'obs'

    Returns
    -------
    Depending on the value of `inplace`, either
    returns a Series with a chain pairing category for each cell
    or adds a `chain_pairing` column to `adata`.
    """
    res = chain_qc(
        adata,
        inplace=inplace,
        key_added=("receptor_type", "receptor_subtype", key_added),
    )
    if not inplace:
        return res[2]


@_check_upgrade_schema()
def chain_qc(
    adata: AnnData,
    *,
    inplace: bool = True,
    key_added: Sequence[str] = ("receptor_type", "receptor_subtype", "chain_pairing"),
) -> Union[None, Tuple[np.ndarray]]:
    """Perform quality control based on the receptor-chain pairing configuration.

    Categorizes cells into their receptor types and according to their chain pairing
    status. The function adds three columns to `adata.obs`, two containing a coarse
    and fine annotation of receptor types, a third classifying cells according
    to the number of matched receptor types.

    `receptor_type` can be one of the following
        * `TCR` (all cells that contain any combination of TRA/TRB/TRG/TRD chains,
          but no IGH/IGK/IGL chains)
        * `BCR` (all cells that contain any combination of IGH/IGK/IGL chains,
          but no TCR chains)
        * `ambiguous` (all cells that contain both BCR and TCR chains)
        * `multichain` (all cells with more than two VJ or more than two VDJ chains)
        * `no IR` (all cells without any detected immune receptor)

    `receptor_subtype` can be one of the following
        * `TRA+TRB` (all cells that have only TRA and/or TRB chains)
        * `TRG+TRD` (all cells that have only TRG and/or TRD chains)
        * `IGH` (all cells that have only IGH chains, but no IGL or IGK)
        * `IGH+IGL` (all cells that have only IGH and IGL chains)
        * `IGH+IGK` (all cells that have only IGH and IGK chains)
        * `multichain` (all cells with more than two VJ or more than two VDJ chains)
        * `ambiguous` (all cells that are none of the above, e.g. TRA+TRD, TRA+IGH or,
          IGH+IGK as the primary and IGH+IGL as the secondary receptor)
        * `no IR` (all cells without any detected immune receptor)

    `chain_pairing` can be one of the following
        * `single pair` (all cells that have exactely one matched VJ and VDJ chain)
        * `orphan VJ` (all cells that have only one VJ chain)
        * `orphan VDJ` (all cells that have only one VDJ chain)
        * `extra VJ` (all cells that have a matched pair of VJ and VDJ chains plus an
          additional VJ-chain)
        * `extra VDJ` (analogous)
        * `two full chains` (all cells that have two matched pairs of VJ and VDJ chains)
        * `ambiguous` (all cells that have unmatched chains, i.e. that have been
          classified as an `ambiguous` receptor_subtype)
        * `multichain` (all cells with more than two VJ or more than two VDJ chains)
        * `no IR` (all chains with not immune receptor chains)

    Parameters
    ----------
    adata
        Annotated data matrix
    inplace
        If True, adds columns to to adata
    key_added
        Tuple specifying the column names for the coarse and fine receptor type
        annotation, respectively

    Returns
    -------
    Depending on the value of `inplace` either adds three columns to
    `adata.obs` or returns a tuple with three numpy arrays containing
    the annotations.
    """
    x = adata.obs

    # initalize result arrays
    string_length = len("multichain")
    res_receptor_type = np.empty(dtype=f"<U{string_length}", shape=(x.shape[0],))
    res_receptor_subtype = np.empty(dtype=f"<U{string_length}", shape=(x.shape[0],))

    mask_has_ir = _is_true(x["has_ir"].values)
    mask_multichain = mask_has_ir & _is_true(x["multi_chain"].values)

    vj_loci = x.loc[:, ["IR_VJ_1_locus", "IR_VJ_2_locus"]].values
    vdj_loci = x.loc[:, ["IR_VDJ_1_locus", "IR_VDJ_2_locus"]].values

    # Build masks for receptor chains
    has_tra = (vj_loci == "TRA").any(axis=1)
    has_trg = (vj_loci == "TRG").any(axis=1)
    has_igk = (vj_loci == "IGK").any(axis=1)
    has_igl = (vj_loci == "IGL").any(axis=1)

    has_trb = (vdj_loci == "TRB").any(axis=1)
    has_trd = (vdj_loci == "TRD").any(axis=1)
    has_igh = (vdj_loci == "IGH").any(axis=1)

    has_tr = has_tra | has_trg | has_trb | has_trd
    has_ig = has_igk | has_igl | has_igh

    # Combine masks into receptor types and subtypes
    type_is_t = has_tr & ~has_ig
    type_is_b = ~has_tr & has_ig

    subtype_is_tab = (has_tra | has_trb) & ~(has_trg | has_trd | has_ig)
    subtype_is_tgd = (has_trg | has_trd) & ~(has_tra | has_trb | has_ig)
    subtype_is_ighk = (has_igk) & ~(has_tr | has_igl)
    subtype_is_ighl = (has_igl) & ~(has_tr | has_igk)
    # orphan IGH
    subtype_is_igh = (has_igh) & ~(has_igk | has_igl | has_tr)

    # Apply masks for receptor type
    res_receptor_type[:] = "ambiguous"
    res_receptor_type[~mask_has_ir] = "no IR"
    res_receptor_type[type_is_t] = "TCR"
    res_receptor_type[type_is_b] = "BCR"
    res_receptor_type[mask_multichain] = "multichain"

    # Apply masks for receptor subtypes
    res_receptor_subtype[:] = "ambiguous"
    res_receptor_subtype[~mask_has_ir] = "no IR"
    res_receptor_subtype[subtype_is_tab] = "TRA+TRB"
    res_receptor_subtype[subtype_is_tgd] = "TRG+TRD"
    res_receptor_subtype[subtype_is_igh] = "IGH"
    res_receptor_subtype[subtype_is_ighl] = "IGH+IGL"
    res_receptor_subtype[subtype_is_ighk] = "IGH+IGK"
    res_receptor_subtype[mask_multichain] = "multichain"

    res_chain_pairing = _chain_pairing(
        adata, res_receptor_subtype == "ambiguous", mask_has_ir, mask_multichain
    )

    if inplace:
        col_receptor_type, col_receptor_subtype, col_chain_pairing = key_added
        adata.obs[col_receptor_type] = res_receptor_type
        adata.obs[col_receptor_subtype] = res_receptor_subtype
        adata.obs[col_chain_pairing] = res_chain_pairing
    else:
        return (res_receptor_type, res_receptor_subtype, res_chain_pairing)


def _chain_pairing(
    adata: AnnData,
    mask_ambiguous: np.ndarray,
    mask_has_ir: np.ndarray,
    mask_multichain: np.ndarray,
) -> np.ndarray:
    """Annotate chain pairing categories.

    Parameters:
    -----------
    mask_ambiguous
        boolean array of the same length as `adata.obs`, marking
        which cells have an ambiguous receptor configuration.
    """
    x = adata.obs
    string_length = len("two full chains")
    results = np.empty(dtype=f"<U{string_length}", shape=(x.shape[0],))

    logging.debug("Done initalizing")

    mask_has_vj1 = ~_is_na(x["IR_VJ_1_junction_aa"].values)
    mask_has_vdj1 = ~_is_na(x["IR_VDJ_1_junction_aa"].values)
    mask_has_vj2 = ~_is_na(x["IR_VJ_2_junction_aa"].values)
    mask_has_vdj2 = ~_is_na(x["IR_VDJ_2_junction_aa"].values)

    logging.debug("Done with masks")

    for m in [mask_has_vj1, mask_has_vdj1, mask_has_vj2, mask_has_vdj2]:
        # no cell can have a junction_aa sequence but no TCR
        assert np.setdiff1d(np.where(m)[0], np.where(mask_has_ir)[0]).size == 0

    results[~mask_has_ir] = "no IR"
    results[mask_has_vj1] = "orphan VJ"
    results[mask_has_vdj1] = "orphan VDJ"
    results[mask_has_vj1 & mask_has_vdj1] = "single pair"
    results[mask_has_vj1 & mask_has_vdj1 & mask_has_vj2] = "extra VJ"
    results[mask_has_vj1 & mask_has_vdj1 & mask_has_vdj2] = "extra VDJ"
    results[
        mask_has_vj1 & mask_has_vdj1 & mask_has_vj2 & mask_has_vdj2
    ] = "two full chains"
    results[mask_ambiguous] = "ambiguous"
    results[mask_multichain] = "multichain"

    assert "" not in results, "One or more chains are not characterized"

    return results
