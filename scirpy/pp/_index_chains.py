from functools import partial
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, cast

import awkward as ak
from anndata import AnnData
from scanpy import logging

from ..io._datastructures import AirrCell
from ..util import _is_na2

SCIRPY_DUAL_IR_MODEL = "scirpy_dual_ir_v0.13"


def index_chains(
    adata: AnnData,
    *,
    productive: bool = True,
    require_junction_aa: bool = True,
    sort_chains_by: Mapping[str, Any] = MappingProxyType(
        {"duplicate_count": 0, "consensus_count": 0, "junction": "", "junction_aa": ""}
    ),
    airr_key: str = "airr",
    key_added: str = "chain_indices",
) -> None:
    """
    Determine which chains are considered primary/secondary VJ/VDJ chains

    This function goes through all chains stored in the :term:`awkward array` in
    `adata.obsm[airr_key]` and

     * labels chains as primary/secondary VJ/VDJ chains
     * labels cells as multichain cells

    based on the expression level of the chains and if they are labelled as "productive" or not.

    TODO #356 link to receptor model

    Parameters
    ----------
    adata
        AnnData object with AIRR information
    productive
        If True, ignore non-productive chains. In that case, non-productive chains will also not count towards
        calling "multichain" cells.
    require_junction_aa
        If True, ignore chains that don't have a junction_aa (CDR3) amino acid sequence. In that case, these
        chains will also not count towards calling "multichain" cells
    sort_chains_by
        A list of sort keys used to determine an ordering of chains. The chain with the highest value
        of this tuple willl be the primary chain, second-highest the secondary chain. If there are more chains, they
        will not be indexed, and the cell receives the "multichain" flag.
    airr_key
        Key under which airr information is stored in `adata.obsm`
    key_added
        Key under which the chain indicies will be stored in `adata.obsm` and metadata will be stored in `adata.uns`.

    Returns
    -------
    Nothing, but adds a dataframe to `adata.obsm[chain_indices]`
    """
    chain_index_list = []
    awk_array = cast(ak.Array, adata.obsm[airr_key])

    # only warn if those fields are in the key (i.e. this should give a warning if those are missing with
    # default settings. If the user specifies their own dictionary, they are on their own)
    if "duplicate_count" in sort_chains_by and "consensus_count" in sort_chains_by:
        if (
            "duplicate_count" not in awk_array.fields
            and "consensus_count" not in awk_array.fields
        ):
            logging.warning(
                "No expression information available. Cannot rank chains by expression. "
            )  # type: ignore
    for cell_chains in awk_array:
        cell_chains = cast(List[ak.Record], cell_chains)

        # Split chains into VJ and VDJ chains
        chain_indices: Dict[str, Any] = {"VJ": list(), "VDJ": list()}
        for i, tmp_chain in enumerate(cell_chains):
            if "locus" not in awk_array.fields:
                continue
            if (
                tmp_chain["locus"] in AirrCell.VJ_LOCI
                and (tmp_chain["productive"] or not productive)
                and (not _is_na2(tmp_chain["junction_aa"]) or not require_junction_aa)
            ):
                chain_indices["VJ"].append(i)
            elif (
                tmp_chain["locus"] in AirrCell.VDJ_LOCI
                and (tmp_chain["productive"] or not productive)
                and (not _is_na2(tmp_chain["junction_aa"]) or not require_junction_aa)
            ):
                chain_indices["VDJ"].append(i)

        # Order chains by expression (or whatever was specified in sort_chains_by)
        for junction_type in ["VJ", "VDJ"]:
            chain_indices[junction_type] = sorted(
                chain_indices[junction_type],
                key=partial(_key_sort_chains, cell_chains, sort_chains_by),  # type: ignore
                reverse=True,
            )

        chain_indices["multichain"] = (
            len(chain_indices["VJ"]) > 2 or len(chain_indices["VDJ"]) > 2
        )
        chain_index_list.append(chain_indices)

    chain_index_awk = ak.Array(chain_index_list)
    for k in ["VJ", "VDJ"]:
        # ensure the length for VJ and VDJ is exactly 2 (such that it can be sliced later)
        # and ensure that the type is always ?int (important if all values are None)
        chain_index_awk[k] = ak.values_astype(
            ak.pad_none(chain_index_awk[k], 2, axis=1, clip=True), int
        )

    adata.obsm[key_added] = chain_index_awk

    # store metadata in .uns
    adata.uns[key_added] = {
        "model": SCIRPY_DUAL_IR_MODEL,  # can be used to distinguish different receptor models that may be added in the future.
        "productive": productive,
        "require_junction_aa": require_junction_aa,
        "airr_key": airr_key,
        "sort_chains_by": str(sort_chains_by),
    }


def _key_sort_chains(
    chains: List[Mapping], sort_chains_by: Mapping[str, Any], idx: int
) -> Sequence:
    """Get key to sort chains by expression.

    Parameters
    ----------
    chains
        List of dictionaries with chains
    keys
        Dictionary with sort keys and default values should the key not be found
    idx
        The chain index of the current chain in `chains`
    """
    chain = chains[idx]
    sort_key = []
    for k, default in sort_chains_by.items():
        try:
            v = chain[k]
            if v is None:
                v = default
        except IndexError:
            v = default
        sort_key.append(v)
    return sort_key
