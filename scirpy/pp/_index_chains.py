from typing import Dict, List, Sequence, Tuple, cast

from anndata import AnnData
from ..io import AirrCell
from ..util import _is_na2
import awkward as ak
from scanpy import logging
from functools import partial
import pandas as pd
import itertools

# TODO do we need tests or is it enough to have this tested implicitly through all IO tests?
def index_chains(
    adata: AnnData,
    *,
    productive: bool = True,
    require_junction_aa: bool = True,
    sort_chains_by: Sequence[str] = (
        "duplicate_count",
        "consensus_count",
        "junction",
        "junction_aa",
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
        Key under which the chain indicies will be stored in `adata.obsm`.

    Returns
    -------
    Nothing, but adds a dataframe to `adata.obsm[chain_indices]`
    """
    chain_index_df = []
    awk_array = cast(ak.Array, adata.obsm[airr_key])

    # TODO only warn if those fields are in the key
    if (
        "duplicate_count" not in awk_array.fields
        and "consensus_count" not in awk_array.fields
    ):
        logging.warning(
            "No expression information available. Cannot rank chains by expression. "
        )  # type: ignore
    for cell_chains in awk_array:
        cell_chains = cast(List[Dict], cell_chains)
        chain_indices = {"VJ": list(), "VDJ": list()}
        for i, tmp_chain in enumerate(cell_chains):
            if "locus" not in tmp_chain:
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

        # TODO Add multichain calling somewhere here.
        for junction_type in ["VJ", "VDJ"]:
            chain_indices[junction_type] = sorted(
                chain_indices[junction_type],
                key=partial(_key_sort_chains, cell_chains),
                reverse=True,
            )
            # only keep the (up to) two most highly expressed chains
            chain_indices[junction_type] = chain_indices[junction_type][:2]

        res_dict = {}
        for i, junction_type in itertools.product([0, 1], ["VJ", "VDJ"]):
            res_key = f"{junction_type}_{i+1}"
            try:
                res_dict[res_key] = chain_indices[junction_type][i]
            except IndexError:
                res_dict[res_key] = None

        chain_index_df.append(res_dict)

    adata.obsm[key_added] = pd.DataFrame.from_records(chain_index_df)


def _key_sort_chains(chains, idx) -> Tuple:
    """Get key to sort chains by expression. Idx is the index of a chain in `chains`"""
    chain = chains[idx]
    sort_tuple = (
        chain.get("duplicate_count", 0),
        chain.get("consensus_count", 0),
        chain.get("junction", ""),
        chain.get("junction_aa", ""),
    )
    # replace None by -1 to make sure it comes in last
    return tuple(-1 if x is None else x for x in sort_tuple)
