import operator
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from types import MappingProxyType
from typing import Any

import awkward as ak
import numba as nb
import numpy as np
from scanpy import logging

from scirpy.io._datastructures import AirrCell
from scirpy.util import DataHandler

SCIRPY_DUAL_IR_MODEL = "scirpy_dual_ir_v0.13"
# make these constants available to numba
_VJ_LOCI = tuple(AirrCell.VJ_LOCI)
_VDJ_LOCI = tuple(AirrCell.VDJ_LOCI)


@DataHandler.inject_param_docs()
def index_chains(
    adata: DataHandler.TYPE,
    *,
    filter: Callable[[ak.Array], bool] | Sequence[str | Callable[[ak.Array], bool]] = (
        "productive",
        "require_junction_aa",
    ),
    sort_chains_by: Mapping[str, Any] = MappingProxyType(
        # Since AIRR version v1.4.1, `duplicate_count` is deprecated in favor of `umi_count`.
        # We still keep it as sort key for backwards compatibility
        {"umi_count": 0, "duplicate_count": 0, "consensus_count": 0, "junction": "", "junction_aa": ""}
    ),
    airr_mod: str = "airr",
    airr_key: str = "airr",
    key_added: str = "chain_indices",
) -> None:
    """\
    Selects primary/secondary VJ/VDJ chains per cell according to the :ref:`receptor-model`.

    This function iterates through all chains stored in the :term:`awkward array` in
    `adata.obsm[airr_key]` and

     * labels chains as primary/secondary VJ/VDJ chains
     * labels cells as multichain cells

    based on the expression level of the chains and the specified filtering option.
    By default, non-productive chains and chains without a valid CDR3 amino acid sequence are filtered out.

    Additionally, chains without a valid IMGT locus are always filtered out.

    For more details, please refer to the :ref:`receptor-model` and the :ref:`data structure <data-structure>`.

    Parameters
    ----------
    {adata}
    filter
        Option to filter chains. Can be either
          * a callback function that takes the full awkward array with AIRR chains as input and returns
            another awkward array that is a boolean mask which can be used to index the former.
            (True to keep, False to discard)
          * a list of "filtering presets". Possible values are `"productive"` and `"require_junction_aa"`.
            `"productive"` removes non-productive chains and `"require_junction_aa"` removes chains that don't have
            a CDR3 amino acid sequence.
          * a list with a combination of both.

        Multiple presets/functions are combined using `and`. Filtered chains do not count towards calling "multichain" cells.
    sort_chains_by
        A list of sort keys used to determine an ordering of chains. The chain with the highest value
        of this tuple will be the primary chain, second-highest the secondary chain. If there are more chains, they
        will not be indexed, and the cell receives the "multichain" flag.
    {airr_mod}
    {airr_key}
    key_added
        Key under which the chain indicies will be stored in `adata.obsm` and metadata will be stored in `adata.uns`.

    Returns
    -------
    Nothing, but adds a dataframe to `adata.obsm[chain_indices]`
    """
    params = DataHandler(adata, airr_mod, airr_key)

    # prepare filter functions
    if isinstance(filter, Callable):
        filter = [filter]
    filter_presets = {
        "productive": lambda x: x["productive"],
        "require_junction_aa": lambda x: ~ak.is_none(x["junction_aa"], axis=-1),
    }
    filter = [filter_presets[f] if isinstance(f, str) else f for f in filter]

    # only warn if those fields are in the key (i.e. this should give a warning if those are missing with
    # default settings. If the user specifies their own dictionary, they are on their own)
    if "duplicate_count" in sort_chains_by and "consensus_count" in sort_chains_by and "umi_count" in sort_chains_by:
        if (
            "duplicate_count" not in params.airr.fields
            and "consensus_count" not in params.airr.fields
            and "umi_count" not in sort_chains_by
        ):
            logging.warning("No expression information available. Cannot rank chains by expression. ")  # type: ignore

    if "locus" not in params.airr.fields:
        raise ValueError("The scirpy receptor model requires a `locus` field to be specified in the AIRR data.")

    airr = params.airr
    logging.info("Filtering chains...")
    # Get the numeric indices pre-filtering - these are the indices we need in the final output as
    # .obsm["airr"] is and remains unfiltered.
    airr_idx = ak.local_index(airr, axis=1)
    # Filter out chains that do not match the filter criteria
    # we need an initial value that selects all chains in case filter is an empty list
    airr_idx = airr_idx[reduce(operator.and_, (f(airr) for f in filter), ak.ones_like(airr_idx, dtype=bool))]

    res = {}
    is_multichain = np.zeros(len(airr), dtype=bool)
    for chain_type, locus_names in {"VJ": AirrCell.VJ_LOCI, "VDJ": AirrCell.VDJ_LOCI}.items():
        logging.info(f"Indexing {chain_type} chains...")
        # get the indices for all VJ / VDJ chains, respectively
        idx = airr_idx[_awkward_isin(airr["locus"][airr_idx], locus_names)]

        # Now we need to sort the chains by the keys specified in `sort_chains_by`.
        # since `argsort` doesn't support composite keys, we take advantage of the
        # fact that the sorting algorithm is stable and sort the same array several times,
        # starting with the lowest priority key up to the highest priority key.
        for k, default in reversed(sort_chains_by.items()):
            # skip this round of sorting altogether if field not present
            if k in airr.fields:
                logging.debug(f"Sorting chains by {k}")
                tmp_idx = ak.argsort(ak.fill_none(airr[k][idx], default), stable=True, axis=-1, ascending=False)
                idx = idx[tmp_idx]
            else:
                logging.debug(f"Skip sorting by {k} because field not present")

        # We want the result to be lists of exactly 2 - clip if longer, pad with None if shorter.
        res[chain_type] = ak.pad_none(idx, 2, axis=1, clip=True)
        is_multichain |= ak.to_numpy(_awkward_len(idx)) > 2

    # build results
    logging.info("build result array")
    res["multichain"] = is_multichain

    params.adata.obsm[key_added] = ak.zip(res, depth_limit=1)  # type: ignore

    # store metadata in .uns
    params.adata.uns[key_added] = {
        "model": SCIRPY_DUAL_IR_MODEL,  # can be used to distinguish different receptor models that may be added in the future.
        "filter": str(filter),
        "airr_key": airr_key,
        "sort_chains_by": str(sort_chains_by),
    }


@nb.njit
def _awkward_len_inner(arr, ab):
    for row in arr:
        ab.append(len(row))
    return ab


def _awkward_len(arr):
    return _awkward_len_inner(arr, ak.ArrayBuilder()).snapshot()


@nb.njit()
def _awkward_isin_inner(arr, haystack, ab):
    for row in arr:
        ab.begin_list()
        for v in row:
            ab.append(v in haystack)
        ab.end_list()
    return ab


def _awkward_isin(arr, haystack):
    haystack = tuple(haystack)
    return _awkward_isin_inner(arr, haystack, ak.ArrayBuilder()).snapshot()


# For future reference, here would be two alternative implementations that are a bit
# slower, but work without the need for numba.
# def _awkward_len(arr):
#     return ak.max(ak.local_index(arr, axis=1), axis=1)
#
# def _awkward_isin(arr, haystack):
#     return reduce(operator.or_, (arr == el for el in haystack))


def _key_sort_chains(chains, sort_chains_by: Mapping[str, Any], idx: int) -> Sequence:
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
        except (IndexError, KeyError):
            v = default
        sort_key.append(v)
    return sort_key
