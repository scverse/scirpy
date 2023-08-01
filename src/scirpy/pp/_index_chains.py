import operator
from collections.abc import Mapping, Sequence
from functools import partial, reduce
from types import MappingProxyType
from typing import Any, Callable, Union

import awkward as ak
import numba as nb
import numpy as np
from scanpy import logging

from scirpy.io._datastructures import AirrCell
from scirpy.util import DataHandler, _is_na2, tqdm

SCIRPY_DUAL_IR_MODEL = "scirpy_dual_ir_v0.13"
# make these constants available to numba
_VJ_LOCI = tuple(AirrCell.VJ_LOCI)
_VDJ_LOCI = tuple(AirrCell.VDJ_LOCI)


@DataHandler.inject_param_docs()
def index_chains(
    adata: DataHandler.TYPE,
    *,
    filter: Union[Callable[[Mapping], bool], Sequence[Union[str, Callable[[Mapping], bool]]]] = (
        "productive",
        "require_junction_aa",
    ),
    sort_chains_by: Mapping[str, Any] = MappingProxyType(
        {"duplicate_count": 0, "consensus_count": 0, "junction": "", "junction_aa": ""}
    ),
    airr_mod: str = "airr",
    airr_key: str = "airr",
    key_added: str = "chain_indices",
) -> None:
    """\
    Selects primary/secondary VJ/VDJ cells per chain according to the :ref:`receptor-model`.

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
          * a callback function that takes a chain-dictionary as input and returns a boolean (True to keep, False to discard)
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
    chain_index_list = []
    params = DataHandler(adata, airr_mod, airr_key)

    # prepare filter functions
    if isinstance(filter, Callable):
        filter = [filter]
    filter_presets = {
        "productive": lambda x: x["productive"],
        "require_junction_aa": lambda x: not _is_na2(x["junction_aa"]),
    }
    filter = [filter_presets[f] if isinstance(f, str) else f for f in filter]

    # only warn if those fields are in the key (i.e. this should give a warning if those are missing with
    # default settings. If the user specifies their own dictionary, they are on their own)
    if "duplicate_count" in sort_chains_by and "consensus_count" in sort_chains_by:
        if "duplicate_count" not in params.airr.fields and "consensus_count" not in params.airr.fields:
            logging.warning("No expression information available. Cannot rank chains by expression. ")  # type: ignore

    # in chunks of 5000-10000 this is fastest. Not sure why there is additional
    # overhead when running `to_list` on the full array. It's anyway friendlier to memory this way.
    CHUNKSIZE = 5000
    for i in tqdm(range(0, len(params.airr), CHUNKSIZE)):
        cells = ak.to_list(params.airr[i : i + CHUNKSIZE])
        for cell_chains in cells:
            # cell_chains = cast(List[ak.Record], cell_chains)

            # Split chains into VJ and VDJ chains
            chain_indices: dict[str, Any] = {"VJ": [], "VDJ": []}
            for i, tmp_chain in enumerate(cell_chains):
                if all(f(tmp_chain) for f in filter) and "locus" in params.airr.fields:
                    if tmp_chain["locus"] in AirrCell.VJ_LOCI:
                        chain_indices["VJ"].append(i)
                    elif tmp_chain["locus"] in AirrCell.VDJ_LOCI:
                        chain_indices["VDJ"].append(i)

            # Order chains by expression (or whatever was specified in sort_chains_by)
            for junction_type in ["VJ", "VDJ"]:
                chain_indices[junction_type] = sorted(
                    chain_indices[junction_type],
                    key=partial(_key_sort_chains, cell_chains, sort_chains_by),  # type: ignore
                    reverse=True,
                )

            chain_indices["multichain"] = len(chain_indices["VJ"]) > 2 or len(chain_indices["VDJ"]) > 2
            chain_index_list.append(chain_indices)

    chain_index_awk = ak.Array(chain_index_list)
    for k in ["VJ", "VDJ"]:
        # ensure the length for VJ and VDJ is exactly 2 (such that it can be sliced later)
        # and ensure that the type is always ?int (important if all values are None)
        chain_index_awk[k] = ak.values_astype(
            ak.pad_none(chain_index_awk[k], 2, axis=1, clip=True),
            int,
            including_unknown=True,
        )

    params.adata.obsm[key_added] = chain_index_awk  # type: ignore

    # store metadata in .uns
    params.adata.uns[key_added] = {
        "model": SCIRPY_DUAL_IR_MODEL,  # can be used to distinguish different receptor models that may be added in the future.
        "filter": str(filter),
        "airr_key": airr_key,
        "sort_chains_by": str(sort_chains_by),
    }


@DataHandler.inject_param_docs()
def index_chains_numba(
    adata: DataHandler.TYPE,
    *,
    filter: Union[Callable[[ak.Array], bool], Sequence[Union[str, Callable[[ak.Array], bool]]]] = (
        "productive",
        "require_junction_aa",
    ),
    sort_chains_by: Mapping[str, Any] = MappingProxyType(
        {"duplicate_count": 0, "consensus_count": 0, "junction": "", "junction_aa": ""}
    ),
    airr_mod: str = "airr",
    airr_key: str = "airr",
    key_added: str = "chain_indices",
) -> None:
    """\
    Selects primary/secondary VJ/VDJ cells per chain according to the :ref:`receptor-model`.

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
          * a callback function that takes a chain-dictionary as input and returns a boolean (True to keep, False to discard)
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
    if "duplicate_count" in sort_chains_by and "consensus_count" in sort_chains_by:
        if "duplicate_count" not in params.airr.fields and "consensus_count" not in params.airr.fields:
            logging.warning("No expression information available. Cannot rank chains by expression. ")  # type: ignore

    if "locus" not in params.airr.fields:
        raise ValueError("The scirpy receptor model requires a `locus` field to be specified in the AIRR data.")

    # Filter out chains that do not match the filter criteria
    airr = params.airr
    airr_idx = ak.local_index(airr, axis=1)
    airr_idx = airr_idx[reduce(operator.and_, (f(airr) for f in filter))]

    # # from now on, only need the keys required for sorting + locus
    # keep_keys = list(sort_chains_by.keys())
    # keep_keys.append("locus")
    # airr = airr[keep_keys]

    res = {}
    is_multichain = np.zeros(len(airr), dtype=bool)
    for chain_type, locus_names in {"VJ": AirrCell.VJ_LOCI, "VDJ": AirrCell.VDJ_LOCI}.items():
        idx = airr_idx[_awkward_isin(airr["locus"][airr_idx], locus_names)]

        # sort array - take advantage of the fact that the sorting algorithm is stable.
        for k, default in reversed(sort_chains_by.items()):
            tmp_idx = ak.argsort(ak.fill_none(airr[k][idx], default), stable=True, axis=-1)
            idx = idx[tmp_idx]
        res[chain_type] = ak.pad_none(idx, 2, axis=1, clip=True)
        is_multichain &= ak.to_numpy(_awkward_len(idx)) > 2

    res["multichain"] = is_multichain
    # return res
    return ak.zip(res, depth_limit=1)

    vj = airr[_is_vj(airr)]

    # sort array
    prev_arr = vj
    for k, default in reversed(sort_chains_by.items()):
        idx = ak.argsort(ak.fill_none(prev_arr[k], default), stable=True, axis=-1)
        prev_arr = vj[idx]

    ak.Array({})
    multichain = _awkward_len(vj) > 2
    vj = ak.pad_none(idx, 2, axis=1, clip=True)

    return vj, multichain

    chain_index_awk = _index_chains_inner(airr_filtered, ak.ArrayBuilder(), sort_chains_by).snapshot()

    return chain_index_awk

    params.adata.obsm[key_added] = chain_index_awk  # type: ignore

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


@nb.njit
def _index_chains_inner(cells: ak.Array, ab: ak.ArrayBuilder, sort_chains_by: tuple):
    for cell_chains in cells:
        # Split chains into VJ and VDJ chains

        vj_indices = nb.typed.List()
        vdj_indices = nb.typed.List()
        # TODO can this be solved faster with np ufuncs?
        for i, tmp_chain in enumerate(cell_chains):
            if tmp_chain["locus"] in _VJ_LOCI:
                vj_indices.append(i)
            elif tmp_chain["locus"] in _VDJ_LOCI:
                vdj_indices.append(i)

        # Order chains by expression (or whatever was specified in sort_chains_by)
        # TODO proper sorting
        vj_indices = sorted(vj_indices, key=partial(_key_sort_chains_numba, cell_chains, sort_chains_by), reverse=True)
        vj_indices = sorted(vj_indices)
        vdj_indices = sorted(vdj_indices)

        N_CHAINS = 2
        n_vj = min(2, len(vj_indices))
        n_vdj = min(2, len(vdj_indices))

        # build awkward array with chain indices
        ab.begin_record()
        ab.field("VJ")
        ab.begin_list()
        # Add up to two chains
        for i in vj_indices[:n_vj]:
            ab.append(i)
        # pad with None such thatwe have exactly 2 entries
        for _ in range(N_CHAINS - n_vj):
            ab.append(None)
        ab.end_list()
        ab.field("VDJ")
        ab.begin_list()
        for i in vdj_indices[:n_vdj]:
            ab.append(i)
        for _ in range(N_CHAINS - n_vdj):
            ab.append(None)
        ab.end_list()
        ab.field("multichain")
        ab.boolean(len(vj_indices) > 2 or len(vdj_indices) > 2)
        ab.end_record()

    return ab


@nb.njit
def _key_sort_chains_numba(chains, sort_chains_by, idx: int) -> Sequence:
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
    sort_key = nb.typed.List()
    for k, default in sort_chains_by:
        try:
            v = chain[k]
            if v is None:
                v = default
        except (IndexError, KeyError):
            v = default
        sort_key.append(v)
    return sort_key


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
