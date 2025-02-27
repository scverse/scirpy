from collections.abc import Sequence
from importlib.util import find_spec

import awkward as ak
import numba as nb
import pandas as pd

from scirpy.util import DataHandler


@nb.njit
def _hamming_distance(sequence: str | None, germline: str | None, ignore_chars: tuple[str]):
    """Compute the hamming distance between two strings. Characters in `ignore_chars` are not counted towards the distance"""
    if sequence is None or germline is None:
        return None
    if len(sequence) != len(germline):
        raise ValueError("Sequences might not be IMGT aligned")
    ignore_chars: tuple[str] = (".", "N")
    distance = 0
    num_chars = len(sequence)

    for l1, l2 in zip(sequence, germline):  # noqa: B905 (`strict` not supported by numba)
        if l1 in ignore_chars or l2 in ignore_chars:
            num_chars -= 1
        if l1 != l2:
            distance += 1

    if num_chars == 0:
        # no useful comparison has been performed -- return None -> can be used as a flag for filtering
        return None

    return distance


@nb.njit()
def _apply_hamming_to_chains(
    ab: ak.ArrayBuilder,
    start: int | None = None,
    end: int | None = None,
    *,
    add_junction_length: bool = False,
    arr_sequence: ak.Array,
    arr_germline: ak.Array,
    arr_junction: ak.Array,
    ignore_chars: tuple[str],
) -> ak.ArrayBuilder:
    """
    Compute the (abolute) mutational load between IMGT aligned sequence and germline for all chains in the scirpy-formatted
    AIRR Awkward array.

    `(start, end)` refer to a slice to select subregions of the alignment.
    If `add_junction` is True, the length of the `junction` sequence is added to the `end` index.
    """
    assert len(arr_sequence) == len(arr_germline) == len(arr_junction)
    for r_seq, r_germ, r_junc in zip(  # noqa: B905 (`strict` not supported by numba)
        arr_sequence, arr_germline, arr_junction
    ):
        assert len(r_seq) == len(r_germ) == len(r_junc)
        ab.begin_list()
        for seq, germline, junction in zip(r_seq, r_germ, r_junc):  # noqa: B905 (`strict` not supported by numba)
            if add_junction_length:
                end += len(junction)
            ab.append(_hamming_distance(seq[start:end], germline[start:end], ignore_chars))
        ab.end_list()
    return ab


@DataHandler.inject_param_docs()
def mutational_load(
    adata: DataHandler.TYPE,
    *,
    regions: Sequence[str] = ("full", "v", "fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3"),
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    sequence_key: str = "sequence_alignment",
    germline_key: str = "germline_alignment_d_mask",
    junction_key: str = "junction",
    ignore_chars: Sequence[str] = (".", "N"),
    frequency: bool = False,
) -> None | pd.DataFrame:
    """\
    Calculates observable mutation by comparing sequences with corresponding germline alignments and counting differences.
    Needs germline alignment information, which can be obtained by using the interoperability to Dandelion (https://sc-dandelion.readthedocs.io/en/latest/notebooks/5_dandelion_diversity_and_mutation-10x_data.html)

    Parameters
    ----------
    {adata}
    {airr_mod}
    {airr_key}
    chain_idx_key
        Key to select chain indices
    sequence_key
        Awkward array key to access sequence alignment information
    germline_key
        Awkward array key to access germline alignment information -> best practice mask D-gene segment (https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-015-0243-2)
    chains
        One or multiple chains from which to use CDR3 sequences
    region
        Specify the way mutations of the V segment are calculated according to IMGT unique numbering scheme (https://doi.org/10.1016/S0145-305X(02)00039-3)
        * `"IMGT_V(D)J"` - Includes the entire available sequence and germline alignment without any sub-regions/divisions
        * `"IMGT_V_segment"` - Only V_segment as defined by the IMGT unique numbering scheme is included => Nucleotide 1 to 312
        * `"subregion"` - Similar to IMGT_V(D)J, but independent calculation of each subregion (CDR1-3 and FWR1-4, respectively)
    junction_col
        Awkward array key to access junction region information
    frequency
        If `True`, compute relative counts in addition to absolute counts (requires `pyarrow` as optional dependency).
    ignore_chars
        A list of characters to ignore while calculating differences. The default s to ignore the following:
        * `"N"` - masked or degraded nucleotide, i.e. D-segment is recommended to mask, because of lower sequence quality
        * `"."` - "IMGT-gaps", distinct from "normal gaps ('-')" => beneficial to ignore, because sometimes sequence alignments are "clipped" at the beginning, which would cause artificial mutations
    {inplace}

    Returns
    -------
    Depending on the value of inplace adds a column to adata `.obs` or returns a pd.DataFrame
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    ignore_chars = tuple(ignore_chars)

    if frequency:
        if find_spec("pyarrow") is None:
            raise ImportError(
                "Calculating frequencies requires `pyarrow` as optional dependency. Run `pip install pyarrow` to install it."
            ) from None

    arrays = {
        "arr_sequence": params.airr[sequence_key],
        "arr_germline": params.airr[germline_key],
        "arr_junction": params.airr[junction_key],
    }

    if "full" in regions:
        params.airr["mutation_count"] = _apply_hamming_to_chains(ak.ArrayBuilder(), None, None, **arrays).snapshot()

    if "v" in regions:
        # calculate SHM up to nucleotide 312. Referring to the IMGT unique numbering scheme this includes:
        # fwr1, cdr1, fwr2, cdr2, fwr3, but not cdr3 and fwr4
        params.airr["v_mutation_count"] = _apply_hamming_to_chains(ak.ArrayBuilder(), 0, 312, **arrays).snapshot()

    regions = {
        "fwr1": {"start": 0, "end": 78},
        "cdr1": (78, 114),
        "fwr2": (114, 165),
        "cdr2": (165, 195),
        "fwr3": (195, 312),
        # "cdr3": (312, 312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6),
        # "fwr4": (
        #     312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6,
        #     len(airr_df.iloc[row].loc[f"{chain}_{germline_key}"]),
        # ),
    }
    for region, (start, end) in regions.items():
        if region in regions:
            params.airr[f"{region}_mutation_count"] = _apply_hamming_to_chains(
                ak.ArrayBuilder(),
                start,
                end,
                # sequence_key=sequence_key,
                # germline_key=germline_key,
                # ignore_chars=ignore_chars,
            ).snapshot()

    # calculate frequencies
    if frequency:
        for region in regions:
            # TODO
            if region in ["fwr4", "cdr3"]:
                continue
            key = "" if region == "full" else f"{region}_"
            params.airr[f"{key}mutation_freq"] = params.airr[f"{key}mutation_count"] / ak.str.length(
                params.airr[sequence_key]
            )
