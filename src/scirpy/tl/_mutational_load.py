from collections.abc import Sequence

import awkward as ak
import numba as nb
import pandas as pd

from scirpy.util import DataHandler


@nb.njit
def _hamming_distance(sequence: str | None, germline: str | None, ignore_chars: tuple[str] = (".", "N")):
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
        return None  # can be used as a flag for filtering

    return distance


@nb.njit()
def _apply_hamming_to_chains(
    arr: ak.Array,
    ab: ak.ArrayBuilder,
    start: int | None = None,
    end: int | None = None,
    # add_junction: bool = False,
    *,
    # sequence_key: str = "sequence_alignment",
    # germline_key: str = "germline_alignment_d_mask",
    # junction_key: str = "junction",
    ignore_chars: tuple[str],
) -> ak.ArrayBuilder:
    """
    Compute the (abolute) mutational load between IMGT aligned sequence and germline for all chains in the scirpy-formatted
    AIRR Awkward array.

    `(start, end)` refer to a slice to select subregions of the alignment.
    If `add_junction` is True, the length of the `junction` sequence is added to the `end` index.
    """
    sequence_key: str = "sequence_alignment"
    germline_key: str = "germline_alignment_d_mask"
    for row in arr:
        ab.begin_list()
        for chain in row:
            # if add_junction:
            #     if chain[junction_key] is not None:
            #         end += len(chain[junction_key])
            #     else:
            #         # Can't compute sequence if junction not available
            #         ab.append(None)
            #         continue
            ab.append(_hamming_distance(chain[sequence_key][start:end], chain[germline_key][start:end], ignore_chars))
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
    junction_col: str = "junction",
    ignore_chars: Sequence[str] = (".", "N"),
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
        Specify to obtain either total or relative counts
    ignore_chars
        A list of characters to ignore while calculating differences. The default "None" ignors the following:
        * `"N"` - masked or degraded nucleotide, i.e. D-segment is recommended to mask, because of lower sequence quality
        * `"."` - "IMGT-gaps", distinct from "normal gaps ('-')" => beneficial to ignore, because sometimes sequence alignments are "clipped" at the beginning, which would cause artificial mutations
    {inplace}

    Returns
    -------
    Depending on the value of inplace adds a column to adata `.obs` or returns a pd.DataFrame
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    ignore_chars = tuple(ignore_chars)

    @nb.njit()
    def _apply_hamming_to_chains(
        arr: ak.Array,
        ab: ak.ArrayBuilder,
        start: int | None = None,
        end: int | None = None,
        # add_junction: bool = False,
        *,
        # sequence_key: str = "sequence_alignment",
        # germline_key: str = "germline_alignment_d_mask",
        # junction_key: str = "junction",
        ignore_chars: tuple[str],
    ) -> ak.ArrayBuilder:
        """
        Compute the (abolute) mutational load between IMGT aligned sequence and germline for all chains in the scirpy-formatted
        AIRR Awkward array.

        `(start, end)` refer to a slice to select subregions of the alignment.
        If `add_junction` is True, the length of the `junction` sequence is added to the `end` index.
        """
        for row in arr:
            ab.begin_list()
            for chain in row:
                # if add_junction:
                #     if chain[junction_key] is not None:
                #         end += len(chain[junction_key])
                #     else:
                #         # Can't compute sequence if junction not available
                #         ab.append(None)
                #         continue
                ab.append(
                    _hamming_distance(chain[sequence_key][start:end], chain[germline_key][start:end], ignore_chars)
                )
            ab.end_list()
        return ab

    if "full" in regions:
        params.airr["mutation_count"] = _apply_hamming_to_chains(
            params.airr,
            ak.ArrayBuilder(),
            None,
            None,
            # sequence_key=sequence_key,
            # germline_key=germline_key,
            # ignore_chars=ignore_chars,
        ).snapshot()
    # params.airr["mutation_freq"] = params.airr["mutation_count"] / ak.str.length(params.airr[sequence_key])

    if "v" in regions:
        # calculate SHM up to nucleotide 312. Referring to the IMGT unique numbering scheme this includes:
        # fwr1, cdr1, fwr2, cdr2, fwr3, but not cdr3 and fwr4
        params.airr["v_mutation_count"] = _apply_hamming_to_chains(
            params.airr,
            ak.ArrayBuilder(),
            0,
            312,
            # sequence_key=sequence_key,
            # germline_key=germline_key,
            # ignore_chars=ignore_chars,
        ).snapshot()

    regions = {
        "fwr1": (0, 78),
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
                params.airr,
                ak.ArrayBuilder(),
                start,
                end,
                # sequence_key=sequence_key,
                # germline_key=germline_key,
                # ignore_chars=ignore_chars,
            ).snapshot()

    # calculate frequencies
    for region in regions:
        # TODO
        if region in ["fwr4", "cdr3"]:
            continue
        key = "" if region == "full" else f"{region}_"
        params.airr[f"{key}mutation_freq"] = params.airr[f"{key}mutation_count"] / ak.str.length(
            params.airr[sequence_key]
        )
