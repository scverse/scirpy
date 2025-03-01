from collections.abc import Sequence

import awkward as ak
import numba as nb

from scirpy.util import DataHandler


@nb.njit
def _hamming_distance(sequence: str | None, germline: str | None, ignore_chars: tuple[str]):
    """
    Compute the hamming distance between two strings.

     - Characters in `ignore_chars` are not counted towards the distance.
     - Sequences are expected to be IMGT-aligned and of the same length. This is checked beforehand, therefore
       we don't check it here.
    """
    if sequence is None or germline is None:
        return None
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


@nb.njit
def _get_slice(region, junction):
    """
    Get the start and end coordinates for a given region identifier.

    The coordinates for cdr3 and fwr4 depend depend on the junction length. Therefore
    the junction sequence needs to be provided here.
    Reference: https://shazam.readthedocs.io/en/stable/topics/setRegionBoundaries/

    -1 is a sentinel for None i.e., "slice from start" or "slice to end".
    """
    if region == "full":
        return -1, -1
    elif region == "v":
        return 0, 312
    elif region == "fwr1":
        return 0, 78
    elif region == "cdr1":
        return 78, 114
    elif region == "fwr2":
        return 114, 165
    elif region == "cdr2":
        return 165, 195
    elif region == "fwr3":
        return 195, 312
    elif region == "cdr3":
        # For CDR3, we need to handle the junction length
        return 312, 312 + len(junction) - 6  # Adjusted based on junction length
    elif region == "fwr4":
        # FWR4 starts after the junction
        return 312 + len(junction) - 6, -1
    else:
        raise ValueError("Invalid region")


@nb.njit()
def _apply_hamming_to_chains(
    ab: ak.ArrayBuilder,
    ab_freq: ak.ArrayBuilder,
    region: str,
    *,
    arr_sequence: ak.Array,
    arr_germline: ak.Array,
    arr_junction: ak.Array,
    ignore_chars: tuple[str],
) -> tuple[ak.ArrayBuilder]:
    """
    Compute the absolute and relative mutational load between IMGT aligned sequence and germline for a specified region.

    `ab` and `ab_freq` are awkward array builders for the absolute and relative mutational load, respecitively.
    They will have the same structure as `arr_sequence`, `arr_germline` and `arr_junction`.

    `region` specifies the region, see `_get_slice` on how regions are transformed into coordinates.

    `arr_sequence`, `arr_germline`, and `arr_junction` are awkward arrays with *identical* structure. They
    are all slices of the scirpy formatted Awkward array in `adata.obsm`.

    `ignore_chars` is a tuple of characters that are not considered when calculating the hamming distance.
    """
    assert len(arr_sequence) == len(arr_germline) == len(arr_junction)
    for r_seq, r_germ, r_junc in zip(  # noqa: B905 (`strict` not supported by numba)
        arr_sequence, arr_germline, arr_junction
    ):
        assert len(r_seq) == len(r_germ) == len(r_junc)
        ab.begin_list(), ab_freq.begin_list()
        for seq, germline, junction in zip(r_seq, r_germ, r_junc):  # noqa: B905 (`strict` not supported by numba)
            if len(seq) != len(germline):
                raise ValueError(
                    "Length aligned sequences length does not match. Are you sure sequences are IMGT aligned?"
                )
            start, end = _get_slice(region, junction)
            # workaround: numba doesn't support slicing with `None`, see https://github.com/numba/numba/issues/7948
            # therefore setting to 0 == start and len(seq) == end
            start = 0 if start < 0 else start
            end = len(seq) if end < 0 else end
            seq_sliced = seq[start:end]
            germline_sliced = germline[start:end]
            dist = _hamming_distance(seq_sliced, germline_sliced, ignore_chars)
            ab.append(dist)
            ab_freq.append(None if dist is None else dist / len(seq_sliced))
        ab.end_list(), ab_freq.end_list()
    return ab, ab_freq


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
) -> None:
    """\
    Calculates observable mutation by comparing sequences with corresponding germline alignments and counting differences.
    Needs germline alignment information, which can be obtained by using the interoperability to Dandelion (https://sc-dandelion.readthedocs.io/en/latest/notebooks/5_dandelion_diversity_and_mutation-10x_data.html)

    https://shazam.readthedocs.io/en/stable/topics/setRegionBoundaries/

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

    for region in regions:
        key = "" if region == "full" else f"{region}_"
        ab, ab_freq = _apply_hamming_to_chains(
            ak.ArrayBuilder(),
            ak.ArrayBuilder(),
            region,
            arr_sequence=params.airr[sequence_key],
            arr_germline=params.airr[germline_key],
            arr_junction=params.airr[junction_key],
            ignore_chars=ignore_chars,
        )
        params.airr[f"{key}mutation_count"] = ab.snapshot()
        params.airr[f"{key}mutation_freq"] = ab_freq.snapshot()
