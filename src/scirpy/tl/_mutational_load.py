from collections.abc import Sequence
from typing import Literal

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
    # count number of compared characters
    num_chars = len(sequence)

    for l1, l2 in zip(sequence, germline):  # noqa: B905 (`strict` not supported by numba)
        if l1 in ignore_chars or l2 in ignore_chars:
            num_chars -= 1
        elif l1 != l2:
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
            if seq is None or germline is None:
                dist = None
            else:
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
    regions: Sequence[Literal["full", "v", "fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3"]] = (
        "full",
        "v",
        "fwr1",
        "fwr2",
        "fwr3",
        "fwr4",
        "cdr1",
        "cdr2",
        "cdr3",
    ),
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    sequence_key: str = "sequence_alignment",
    germline_key: str = "germline_alignment_d_mask",
    junction_key: str = "junction",
    ignore_chars: Sequence[str] = (".", "N"),
) -> None:
    """\
    Calculates absolute and relative mutational load of receptor sequences based on germline alignment.

    Receptor sequences MUST be IMGT-aligned and the corresponding germline sequence MUST be available
    (See `sequence_key` and `germline_key` parameters).

    IMGT-alignments can be obtained by using the `interoperability with Dandelion <https://sc-dandelion.readthedocs.io/en/latest/notebooks/5_dandelion_diversity_and_mutation-10x_data.html>`_.

    Region boundaries are implemented as described in the `shazam documentation <https://shazam.readthedocs.io/en/stable/topics/setRegionBoundaries/>`_
    which follows the `IMGT unique numbering scheme <https://doi.org/10.1016/S0145-305X(02)00039-3>`_.

    Parameters
    ----------
    {adata}
    regions
        Specify for which regions to calculate the mutational load. By default, calculate it for *all* regions.
        The segments follow the definition described in the `shazam documentation <https://shazam.readthedocs.io/en/stable/topics/setRegionBoundaries/>`_.

        * `full`: the full sequence without any sub-regions/divisions
        * `v`: Only V_segment (Nucleotides 1 to 312)
        * `fwr1`: Positions 1 to 78.
        * `cdr1`: Positions 79 to 114.
        * `fwr2`: Positions 115 to 165.
        * `cdr2`: Positions 166 to 195.
        * `fwr3`: Positions 196 to 312.
        * `cdr3`: Positions 313 to (313 + juncLength - 6) since the junction sequence includes (on the left) the last codon from FWR3 and (on the right) the first codon from FWR4.
        * `fwr4`: Positions (313 + juncLength - 6 + 1) to the end of the sequence.
    {airr_mod}
    {airr_key}
    chain_idx_key
        Key to select chain indices
    sequence_key
        Awkward array key to access sequence alignment information. The sequence must be IMGT-aligned.
    germline_key
        Awkward array key to access germline alignment information. This must be the TMGT germline reference.
        It is recommended to mask the d-segment with `N`s (see `Yaari et al. (2015) <https://doi.org/10.1186/s13073-015-0243-2>`_)
    junction_key
        Awkward array key to access the nucleotide junction sequence. This information is required to obtain
        the junction length required to calculate the coordinates of the `cdr3` and `fwr4` regions.
    ignore_chars
        A list of characters to ignore while calculating differences. The default is to ignore the following:

        * `"N"`: masked or degraded nucleotide. For instance, it is recommended to mask the D-segment, because of lower sequence quality
        * `"."`: "IMGT-gaps", distinct from "normal gaps ('-')". It is beneficial to ignore these, because sometimes
          sequence alignments are "clipped" at the beginning, which would inflate the mutaiton count.

    Returns
    -------
    A value for each chain is stored in the awkward array used as input (typically `adata.obsm["airr"]`) under the keys
    `"{{region}}_mutation_count"` and `"{{region}}_mutation_freq" for each region specified in the `regions` parameter.
    The mutational load for the `"full"` region is stored in `mutation_count` and `mutation_freq`, respectively
    (i.e. without the `{{region}}` prefix). Use :func:`scirpy.get.airr` to retrieve the values as a Dataframe.
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
