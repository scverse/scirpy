from collections.abc import Sequence
from collections import defaultdict
from typing import Literal, Union

import numpy as np
import pandas as pd

from scirpy.get import airr as get_airr
from scirpy.util import DataHandler


def simple_hamming_distance(sequence: str, germline: str, frequency: bool, ignore_chars: list[str]):
    if not sequence or not germline or len(sequence) != len(germline):
        raise ValueError("Sequences might not be IMGT aligned")

    distance = 0
    num_chars = len(sequence)

    for l1, l2 in zip(sequence, germline):
        if l1 in ignore_chars or l2 in ignore_chars:
            num_chars -= 1

        elif l1 != l2:
            distance += 1

    if num_chars == 0:
        return np.nan  # can be used as a flag for filtering

    elif frequency:
        return distance / num_chars

    else:
        return distance


@DataHandler.inject_param_docs()
def mutational_load(
    adata: DataHandler.TYPE,
    *,
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    sequence_alignment: str = "sequence_alignment",
    germline_alignment: str = "germline_alignment_d_mask",
    chains: Union[
        Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"],
        Sequence[Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"]],
    ] = "VDJ_1",
    region: Literal["IMGT_V(D)J", "IMGT_V_segment", "subregion"] = "IMGT_VDJ",
    junction_col: str = "junction",
    frequency: bool = True,
    ignore_chars: list[str] = "None",
    inplace: bool = True,
) -> Union[None, pd.DataFrame]:
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
    sequence_alignment
        Awkward array key to access sequence alignment information
    germline_alignment
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
    airr_df = get_airr(params, [sequence_alignment, germline_alignment, junction_col], chains)
    if not inplace:
        mutation_df = pd.DataFrame(index=airr_df.index)

    if frequency:
        frequency_string = "mu_freq"
    else:
        frequency_string = "mu_count"
    if ignore_chars == "None":
        ignore_chars = [".", "N"]

    if region == "IMGT_V(D)J":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
                if (airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"] is None) or (airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"] is None):
                    mutations.append(np.nan)
                else:
                    mutation = simple_hamming_distance(
                        airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"],
                        airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"],
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                    mutations.append(mutation)

            if inplace:
                params.set_obs(f"{chain}_IMGT_V(D)J_{frequency_string}", mutations)

            else:
                mutation_df[f"{chain}_IMGT_V(D)J_{frequency_string}"] = mutations
        try:
            return mutation_df
        except:
            return None

    # calculate SHM up to nucleotide 312. Referring to the IMGT unique numbering scheme this includes:
    # fwr1, cdr1, fwr2, cdr2, fwr3, but not cdr3 and fwr4
    if region == "IMGT_V_segment":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
                if (airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"] is None) or (airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"] is None):
                    mutations.append(np.nan)
                else:
                    v_region_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][:312]
                    v_region_sequence = airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"][:312]

                    mutation = simple_hamming_distance(
                        v_region_sequence, v_region_germline, frequency=frequency, ignore_chars=ignore_chars
                    )
                    mutations.append(mutation)

            if inplace:
                params.set_obs(f"{chain}_v_segment_{frequency_string}", mutations)

            else:
                mutation_df[f"{chain}_v_segment_{frequency_string}"] = mutations
        try:
            return mutation_df
        except:
            return None

    if region == "subregion":
        #subregion_df = get_airr(params, ["fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3"], chains)

        for chain in chains:
            airr_df[f"{chain}_junction_len"] = [len(a) for a in airr_df[f"{chain}_junction"]]

            mutation_dict = defaultdict(list)

            for row in range(len(airr_df)):
                if (airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"] is None) or (airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"] is None):
                    for k in list(mutation_dict.keys()):
                        mutation_dict[k].append(np.nan)
                else:
                    regions = {
                        "fwr1": (0, 78),
                        "cdr1": (78, 114),
                        "fwr2": (114, 165),
                        "cdr2": (165, 195),
                        "fwr3": (195, 312),
                        "cdr3": (312, 312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6),
                        "fwr4": (
                            312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6,
                            len(airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"]),
                            ),
                        }

                    for v_region, coordinates in regions.items():
                        mutation_dict[v_region].append(
                            simple_hamming_distance(
                                airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"][slice(*coordinates)],
                                airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][slice(*coordinates)],
                                frequency=frequency,
                                ignore_chars=ignore_chars,
                            )
                        )

            for key in mutation_dict:
                if inplace:
                    params.set_obs(f"{chain}_{key}_{frequency_string}", mutation_dict[key])
                if not inplace:
                    mutation_df[f"{chain}_{key}_{frequency_string}"] = mutation_dict[key]

        try:
            return mutation_df
        except:
            return None
