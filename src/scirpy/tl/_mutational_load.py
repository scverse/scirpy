from collections.abc import Sequence
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
    ignore_chars: list[str] = [".", "N"],
    inplace: bool = True,
) -> Union[None, pd.DataFrame]:
    """\
    Calculates observable mutation by comparing sequence with germline alignment and counting differences
    Needs germline alignment information, which can be obtained by using the interoperability to Dandelion (https://sc-dandelion.readthedocs.io/en/latest/notebooks/5_dandelion_diversity_and_mutation-10x_data.html)
    Behaviour: N- (masked/decayed base calls) and .-(gaps) characters are ignored

    Parameters
    ----------
    {adata}
    {airr_mod}
    {airr_key}
    chain_idx_key
        key to select chain indices
    sequence_alignment
        Awkward array key to access sequence alignment information
    germline_alignment
        Awkward array key to access sequence alignment information -> best practice to select D gene-masked alignment
    chains
        One or multiple chains from which to use CDR3 sequences
    region
        Specify the way mutations of the V segment are calculated according to IMGT unique numbering scheme (https://doi.org/10.1016/S0145-305X(02)00039-3)
        IMGT_V(D)J -> Includes the entire available sequence and germline alignment without any sub-regions/divisions
        IMGT_V_segment -> Only V_segment as defined by the IMGT unique numbering scheme is included => Nucleotide 1 to 312
        subregion -> same as IMGT_V(D)J, but distinguish subregions into CDR1-3 and FWR1-4
    junction_col
        Awkward array key to access junction region information
    frequency
        Specify to obtain either total or relative counts
    ignore_chars
        A list of characters to ignore while calculating differences:
        N are masked or degraded nucleotide
        . are gaps => beneficial to ignore as sometimes more gaps appear in sequence in respect to germline
    {inplace}

    Returns
    -------
    Depending on the value of inplace adds a column to adata or returns a pd.DataFrame
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    airr_df = get_airr(params, [sequence_alignment, germline_alignment, junction_col], chains)
    if not inplace:
        mutation_df = pd.DataFrame(index=airr_df.index)

    if frequency:
        frequency_string = "mu_freq"
    else:
        frequency_string = "mu_count"

    if region == "IMGT_V(D)J":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
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
                return mutation_df

    # calculate SHM up to nucleotide 312. Referring to the IMGT unique numbering scheme this includes:
    # fwr1, cdr1, fwr2, cdr2, fwr3, but not cdr3 and fwr4
    if region == "IMGT_V_segment":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
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
                return mutation_df

    if region == "subregion":
        subregion_df = get_airr(params, ["fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3"], chains)
        for chain in chains:
            airr_df[f"{chain}_junction_len"] = [len(a) for a in airr_df[f"{chain}_junction"]]

            mutation_dict = {"fwr1": [], "fwr2": [], "fwr3": [], "fwr4": [], "cdr1": [], "cdr2": [], "cdr3": []}

            for row in range(len(airr_df)):
                fwr1_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][:78]
                cdr1_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][78:114]
                fwr2_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][114:165]
                cdr2_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][165:195]
                fwr3_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][195:312]
                cdr3_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][
                    312 : (312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6)
                ]
                fwr4_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][
                    (312 + airr_df.iloc[row].loc[f"{chain}_junction_len"] - 6) :
                ]

                mutation_dict["fwr1"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr1"],
                        fwr1_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["cdr1"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr1"],
                        cdr1_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["fwr2"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr2"],
                        fwr2_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["cdr2"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr2"],
                        cdr2_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["fwr3"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr3"],
                        fwr3_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["cdr3"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr3"],
                        cdr3_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )
                mutation_dict["fwr4"].append(
                    simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr4"],
                        fwr4_germline,
                        frequency=frequency,
                        ignore_chars=ignore_chars,
                    )
                )

            for key in mutation_dict:
                if inplace:
                    params.set_obs(f"{chain}_{key}_{frequency_string}", mutation_dict[key])
                if not inplace:
                    mutation_df[f"{chain}_{key}_{frequency_string}"] = mutation_dict[key]

            if not inplace:
                return mutation_df
