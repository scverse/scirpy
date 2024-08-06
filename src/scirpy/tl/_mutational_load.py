from collections.abc import Sequence
from typing import Literal, Union

import pandas as pd

from scirpy.get import airr as get_airr
from scirpy.util import DataHandler


def simple_hamming_distance(sequence: str, germline: str, frequency=False):
    if (sequence == None) or (germline == None):
        return None
    if len(sequence) != len(germline):
        raise Exception("Sequences might not be IMGT aligned")

    distance = 0
    if frequency:
        frac_counter = len(germline)
        for i, c in enumerate(germline):
            if (c == ".") or (c == "N"):
                frac_counter -= 1
                continue
            elif (sequence[i] == ".") or (sequence[i] == "N"):
                frac_counter -= 1
                continue
            elif c != sequence[i]:
                distance += 1
        if frac_counter == 0:
            return frac_counter
        else:
            return distance / frac_counter

    else:
        for i, c in enumerate(germline):
            if (c == ".") or (c == "N"):
                continue
            if (sequence[i] == ".") or (sequence[i] == "N"):
                continue
            elif c != sequence[i]:
                distance += 1

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
    inplace: bool = True,
) -> Union[None, pd.DataFrame]:
    """\
    Calculates observable mutation by comparing sequence with germline alignment and counting differences
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
        Specify the way mutations are calculated according to IMGT unique numbering scheme (https://doi.org/10.1016/S0145-305X(02)00039-3)
        IMGT_V(D)J ->
        IMGT_V_segment ->
        subregion ->
    junction_col
        Awkward array key to access junction region information
    frequency
        Specify to obtain either total or relative counts
    {inplace}

    Returns
    -------
    Depending on the value of inplace adds a column to adata or returns a pd.Dataframe
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)
    airr_df = get_airr(params, [sequence_alignment, germline_alignment, junction_col], chains)
    if not inplace:
        mutation_df = pd.DataFrame(index=airr_df.index)

    if region == "IMGT_V(D)J":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
                #
                if frequency:
                    rel_mutation = simple_hamming_distance(
                        airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"],
                        airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"],
                        frequency=True,
                    )
                    mutations.append(rel_mutation)

                else:
                    count_mutation = simple_hamming_distance(
                        airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"],
                        airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"],
                    )

                    mutations.append(count_mutation)

            if inplace and frequency:
                params.set_obs(f"{chain}_V(D)J_mu_freq", mutations)

            if not inplace and frequency:
                mutation_df[f"{chain}_V(D)J_mu_freq"] = mutations

            if inplace and not frequency:
                params.set_obs(f"{chain}_V(D)J_mu_count", mutations)

            if not inplace and not frequency:
                mutation_df[f"{chain}_V(D)J_mu_count"] = mutations

        if not inplace:
            return mutation_df

    # calculate SHM up to nucleotide 312. Referring to the IMGT unique numbering scheme this includes:
    # fwr1, cdr1, fwr2, cdr2, fwr3, but not cdr3 and fwr4
    if region == "IMGT_V_segment":
        for chain in chains:
            mutations = []
            for row in range(len(airr_df)):
                v_region_germline = airr_df.iloc[row].loc[f"{chain}_{germline_alignment}"][:312]
                v_region_sequence = airr_df.iloc[row].loc[f"{chain}_{sequence_alignment}"][:312]

                if frequency:
                    rel_mutation = simple_hamming_distance(v_region_sequence, v_region_germline, frequency=True)
                    mutations.append(rel_mutation)
                else:
                    count_mutation = simple_hamming_distance(v_region_sequence, v_region_germline)
                    mutations.append(count_mutation)

            if inplace and frequency:
                params.set_obs(f"{chain}_v_segment_mu_freq", mutations)

            if not inplace and frequency:
                mutation_df[f"{chain}_v_segment_mu_freq"] = mutations

            if inplace and not frequency:
                params.set_obs(f"{chain}_v_segment_mu_count", mutations)

            if not inplace and not frequency:
                mutation_df[f"{chain}_v_segment_mu_count"] = mutations

        if not inplace:
            return mutation_df

    if region == "subregion":
        subregion_df = get_airr(params, ["fwr1", "fwr2", "fwr3", "fwr4", "cdr1", "cdr2", "cdr3"], chains)
        for chain in chains:
            airr_df[f"{chain}_junction_len"] = [len(a) for a in airr_df[f"{chain}_junction"]]

            mutation_fwr1 = []
            mutation_fwr2 = []
            mutation_fwr3 = []
            mutation_fwr4 = []
            mutation_cdr1 = []
            mutation_cdr2 = []
            mutation_cdr3 = []

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

                if frequency:
                    fwr1_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr1"], fwr1_germline, frequency=True
                    )
                    cdr1_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr1"], cdr1_germline, frequency=True
                    )
                    fwr2_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr2"], fwr2_germline, frequency=True
                    )
                    cdr2_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr2"], cdr2_germline, frequency=True
                    )
                    fwr3_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr3"], fwr3_germline, frequency=True
                    )
                    cdr3_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_cdr3"], cdr3_germline, frequency=True
                    )
                    fwr4_mu_rel = simple_hamming_distance(
                        subregion_df.iloc[row].loc[f"{chain}_fwr4"], fwr4_germline, frequency=True
                    )

                    mutation_fwr1.append(fwr1_mu_rel)
                    mutation_fwr2.append(fwr2_mu_rel)
                    mutation_fwr3.append(fwr3_mu_rel)
                    mutation_fwr4.append(fwr4_mu_rel)
                    mutation_cdr1.append(cdr1_mu_rel)
                    mutation_cdr2.append(cdr2_mu_rel)
                    mutation_cdr3.append(cdr3_mu_rel)

                else:
                    fwr1_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_fwr1"], fwr1_germline)
                    cdr1_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_cdr1"], cdr1_germline)
                    fwr2_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_fwr2"], fwr2_germline)
                    cdr2_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_cdr2"], cdr2_germline)
                    fwr3_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_fwr3"], fwr3_germline)
                    cdr3_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_cdr3"], cdr3_germline)
                    fwr4_mu_count = simple_hamming_distance(subregion_df.iloc[row].loc[f"{chain}_fwr4"], fwr4_germline)

                    mutation_fwr1.append(fwr1_mu_count)
                    mutation_fwr2.append(fwr2_mu_count)
                    mutation_fwr3.append(fwr3_mu_count)
                    mutation_fwr4.append(fwr4_mu_count)
                    mutation_cdr1.append(cdr1_mu_count)
                    mutation_cdr2.append(cdr2_mu_count)
                    mutation_cdr3.append(cdr3_mu_count)

            if not inplace and frequency:
                mutation_df[f"{chain}_fwr1_mu_freq"] = mutation_fwr1
                mutation_df[f"{chain}_cdr1_mu_freq"] = mutation_cdr1
                mutation_df[f"{chain}_fwr2_mu_freq"] = mutation_fwr2
                mutation_df[f"{chain}_cdr2_mu_freq"] = mutation_cdr2
                mutation_df[f"{chain}_fwr3_mu_freq"] = mutation_fwr3
                mutation_df[f"{chain}_cdr3_mu_freq"] = mutation_cdr3
                mutation_df[f"{chain}_fwr4_mu_freq"] = mutation_fwr4

            if inplace and frequency:
                params.set_obs(f"{chain}_fwr1_mu_freq", mutation_fwr1)
                params.set_obs(f"{chain}_cdr1_mu_freq", mutation_cdr1)
                params.set_obs(f"{chain}_fwr2_mu_freq", mutation_fwr2)
                params.set_obs(f"{chain}_cdr2_mu_freq", mutation_cdr2)
                params.set_obs(f"{chain}_fwr3_mu_freq", mutation_fwr3)
                params.set_obs(f"{chain}_cdr3_mu_freq", mutation_cdr3)
                params.set_obs(f"{chain}_fwr4_mu_freq", mutation_fwr4)

            if inplace and not frequency:
                params.set_obs(f"{chain}_fwr1_mu_count", mutation_fwr1)
                params.set_obs(f"{chain}_cdr1_mu_count", mutation_cdr1)
                params.set_obs(f"{chain}_fwr2_mu_count", mutation_fwr2)
                params.set_obs(f"{chain}_cdr2_mu_count", mutation_cdr2)
                params.set_obs(f"{chain}_fwr3_mu_count", mutation_fwr3)
                params.set_obs(f"{chain}_cdr3_mu_count", mutation_cdr3)
                params.set_obs(f"{chain}_fwr4_mu_count", mutation_fwr4)

            if not inplace and not frequency:
                mutation_df[f"{chain}_fwr1_mu_count"] = mutation_fwr1
                mutation_df[f"{chain}_cdr1_mu_count"] = mutation_cdr1
                mutation_df[f"{chain}_fwr2_mu_count"] = mutation_fwr2
                mutation_df[f"{chain}_cdr2_mu_count"] = mutation_cdr2
                mutation_df[f"{chain}_fwr3_mu_count"] = mutation_fwr3
                mutation_df[f"{chain}_cdr3_mu_count"] = mutation_cdr3
                mutation_df[f"{chain}_fwr4_mu_count"] = mutation_fwr4

        if not inplace:
            return mutation_df
