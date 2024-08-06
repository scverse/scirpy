from scirpy.util import DataHandler
from typing import Callable, Literal, Union
from collections.abc import Sequence
from scirpy.get import obs_context
from scirpy.get import airr as get_airr

import numpy as np
import pandas as pd

import palmotif as palm
from IPython.display import SVG

@DataHandler.inject_param_docs()
def logoplot_cdr3_motif(
    adata: DataHandler.TYPE,
    chains: Union[
    Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"],
    Sequence[Literal["VJ_1", "VDJ_1", "VJ_2", "VDJ_2"]],
    ] = "VDJ_1",
    airr_mod="airr",
    airr_key="airr",
    chain_idx_key="chain_indices",
    cdr3_col: str = "junction_aa",
    *,
    by: Sequence[Literal["gene_segment", "clonotype", "length"]] = "length",
    target_col: Union[None, str] = None,
    gene_annotation: Union[None, list] = None,
    clonotype_id: Union[None, list] = None,
    clonotype_key: Union[None, str] = None,
    cdr_len: int,
    plot: bool = True,
    color_scheme: Sequence[Literal["nucleotide", "base_pairing", "hydrophobicity", "chemistry", "charge", "taylor",
                      "logojs", "shapely"]] = "taylor"
):
    """
    A user friendly wrapper function for the palmotif python package.
    Enables the analysis of potential amino acid motifs by displaying logo plots.

    Parameters
    ----------
    {adata}
    chains
        One or multiple chains from which to use CDR3 sequences
    {airr_mod}
    {airr_key}
    {chain_idx_key}
    cdr3_col
        key inside awkward array to retrieve junction information (aa)
    by
        Three options for convenient customisation:
        length -- compares all sequences that match the selected length
        clonotype -- compares all sequences that match the selected clonotype cluster(s)
            -> need to define `clonotype_id` and `clonotype_key`
        gene_segment -- compares all sequences that match the selected gene segment(s)
            -> need to define `gene_annotation` and `target_col`
    target_col
        key inside awkward array to retrieve gene annotation information e.g. v_call, j_call
    gene_annotation
        a list of predefined genes deemed interesting to include in a logoplot
    clonotype_id
        predefined clonotype cluster id to investigate as a logoplot
    clonotype_key
        key inside .obs column under which clonotype cluster ids are stored
    cdr_len
        Specify one exact sequence length t compute sequence motifs
    plot
        defaults to true to return a SVG logoplot for direct investigation
        set to false to retrieve the raw sequence motif for customised use
    color_scheme
        different color schemes used by palmotif. see https://github.com/agartland/palmotif/blob/master/palmotif/aacolors.py for more details
    
    Returns
    ----------
    Depending on `plot` either returns a SVG object or the calculated sequence motif as a pd.DataFrame
    """
    params = DataHandler(adata, airr_mod, airr_key, chain_idx_key)

    if by is "length":
        
        airr_df = get_airr(params, [cdr3_col], chains)
        if type(chains) == list:
            if len(chains) > 2:
                raise Exception("Only two different chains are allowed e.g. VDJ_1 and VDJ_2")

            else:
                cdr3_list = airr_df[airr_df[chains[0] + "_" + cdr3_col].str.len() == cdr_len][chains[0] + "_" + cdr3_col].to_list()
                cdr3_list += airr_df[airr_df[chains[1] + "_" + cdr3_col].str.len() == cdr_len][chains[1] + "_" + cdr3_col].to_list()
                motif = palm.compute_motif(cdr3_list)
        else:
            motif = palm.compute_motif(
                airr_df[airr_df[chains + "_" + cdr3_col].str.len() == cdr_len][chains + "_" + cdr3_col].to_list()
                )
            
        if plot == True:
            return SVG(palm.svg_logo(motif, return_str = False, color_scheme=color_scheme))
        else:
            return motif


    if by is "gene_segment":
        if target_col is None or gene_annotation is None:
            raise Exception("Please specify where the gene information is stored (`target_col`) and which genes to include (`gene_annotation`) as a list")
        if type(gene_annotation) is not list:
            gene_annotation = list(gene_annotation.split(" "))

        airr_df = get_airr(params, [cdr3_col, target_col], chains)
        if type(chains) == list:
            if len(chains) > 2:
                raise Exception("Only two different chains are allowed e.g. VDJ_1 and VDJ_2")

            cdr3_list = airr_df[(airr_df[chains[0] + "_" + target_col].isin(gene_annotation)) &
            (airr_df[chains[0] + "_" + cdr3_col].str.len() == cdr_len)][chains[0] + "_" + cdr3_col].to_list()
            cdr3_list += airr_df[(airr_df[chains[1] + "_" + target_col].isin(gene_annotation)) &
            (airr_df[chains[1] + "_" + cdr3_col].str.len() == cdr_len)][chains[1] + "_" + cdr3_col].to_list()
            motif = palm.compute_motif(cdr3_list)
            
        else:
            motif = palm.compute_motif(
            airr_df[(airr_df[chains + "_" + target_col].isin(gene_annotation)) &
            (airr_df[chains + "_" + cdr3_col].str.len() == cdr_len)][chains + "_" + cdr3_col].to_list()
        )

        if plot == True:
            return SVG(palm.svg_logo(motif, return_str = False, color_scheme=color_scheme))
        else:
            return motif
    
    if by is "clonotype":
        if clonotype_id is None or clonotype_key is None:
            raise Exception("Please select desired clonotype cluster and the name of the column where this information is stored!")
        
        if type(clonotype_id) is not list:
            clonotype_id = list(clonotype_id.split(" "))

        if type(chains) is list:
            airr_df = get_airr(params, [cdr3_col], chains)
        else:
            airr_df = get_airr(params, [cdr3_col], [chains])
        airr_df = pd.concat([airr_df, params.get_obs(clonotype_key)])
        airr_df = airr_df.loc[params.get_obs(clonotype_key).isin(clonotype_id)]

   

        if type(chains) is list:
            if len(chains) > 2:
                raise Exception("Only two different chains are allowed e.g. VDJ_1 and VDJ_2")

            else:
                cdr3_list = airr_df[airr_df[chains[0] + "_" + cdr3_col].str.len() == cdr_len][chains[0] + "_" + cdr3_col].to_list()
                cdr3_list += airr_df[airr_df[chains[1] + "_" + cdr3_col].str.len() == cdr_len][chains[1] + "_" + cdr3_col].to_list()
                motif = palm.compute_motif(cdr3_list)
        else:
            motif = palm.compute_motif(
                airr_df[airr_df[chains + "_" + cdr3_col].str.len() == cdr_len][chains + "_" + cdr3_col].to_list()
                )
            
        if plot == True:
            return SVG(palm.svg_logo(motif, return_str = False, color_scheme=color_scheme))
        else:
            return motif

    else:
        raise Exception("Invalid input for parameter `by`!")