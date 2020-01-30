import pandas as pd
import json
from scanpy import AnnData
from ._datastructures import TcrCell, TcrChain
from typing import Iterable, Collection
import numpy as np
from glob import iglob
import pickle
import os.path
from . import tracerlib
import sys

# patch sys.modules to enable pickle import.
# see https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
sys.modules["tracerlib"] = tracerlib


def _sanitize_anndata(adata: AnnData) -> None:
    """Sanitization and sanity checks on TCR-anndata object. 
    Should be executed by every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "
    adata._sanitize()


def _tcr_objs_to_anndata(tcr_objs: Collection) -> AnnData:
    """Convert a list of TcrCells to an AnnData object"""
    tcr_df = pd.DataFrame.from_records(
        (_process_tcr_cell(x) for x in tcr_objs), index="cell_id"
    )
    adata = AnnData(obs=tcr_df, X=np.empty([tcr_df.shape[0], 0]))
    _sanitize_anndata(adata)
    return adata


def _process_tcr_cell(tcr_obj: TcrCell) -> dict:
    """Filter chains to our working model of TCRs

    i.e.
     * There are only alpha and beta chains
     * each cell can contain at most two alpha and two beta chains
     * remove non-productive chains
     * if there are more than four chains, the most abundant ones will be taken. 
       Such cells will be flagged with 'multi_chain' = True

    
    Parameters
    ----------
    tcr_obj
        TcrCell object to process

    Returns
    -------
    Dictionary representing one row of the final `AnnData.obs` 
    data frame. 
    
    """
    res_dict = dict()
    res_dict["cell_id"] = tcr_obj.cell_id
    res_dict["has_tcr"] = True
    chain_dict = dict()
    for c in ["TRA", "TRB"]:
        tmp_chains = sorted(
            [x for x in tcr_obj.chains if x.chain_type == c and x.is_productive],
            key=lambda x: x.expr,
            reverse=True,
        )
        res_dict["multi_chain"] = len(tmp_chains) > 2
        # slice to max two chains
        tmp_chains = tmp_chains[:2]
        # add None if less than two chains
        tmp_chains += [None] * (2 - len(tmp_chains))
        chain_dict[c] = tmp_chains

    for key in [
        "cdr3",
        "cdr3_len",
        "junction_ins",
        "expr",
        "v_gene",
        "d_gene",
        "j_gene",
        "c_gene",
    ]:
        for c, tmp_chains in chain_dict.items():
            for i, chain in enumerate(tmp_chains):
                res_dict["{}_{}_{}".format(c, i, key)] = (
                    getattr(chain, key) if chain is not None else None
                )

    return res_dict


def read_10x_vdj(path: str, filtered: bool = True) -> AnnData:
    """Read TCR data from a 10x genomics sample.
    
    Parameters
    ----------
    path
        Path to all_contig_annotations.json
    filtered
        Only keep filtered contig annotations (= is_cell and high_confidence)


    Returns
    -------
    AnnData object with TCR data in `obs` for each cell.  
    """
    with open(path, "r") as f:
        cells = json.load(f)

    tcr_objs = {}
    for cell in cells:
        if filtered and not (cell["is_cell"] and cell["high_confidence"]):
            continue
        barcode = cell["barcode"]
        if barcode not in tcr_objs:
            tcr_obj = TcrCell(barcode)
            tcr_objs[barcode] = tcr_obj
        else:
            tcr_obj = tcr_objs[barcode]

        genes = dict()
        mapping = {
            "L-REGION+V-REGION": "v",
            "D-REGION": "d",
            "J-REGION": "j",
            "C-REGION": "c",
        }
        for annot in cell["annotations"]:
            feat = annot["feature"]
            if feat["region_type"] in mapping:
                region = mapping[feat["region_type"]]
                assert region not in genes, region
                genes[region] = dict()
                genes[region]["chain"] = feat["chain"]
                genes[region]["gene"] = feat["gene_name"]
                genes[region]["start"] = annot["contig_match_start"]
                genes[region]["end"] = annot["contig_match_end"]

        v_gene = genes["v"]["gene"] if "v" in genes else None
        d_gene = genes["d"]["gene"] if "d" in genes else None
        j_gene = genes["j"]["gene"] if "j" in genes else None
        c_gene = genes["c"]["gene"] if "c" in genes else None

        # check if chain type is consistent
        chain_types = [g["chain"] for g in genes.values()]
        chain_type = chain_types[0] if np.unique(chain_types).size == 1 else "other"
        # for now, gamma/delta is "other" as well.
        chain_type = chain_type if chain_type in ["TRA", "TRB"] else "other"

        # compute inserted nucleotides
        # VJ junction for TRA chains
        # VD + DJ junction for TRB chains
        #
        # Notes on indexing:
        # some tryouts have shown, that the indexes in the json file
        # seem to be python-type indexes (i.e. the 'end' index is exclusive).
        # Therefore, no `-1` needs to be subtracted when computing the number
        # of inserted nucleotides.
        if chain_type == "TRA" and v_gene is not None and j_gene is not None:
            assert d_gene is None, "TRA chains should not have a D region"
            inserted_nts = genes["j"]["start"] - genes["v"]["end"]
        elif (
            chain_type == "TRB"
            and v_gene is not None
            and d_gene is not None
            and j_gene is not None
        ):
            inserted_nts = (genes["d"]["start"] - genes["v"]["end"]) + (
                genes["j"]["start"] - genes["d"]["end"]
            )
            assert inserted_nts >= 0, inserted_nts
        else:
            inserted_nts = None

        tcr_obj.add_chain(
            TcrChain(
                chain_type=chain_type,
                cdr3=cell["cdr3"],
                cdr3_nt=cell["cdr3_seq"],
                expr=cell["umi_count"],
                expr_raw=cell["read_count"],
                is_productive=cell["productive"],
                v_gene=v_gene,
                d_gene=d_gene,
                j_gene=j_gene,
                c_gene=c_gene,
                junction_ins=inserted_nts,
            )
        )

    return _tcr_objs_to_anndata(tcr_objs.values())


def read_tracer(path: str) -> AnnData:
    """Read data from TraCeR. 

    Requires the TraCeR output directory containing a folder for each cell. 
    Unfortunately the results files generated by `tracer summarize` do not
    contain all required information.

    Will read from `filtered_TCR_seqs/<CELL_ID>.pkl` 
    
    Parameters
    ----------
    path
        Path to the TraCeR output folder.

    Returns
    -------
    AnnData object with TCR data in `obs` for each cell. 
    """
    tcr_objs = {}

    def _process_chains(chains, chain_type):
        for tmp_chain in chains:
            if tmp_chain.has_D_segment:
                assert chain_type == "TRB", chain_type
                assert len(tmp_chain.junction_details) == 5
                assert len(tmp_chain.summary) == 8
                v_gene = tmp_chain.summary[0].split("*")[0]
                d_gene = tmp_chain.summary[1].split("*")[0]
                j_gene = tmp_chain.summary[2].split("*")[0]
            else:
                assert chain_type == "TRA", chain_type
                assert len(tmp_chain.junction_details) == 3
                assert len(tmp_chain.summary) == 7
                v_gene = tmp_chain.summary[0].split("*")[0]
                d_gene = None
                j_gene = tmp_chain.summary[1].split("*")[0]

            v_gene = None if v_gene == "N/A" else v_gene
            d_gene = None if d_gene == "N/A" else d_gene
            j_gene = None if j_gene == "N/A" else j_gene
            assert v_gene is None or v_gene[3] == "V", v_gene
            assert d_gene is None or d_gene[3] == "D"
            assert j_gene is None or j_gene[3] == "J"

            ins_nts = (
                len(tmp_chain.junction_details[1])
                if tmp_chain.junction_details[1] != "N/A"
                else np.nan
            )
            if tmp_chain.has_D_segment:
                ins_nts += (
                    len(tmp_chain.junction_details[3])
                    if tmp_chain.junction_details[3] != "N/A"
                    else np.nan
                )

            yield TcrChain(
                chain_type,
                cdr3=tmp_chain.cdr3,
                cdr3_nt=tmp_chain.cdr3nt,
                expr=tmp_chain.TPM,
                is_productive=tmp_chain.productive,
                v_gene=v_gene,
                d_gene=d_gene,
                j_gene=j_gene,
                junction_ins=ins_nts,
            )

    for summary_file in iglob(
        os.path.join(path, "**/filtered_TCR_seqs/*.pkl"), recursive=True
    ):
        cell_name = summary_file.split(os.sep)[-3]
        tcr_obj = TcrCell(cell_name)
        with open(summary_file, "rb") as f:
            tracer_obj = pickle.load(f)
            chains = tracer_obj.recombinants["TCR"]
            if "A" in chains:
                for tmp_chain in _process_chains(chains["A"], "TRA"):
                    tcr_obj.add_chain(tmp_chain)
            if "B" in chains:
                for tmp_chain in _process_chains(chains["B"], "TRB"):
                    tcr_obj.add_chain(tmp_chain)

        tcr_objs[cell_name] = tcr_obj

    return _tcr_objs_to_anndata(tcr_objs.values())
