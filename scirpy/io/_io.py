import pandas as pd
import json
from anndata import AnnData
from ._datastructures import TcrCell, TcrChain
from typing import Collection, Sequence, Union
import numpy as np
from glob import iglob
import pickle
import os.path
from . import _tracerlib
import sys
import airr
from ..util import _doc_params, _is_na, _is_true

# patch sys.modules to enable pickle import.
# see https://stackoverflow.com/questions/2121874/python-pckling-after-changing-a-modules-directory
sys.modules["tracerlib"] = _tracerlib

doc_working_model = """\
Currently, reading data into *Scirpy* has the following limitations: 
 * only alpha- and beta :term:`TCR` chains are supported. Other chains are ignored. 
 * non-productive chains are removed
 * Each chain can contain up to two alpha and two beta chains (:term:`Dual TCR`). 
   Excess chains are removed (those with lowest read count/:term:`UMI` count) 
   and cells flagged as :term:`Multichain-cell`. 

For more information, see :ref:`tcr-model`. 
"""


def _sanitize_anndata(adata: AnnData) -> None:
    """Sanitization and sanity checks on TCR-anndata object. 
    Should be executed by every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "

    # This should always be a categorical with True / False
    has_tcr_mask = _is_true(adata.obs["has_tcr"])
    adata.obs["has_tcr"] = ["True" if x else "False" for x in has_tcr_mask]
    adata._sanitize()


@_doc_params(doc_working_model=doc_working_model)
def from_tcr_objs(tcr_objs: Collection[TcrCell]) -> AnnData:
    """\
    Convert a collection of :class:`TcrCell` objects to an :class:`~anndata.AnnData`. 

    This is useful for converting arbitrary data formats into 
    the scirpy :ref:`data-structure`. 
    
    {doc_working_model}

    Parameters
    ----------
    tcr_objs


    Returns
    -------
    :class:`~anndata.AnnData` object with TCR information in `obs`. 

    """
    tcr_df = pd.DataFrame.from_records(
        (_process_tcr_cell(x) for x in tcr_objs), index="cell_id"
    )
    adata = AnnData(obs=tcr_df, X=np.empty([tcr_df.shape[0], 0]))
    _sanitize_anndata(adata)
    return adata


@_doc_params(doc_working_model=doc_working_model)
def _process_tcr_cell(tcr_obj: TcrCell) -> dict:
    """\
    Process a TcrCell object into a dictionary according
    to wour working model of TCRs. 

    {doc_working_model}
    
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
    chain_dict = dict()
    for c in ["TRA", "TRB"]:
        # sorting subordinately by raw and cdr3 ensures consistency
        # between load from json and load from csv.
        tmp_chains = sorted(
            [x for x in tcr_obj.chains if x.chain_type == c and x.is_productive],
            key=lambda x: (x.expr, x.expr_raw, x.cdr3),
            reverse=True,
        )
        res_dict["multi_chain"] = res_dict.get("multi_chain", False) | (
            len(tmp_chains) > 2
        )
        # slice to max two chains
        tmp_chains = tmp_chains[:2]
        # add None if less than two chains
        tmp_chains += [None] * (2 - len(tmp_chains))
        chain_dict[c] = tmp_chains

    for key in [
        "cdr3",
        "junction_ins",
        "expr",
        "v_gene",
        "d_gene",
        "j_gene",
        "c_gene",
        "cdr3_nt",
    ]:
        for c, tmp_chains in chain_dict.items():
            for i, chain in enumerate(tmp_chains):
                res_dict["{}_{}_{}".format(c, i + 1, key)] = (
                    getattr(chain, key) if chain is not None else None
                )

    # in some weird reasons, it can happen that a cell has been called from
    # TCR-seq but no TCR seqs have been found. `has_tcr` should be equal
    # to "at least one productive chain"
    res_dict["has_tcr"] = not (
        _is_na(res_dict["TRA_1_cdr3"]) and _is_na(res_dict["TRB_1_cdr3"])
    )

    if _is_na(res_dict["TRA_1_cdr3"]):
        assert _is_na(
            res_dict["TRA_2_cdr3"]
        ), "There can't be a secondary chain if there is no primary one: {}".format(
            res_dict
        )
    if _is_na(res_dict["TRB_1_cdr3"]):
        assert _is_na(
            res_dict["TRB_2_cdr3"]
        ), "There can't be a secondary chain if there is no primary one: {}".format(
            res_dict
        )

    return res_dict


def _read_10x_vdj_json(path: str, filtered: bool = True) -> AnnData:
    """Read TCR data from a 10x genomics `all_contig_annotations.json` file"""
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

    return from_tcr_objs(tcr_objs.values())


def _read_10x_vdj_csv(path: str, filtered: bool = True) -> AnnData:
    """Read TCR data from a 10x genomics `_contig_annotations.csv` file """
    df = pd.read_csv(path)

    tcr_objs = {}
    if filtered:
        df = df.loc[_is_true(df["is_cell"]) & _is_true(df["high_confidence"]), :]
    for barcode, cell_df in df.groupby("barcode"):
        tcr_obj = TcrCell(barcode)
        for _, chain_series in cell_df.iterrows():
            tcr_obj.add_chain(
                TcrChain(
                    chain_type=chain_series["chain"]
                    if chain_series["chain"] in ["TRA", "TRB"]
                    else "other",
                    cdr3=chain_series["cdr3"],
                    cdr3_nt=chain_series["cdr3_nt"],
                    expr=chain_series["umis"],
                    expr_raw=chain_series["reads"],
                    is_productive=chain_series["productive"],
                    v_gene=chain_series["v_gene"],
                    d_gene=chain_series["d_gene"],
                    j_gene=chain_series["j_gene"],
                    c_gene=chain_series["c_gene"],
                )
            )

        tcr_objs[barcode] = tcr_obj

    return from_tcr_objs(tcr_objs.values())


@_doc_params(doc_working_model=doc_working_model)
def read_10x_vdj(path: str, filtered: bool = True) -> AnnData:
    """\
    Read TCR data from 10x Genomics cell-ranger output. 

    Supports `all_contig_annotations.json` and
    `{{all,filtered}}_contig_annotations.csv`. 

    If the `json` file is available, it is preferable as it 
    contains additional information about V(D)J-junction insertions. Other than 
    that there should be no difference.  

    {doc_working_model}

    Parameters
    ----------
    path
        Path to `filterd_contig_annotations.csv`, `all_contig_annotations.csv` or
        `all_contig_annotations.json`. 
    filtered
        Only keep filtered contig annotations (i.e. `is_cell` and `high_confidence`).
        If using `filtered_contig_annotations.csv` already, this option
        is futile. 

    Returns
    -------
    AnnData object with TCR data in `obs` for each cell. For more details see
    :ref:`data-structure`.   
    """
    if path.endswith("json"):
        return _read_10x_vdj_json(path, filtered)
    else:
        return _read_10x_vdj_csv(path, filtered)


@_doc_params(doc_working_model=doc_working_model)
def read_tracer(path: str) -> AnnData:
    """\
    Read data from `TraCeR <https://github.com/Teichlab/tracer>`_ (:cite:`Stubbington2016-kh`). 

    Requires the TraCeR output directory which contains a folder for each cell. 
    Unfortunately the results files generated by `tracer summarize` do not
    contain all required information.

    The function will read TCR information from the `filtered_TCR_seqs/<CELL_ID>.pkl` 
    files. 

    {doc_working_model}
    
    Parameters
    ----------
    path
        Path to the TraCeR output folder.

    Returns
    -------
    AnnData object with TCR data in `obs` for each cell. For more details see
    :ref:`data-structure`.    
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
        try:
            with open(summary_file, "rb") as f:
                tracer_obj = pickle.load(f)
                chains = tracer_obj.recombinants["TCR"]
                if "A" in chains and chains["A"] is not None:
                    for tmp_chain in _process_chains(chains["A"], "TRA"):
                        tcr_obj.add_chain(tmp_chain)
                if "B" in chains and chains["B"] is not None:
                    for tmp_chain in _process_chains(chains["B"], "TRB"):
                        tcr_obj.add_chain(tmp_chain)
        except Exception as e:
            raise Exception(
                "Error loading TCR data from cell {}".format(summary_file)
            ) from e

        tcr_objs[cell_name] = tcr_obj

    if not len(tcr_objs):
        raise IOError(
            "Could not find any TraCeR *.pkl files. Make sure you are "
            "using a TraCeR output folder that looks like "
            "<CELL>/filtered_TCR_seqs/*.pkl"
        )

    return from_tcr_objs(tcr_objs.values())


@_doc_params(doc_working_model=doc_working_model)
def read_airr(path: Union[str, Sequence[str]]) -> AnnData:
    """\
    Read AIRR-compliant data. 

    Reads data organized in the `AIRR rearrangement schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_.
    
    The following columns are required:
     * `cell_id`
     * `productive`
     * `locus`
     * `consensus_count`
     * at least one of `junction_aa` or `junction`. 


    {doc_working_model}
    
    Parameters
    ----------
    path
        Path to the AIRR rearrangement tsv file. If different
        chains are split up into multiple files, these can be specified
        as a List, e.g. `["path/to/tcr_alpha.tsv", "path/to/tcr_beta.tsv"]`. 

    Returns
    -------
    AnnData object with TCR data in `obs` for each cell. For more details see
    :ref:`data-structure`.    
    """
    tcr_objs = {}

    if isinstance(path, str):
        path = [path]

    for tmp_path in path:
        reader = airr.read_rearrangement(tmp_path)
        for row in reader:
            cell_id = row["cell_id"]
            try:
                tmp_cell = tcr_objs[cell_id]
            except KeyError:
                tmp_cell = TcrCell(cell_id=cell_id)
                tcr_objs[cell_id] = tmp_cell

            try:
                # this is not an official field
                expr = row["umi_count"]
                expr_raw = row["consensus_count"]
            except KeyError:
                expr = row["consensus_count"]
                expr_raw = None

            tmp_cell.add_chain(
                TcrChain(
                    is_productive=row["productive"],
                    chain_type=row["locus"],
                    v_gene=row["v_call"] if "v_call" in row else None,
                    d_gene=row["d_call"] if "d_call" in row else None,
                    j_gene=row["j_call"] if "j_call" in row else None,
                    c_gene=row["c_call"] if "c_call" in row else None,
                    cdr3=row["junction_aa"] if "junction_aa" in row else None,
                    cdr3_nt=row["junction"] if "junction" in row else None,
                    expr=expr,
                    expr_raw=expr_raw,
                )
            )

    return from_tcr_objs(tcr_objs.values())
