import pandas as pd
import json
from anndata import AnnData
from ._datastructures import IrCell, IrChain
from typing import Sequence, Union
import numpy as np
from glob import iglob
import pickle
import os.path
from . import _tracerlib
import sys
from pathlib import Path
import airr
from ..util import _doc_params, _is_true, _translate_dna_to_protein
from ._convert_anndata import from_ir_objs
from ._common_doc import doc_working_model


# patch sys.modules to enable pickle import.
# see https://stackoverflow.com/questions/2121874/python-pckling-after-changing-a-modules-directory
sys.modules["tracerlib"] = _tracerlib


def _read_10x_vdj_json(path: Union[str, Path], filtered: bool = True) -> AnnData:
    """Read IR data from a 10x genomics `all_contig_annotations.json` file"""
    with open(path, "r") as f:
        cells = json.load(f)

    ir_objs = {}
    for cell in cells:
        if filtered and not (cell["is_cell"] and cell["high_confidence"]):
            continue
        barcode = cell["barcode"]
        if barcode not in ir_objs:
            ir_obj = IrCell(barcode)
            ir_objs[barcode] = ir_obj
        else:
            ir_obj = ir_objs[barcode]

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
        chain_type = chain_type if chain_type in IrChain.VALID_LOCI else "other"

        # compute inserted nucleotides
        # VJ junction for TRA, TRG, IGK, IGL chains
        # VD + DJ junction for TRB, TRD, IGH chains
        #
        # Notes on indexing:
        # some tryouts have shown, that the indexes in the json file
        # seem to be python-type indexes (i.e. the 'end' index is exclusive).
        # Therefore, no `-1` needs to be subtracted when computing the number
        # of inserted nucleotides.
        if chain_type in IrChain.VJ_LOCI and v_gene is not None and j_gene is not None:
            assert (
                d_gene is None
            ), "TRA, TRG or IG-light chains should not have a D region"
            inserted_nts = genes["j"]["start"] - genes["v"]["end"]
        elif (
            chain_type in IrChain.VDJ_LOCI
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

        ir_obj.add_chain(
            IrChain(
                locus=chain_type,
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

    return from_ir_objs(ir_objs.values())


def _read_10x_vdj_csv(path: Union[str, Path], filtered: bool = True) -> AnnData:
    """Read IR data from a 10x genomics `_contig_annotations.csv` file """
    df = pd.read_csv(path)

    ir_objs = {}
    if filtered:
        df = df.loc[_is_true(df["is_cell"]) & _is_true(df["high_confidence"]), :]
    for barcode, cell_df in df.groupby("barcode"):
        ir_obj = IrCell(barcode)
        for _, chain_series in cell_df.iterrows():
            ir_obj.add_chain(
                IrChain(
                    locus=(
                        chain_series["chain"]
                        if chain_series["chain"] in IrChain.VALID_LOCI
                        else "other"
                    ),
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

        ir_objs[barcode] = ir_obj

    return from_ir_objs(ir_objs.values())


@_doc_params(doc_working_model=doc_working_model)
def read_10x_vdj(path: Union[str, Path], filtered: bool = True) -> AnnData:
    """\
    Read :term:`IR` data from 10x Genomics cell-ranger output.

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
    AnnData object with IR data in `obs` for each cell. For more details see
    :ref:`data-structure`.
    """
    path = Path(path)
    if path.suffix == ".json":
        return _read_10x_vdj_json(path, filtered)
    else:
        return _read_10x_vdj_csv(path, filtered)


@_doc_params(doc_working_model=doc_working_model)
def read_tracer(path: Union[str, Path]) -> AnnData:
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
    path = str(path)

    def _process_chains(chains, chain_type):
        for tmp_chain in chains:
            if tmp_chain.has_D_segment:
                assert chain_type in IrChain.VDJ_LOCI
                assert len(tmp_chain.junction_details) == 5
                assert len(tmp_chain.summary) == 8
                v_gene = tmp_chain.summary[0].split("*")[0]
                d_gene = tmp_chain.summary[1].split("*")[0]
                j_gene = tmp_chain.summary[2].split("*")[0]
            else:
                assert chain_type in IrChain.VJ_LOCI
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

            if tmp_chain.cdr3 == "N/A" or tmp_chain.cdr3nt == "N/A":
                # ignore chains that have no sequence
                continue

            yield IrChain(
                locus=chain_type,
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
        tcr_obj = IrCell(cell_name)
        try:
            with open(summary_file, "rb") as f:
                tracer_obj = pickle.load(f)
                chains = tracer_obj.recombinants["TCR"]
                for chain_id in "ABGD":
                    if chain_id in chains and chains[chain_id] is not None:
                        for tmp_chain in _process_chains(
                            chains[chain_id], f"TR{chain_id}"
                        ):
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

    return from_ir_objs(tcr_objs.values())


@_doc_params(doc_working_model=doc_working_model)
def read_airr(path: Union[str, Sequence[str], Path, Sequence[Path]]) -> AnnData:
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
    AnnData object with IR data in `obs` for each cell. For more details see
    :ref:`data-structure`.
    """
    ir_objs = {}

    if isinstance(path, (str, Path, pd.DataFrame)):
        path = [path]

    for tmp_path in path:
        if isinstance(tmp_path, pd.DataFrame):
            _, iterator = zip(*tmp_path.copy().iterrows())
        else:
            iterator = airr.read_rearrangement(str(tmp_path))

        for row in iterator:
            cell_id = row["cell_id"]
            try:
                tmp_cell = ir_objs[cell_id]
            except KeyError:
                tmp_cell = IrCell(cell_id=cell_id)
                ir_objs[cell_id] = tmp_cell
            try:
                try:
                    expr = row["umi_count"]  # this is not an official field
                except KeyError:
                    expr = row["duplicate_count"]
                expr_raw = row["consensus_count"]
            except KeyError:
                expr = row["consensus_count"]
                expr_raw = None
            if "locus" not in row:
                tmp = [row["v_call"], row["d_call"], row["j_call"], row["c_call"]]
                for t in tmp:
                    if t == "" or t is None or t != t:
                        tmp.remove(t)
                if all("tra" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "TRA"
                elif all("trb" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "TRB"
                elif all("trd" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "TRD"
                elif all("trg" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "TRG"
                elif all("igh" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "IGH"
                elif all("igk" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "IGK"
                elif all("igl" in x.lower() for x in tmp if not pd.isnull(x)):
                    locus = "IGL"
                else:
                    locus = None
                row["locus"] = locus
            tmp_cell.add_chain(
                IrChain(
                    is_productive=row["productive"],
                    locus=row["locus"] if "locus" in row else None,
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

    return from_ir_objs(ir_objs.values())


@_doc_params(doc_working_model=doc_working_model)
def read_bracer(path: Union[str, Path]) -> AnnData:
    """\
    Read data from `BraCeR <https://github.com/Teichlab/bracer>`_ (:cite:`Lindeman2018`).

    Requires the `changeodb.tab` file as input which is created by the
    `bracer summarise` step.

    {doc_working_model}

    Parameters
    ----------
    path
        Path to the `changeodb.tab` file.

    Returns
    -------
    AnnData object with BCR data in `obs` for each cell. For more details see
    :ref:`data-structure`.
    """
    changeodb = pd.read_csv(path, sep="\t", na_values=["None"])

    bcr_cells = dict()
    for _, row in changeodb.iterrows():
        cell_id = row["CELL"]
        try:
            tmp_ir_cell = bcr_cells[cell_id]
        except KeyError:
            tmp_ir_cell = IrCell(cell_id)
            bcr_cells[cell_id] = tmp_ir_cell

        v_gene = row["V_CALL"] if not pd.isnull(row["V_CALL"]) else None
        d_gene = row["D_CALL"] if not pd.isnull(row["D_CALL"]) else None
        j_gene = row["J_CALL"] if not pd.isnull(row["J_CALL"]) else None
        c_gene = row["C_CALL"].split("*")[0] if not pd.isnull(row["C_CALL"]) else None
        locus = "IG" + row["LOCUS"]

        if (
            locus in IrChain.VJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            assert pd.isnull(
                row["D_SEQ_START"]
            ), "TRA, TRG or IG-light chains should not have a D region" + str(row)
            inserted_nts = row["J_SEQ_START"] - (
                row["V_SEQ_START"] + row["V_SEQ_LENGTH"]
            )

        elif (
            locus in IrChain.VDJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["D_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            inserted_nts = (
                row["D_SEQ_START"] - (row["V_SEQ_START"] + row["V_SEQ_LENGTH"])
            ) + (row["J_SEQ_START"] - (row["D_SEQ_START"] + row["D_SEQ_LENGTH"]))
        else:
            inserted_nts = None

        cdr3_nt = row["JUNCTION"] if not pd.isnull(row["JUNCTION"]) else None
        cdr3_aa = _translate_dna_to_protein(cdr3_nt) if cdr3_nt is not None else None

        tmp_chain = IrChain(
            locus=locus,
            cdr3=cdr3_aa,
            cdr3_nt=cdr3_nt,
            expr=row["TPM"],
            is_productive=row["FUNCTIONAL"],
            v_gene=v_gene,
            d_gene=d_gene,
            j_gene=j_gene,
            c_gene=c_gene,
            junction_ins=inserted_nts,
        )
        tmp_ir_cell.add_chain(tmp_chain)

    return from_ir_objs(bcr_cells.values())


def read_dandelion(dandelion, import_all=False):
    try:
        import dandelion as ddl
    except:
        raise ImportError("Please install dandelion: pip install sc-dandelion.")
    adata = read_airr(dandelion.data)
    if import_all:
        ddl.tl.transfer(
            adata, dandelion
        )  # need to make a version that is not so verbose?
    return adata
