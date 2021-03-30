import pandas as pd
import json
from anndata import AnnData
from ._datastructures import AirrCell
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
from ._convert_anndata import from_ir_objs, to_ir_objs
from ._util import doc_working_model, _IOLogger
from .._compat import Literal
from airr import RearrangementSchema


# patch sys.modules to enable pickle import.
# see https://stackoverflow.com/questions/2121874/python-pckling-after-changing-a-modules-directory
sys.modules["tracerlib"] = _tracerlib


def _read_10x_vdj_json(path: Union[str, Path], filtered: bool = True) -> AnnData:
    """Read IR data from a 10x genomics `all_contig_annotations.json` file"""
    logger = _IOLogger()
    with open(path, "r") as f:
        cells = json.load(f)

    ir_objs = {}
    for cell in cells:
        if filtered and not (cell["is_cell"] and cell["high_confidence"]):
            continue
        barcode = cell["barcode"]
        if barcode not in ir_objs:
            ir_obj = AirrCell(barcode, logger=logger)
            ir_objs[barcode] = ir_obj
        else:
            ir_obj = ir_objs[barcode]

        # AIRR-compliant chain dict
        chain = AirrCell.empty_chain_dict()

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

        chain["v_call"] = genes["v"]["gene"] if "v" in genes else None
        chain["d_call"] = genes["d"]["gene"] if "d" in genes else None
        chain["j_call"] = genes["j"]["gene"] if "j" in genes else None
        chain["c_call"] = genes["c"]["gene"] if "c" in genes else None

        # check if chain type is consistent
        chain_types = [g["chain"] for g in genes.values()]
        chain_type = chain_types[0] if np.unique(chain_types).size == 1 else "other"
        chain_type = chain_type if chain_type in AirrCell.VALID_LOCI else "other"

        # compute inserted nucleotides
        # VJ junction for TRA, TRG, IGK, IGL chains
        # VD + DJ junction for TRB, TRD, IGH chains
        #
        # Notes on indexing:
        # some tryouts have shown, that the indexes in the json file
        # seem to be python-type indexes (i.e. the 'end' index is exclusive).
        # Therefore, no `-1` needs to be subtracted when computing the number
        # of inserted nucleotides.
        chain["np1_length"] = None
        chain["np2_length"] = None
        if (
            chain_type in AirrCell.VJ_LOCI
            and chain["v_call"] is not None
            and chain["j_call"] is not None
        ):
            assert (
                chain["d_call"] is None
            ), "TRA, TRG or IG-light chains should not have a D region"
            chain["np1_length"] = genes["j"]["start"] - genes["v"]["end"]
        elif (
            chain_type in AirrCell.VDJ_LOCI
            and chain["v_call"] is not None
            and chain["j_call"] is not None
            and chain["d_call"] is not None
        ):
            chain["np1_length"] = genes["d"]["start"] - genes["v"]["end"]
            chain["np2_length"] = genes["j"]["start"] - genes["d"]["end"]

        chain["locus"] = chain_type
        chain["junction"] = cell["cdr3_seq"]
        chain["junction_aa"] = cell["cdr3"]
        chain["duplicate_count"] = cell["umi_count"]
        chain["consensus_count"] = cell["read_count"]
        chain["productive"] = cell["productive"]
        # TODO explicitly add as cell attributes?
        chain["is_cell"] = cell["is_cell"]
        chain["high_confidence"] = cell["high_confidence"]

        ir_obj.add_chain(chain)

    return from_ir_objs(ir_objs.values())


def _read_10x_vdj_csv(path: Union[str, Path], filtered: bool = True) -> AnnData:
    """Read IR data from a 10x genomics `_contig_annotations.csv` file """
    logger = _IOLogger()
    df = pd.read_csv(path)

    ir_objs = {}
    if filtered:
        df = df.loc[_is_true(df["is_cell"]) & _is_true(df["high_confidence"]), :]
    for barcode, cell_df in df.groupby("barcode"):
        ir_obj = AirrCell(barcode, logger=logger)
        for _, chain_series in cell_df.iterrows():
            chain_dict = AirrCell.empty_chain_dict()
            chain_dict.update(
                locus=(
                    chain_series["chain"]
                    if chain_series["chain"] in AirrCell.VALID_LOCI
                    else "other"
                ),
                junction_aa=chain_series["cdr3"],
                junction=chain_series["cdr3_nt"],
                duplicate_count=chain_series["umis"],
                consensus_count=chain_series["reads"],
                productive=_is_true(chain_series["productive"]),
                v_call=chain_series["v_gene"],
                d_call=chain_series["d_gene"],
                j_call=chain_series["j_gene"],
                c_call=chain_series["c_gene"],
                # TODO add this explicitly as cell attributes?
                is_cell=chain_series["is_cell"],
                high_confidence=chain_series["high_confidence"],
            )
            ir_obj.add_chain(chain_dict)

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
    logger = _IOLogger()
    tcr_objs = {}
    path = str(path)

    def _process_chains(chains, chain_type):
        for tmp_chain in chains:
            if tmp_chain.cdr3 == "N/A" or tmp_chain.cdr3nt == "N/A":
                # ignore chains that have no sequence
                continue

            # AIRR-rearrangement compliant chain dictionary
            chain_dict = AirrCell.empty_chain_dict()
            if tmp_chain.has_D_segment:
                assert chain_type in AirrCell.VDJ_LOCI
                assert len(tmp_chain.junction_details) == 5
                assert len(tmp_chain.summary) == 8
                chain_dict["v_call"] = tmp_chain.summary[0].split("*")[0]
                chain_dict["d_call"] = tmp_chain.summary[1].split("*")[0]
                chain_dict["j_call"] = tmp_chain.summary[2].split("*")[0]
            else:
                assert chain_type in AirrCell.VJ_LOCI
                assert len(tmp_chain.junction_details) == 3
                assert len(tmp_chain.summary) == 7
                chain_dict["v_call"] = tmp_chain.summary[0].split("*")[0]
                chain_dict["d_call"] = None
                chain_dict["j_call"] = tmp_chain.summary[1].split("*")[0]

            for call_key in ["v_call", "d_call", "j_call"]:
                if chain_dict[call_key] == "N/A":
                    chain_dict[call_key] = None

                if chain_dict[call_key] is not None:
                    assert chain_dict[call_key][3] == call_key[0].upper()

            chain_dict["np1_length"] = (
                len(tmp_chain.junction_details[1])
                if tmp_chain.junction_details[1] != "N/A"
                else None
            )
            try:
                # only in VDJ
                chain_dict["np2_length"] = (
                    len(tmp_chain.junction_details[3])
                    if tmp_chain.junction_details[3] != "N/A"
                    else None
                )
            except IndexError:
                chain_dict["np2_length"] = None

            chain_dict["locus"] = chain_type
            chain_dict["consensus_count"] = tmp_chain.TPM
            chain_dict["productive"] = tmp_chain.productive
            chain_dict["junction"] = tmp_chain.cdr3nt
            chain_dict["junction_aa"] = tmp_chain.cdr3

            yield chain_dict

    for summary_file in iglob(
        os.path.join(path, "**/filtered_TCR_seqs/*.pkl"), recursive=True
    ):
        cell_name = summary_file.split(os.sep)[-3]
        tcr_obj = AirrCell(cell_name, logger=logger)
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
        except ImportError as e:
            # except Exception as e:
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
def read_airr(
    path: Union[str, Sequence[str], Path, Sequence[Path]],
    use_umi_count_col: Union[bool, Literal["auto"]] = "auto",
    infer_locus: bool = True,
) -> AnnData:
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
    use_umi_count_col
        Whether to add UMI counts from the non-strandard (but common) `umi_count`
        column. When this column is used, the UMI counts are moved over to the
        standard `duplicate_count` column. Default: Use `umi_count` if there is
        no `duplicate_count` column present.
    infer_locus
        Try to infer the `locus` column from gene names, in case it is not specified.

    Returns
    -------
    AnnData object with IR data in `obs` for each cell. For more details see
    :ref:`data-structure`.
    """
    ir_objs = {}
    logger = _IOLogger()

    if isinstance(path, (str, Path, pd.DataFrame)):
        path = [path]

    def _decide_use_umi_count_col(chain_dict):
        """Logic to decide whether or not to use counts form the `umi_counts` column."""
        if (
            "umi_count" in chain_dict
            and use_umi_count_col == "auto"
            and "duplicate_count" not in chain_dict
        ):
            logger.warning(
                "Renaming the non-standard `umi_count` column to `duplicate_count`. "
            )  # type: ignore
            return True
        elif use_umi_count_col is True:
            return True
        else:
            return False

    for tmp_path in path:
        if isinstance(tmp_path, pd.DataFrame):
            iterator = tmp_path.to_dict(orient="records")
        else:
            iterator = airr.read_rearrangement(str(tmp_path))

        for chain_dict in iterator:
            cell_id = chain_dict.pop("cell_id")

            if _decide_use_umi_count_col(chain_dict):
                chain_dict["duplicate_count"] = RearrangementSchema.to_int(
                    chain_dict.pop("umi_count")
                )

            if infer_locus and "locus" not in chain_dict:
                chain_dict["locus"] = _infer_locus_from_gene_names(chain_dict)

            try:
                tmp_cell = ir_objs[cell_id]
            except KeyError:
                tmp_cell = AirrCell(cell_id=cell_id, logger=logger)
                ir_objs[cell_id] = tmp_cell

            tmp_cell.add_chain(chain_dict)

    return from_ir_objs(ir_objs.values())


def _infer_locus_from_gene_names(chain_dict):
    """Infer the IGMT locus name from VDJ calls"""
    tmp = [
        chain_dict["v_call"],
        chain_dict["d_call"],
        chain_dict["j_call"],
        chain_dict["c_call"],
    ]
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

    return locus


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
    logger = _IOLogger()
    changeodb = pd.read_csv(path, sep="\t", na_values=["None"])

    bcr_cells = dict()
    for _, row in changeodb.iterrows():
        cell_id = row["CELL"]
        try:
            tmp_ir_cell = bcr_cells[cell_id]
        except KeyError:
            tmp_ir_cell = AirrCell(cell_id, logger=logger)
            bcr_cells[cell_id] = tmp_ir_cell

        chain_dict = AirrCell.empty_chain_dict()

        chain_dict["v_call"] = row["V_CALL"] if not pd.isnull(row["V_CALL"]) else None
        chain_dict["d_call"] = row["D_CALL"] if not pd.isnull(row["D_CALL"]) else None
        chain_dict["j_call"] = row["J_CALL"] if not pd.isnull(row["J_CALL"]) else None
        chain_dict["c_call"] = (
            row["C_CALL"].split("*")[0] if not pd.isnull(row["C_CALL"]) else None
        )
        chain_dict["locus"] = "IG" + row["LOCUS"]

        chain_dict["np1_length"] = None
        chain_dict["np2_length"] = None
        if (
            chain_dict["locus"] in AirrCell.VJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            assert pd.isnull(
                row["D_SEQ_START"]
            ), "TRA, TRG or IG-light chains should not have a D region" + str(row)
            chain_dict["np1_length"] = row["J_SEQ_START"] - (
                row["V_SEQ_START"] + row["V_SEQ_LENGTH"]
            )  # type: ignore
        elif (
            chain_dict["locus"] in AirrCell.VDJ_LOCI
            and not pd.isnull(row["V_SEQ_START"])
            and not pd.isnull(row["D_SEQ_START"])
            and not pd.isnull(row["J_SEQ_START"])
        ):
            chain_dict["np1_length"] = row["D_SEQ_START"] - (
                row["V_SEQ_START"] + row["V_SEQ_LENGTH"]
            )  # type: ignore
            chain_dict["np2_length"] = row["J_SEQ_START"] - (
                row["D_SEQ_START"] + row["D_SEQ_LENGTH"]
            )  # type: ignore

        chain_dict["junction"] = (
            row["JUNCTION"] if not pd.isnull(row["JUNCTION"]) else None
        )
        chain_dict["junction_aa"] = (
            _translate_dna_to_protein(chain_dict["junction"])
            if chain_dict["junction"] is not None
            else None
        )
        chain_dict["consensus_count"] = row["TPM"]
        chain_dict["productive"] = row["FUNCTIONAL"]

        tmp_ir_cell.add_chain(chain_dict)

    return from_ir_objs(bcr_cells.values())


def write_airr(adata: AnnData, filename: Union[str, Path]) -> None:
    # TODO roundtrip test
    """Write Immune receptor fields from adata.obs in AIRR Rearrangement TSV format.

    Parameters:
    -----------
    adata
        annotated data matrix
    filename
        destination filename
    """
    airr_cells = to_ir_objs(adata)
    writer = airr.create_rearrangement(filename)
    for tmp_cell in airr_cells:
        for chain in tmp_cell.chains:
            writer.write(chain)
    writer.close()


def update_schema(adata) -> None:
    """Update older versions of a scirpy anndata object to the latest schema.

    Modifies adata inplace.

    Parameters
    ----------
    adata
        annotated data matrix
    """
    # TODO
    pass


def _update_schema_decorator(f):
    """Decorator that checks that anndata uses the latest schema"""
    # TODO
    pass


def to_dandelion(adata: AnnData):
    """
    Convert a scirpy-initialized AnnData object to Dandelion format using `to_ir_objs`.

    Parameters
    ----------
    adata
        annotated data matrix with :term:`IR` annotations.

    Returns
    -------
    `Dandelion` object.

    """
    try:
        import dandelion as ddl
    except:
        raise ImportError("Please install dandelion: pip install sc-dandelion.")
    airr_cells = to_ir_objs(adata)

    contig_dicts = {}
    for tmp_cell in airr_cells:
        for i, tmp_chain in enumerate(tmp_cell.chains, start=1):
            tmp_chain.update(
                {
                    "cell_id": tmp_cell.cell_id,
                    "sequence_id": f"{tmp_cell.cell_id}_contig_{i}",
                }
            )
            tmp_chain["umi_count"] = tmp_chain.pop("duplicate_count")
            contig_dicts[tmp_chain["sequence_id"]] = tmp_chain

    data = pd.DataFrame.from_dict(contig_dicts, orient="index")
    return ddl.Dandelion(ddl.load_data(data))


def from_dandelion(dandelion, transfer=False) -> AnnData:
    """Import data from dandelion :cite:`Stephenson2021`.

    Parameters
    ----------
    dandelion
        a :class:`dandelion.Dandelion` instance
    transfer
        Whether to execute :func:`dandelion.tl.transfer` to transfer all data
        to the :class:`anndata.AnnData` instance.

    Returns
    -------
    A :class:`~anndata.AnnData` instance with AIRR information stored in `obs`.
    """
    try:
        import dandelion as ddl
    except ImportError:
        raise ImportError("Please install dandelion: pip install sc-dandelion.")

    dandelion_df = dandelion.data.copy()
    # replace "unassigned" with None
    for col in dandelion_df.columns:
        dandelion_df.loc[dandelion_df[col] == "unassigned", col] = None

    adata = read_airr(dandelion_df)
    if transfer:
        ddl.tl.transfer(
            adata, dandelion
        )  # need to make a version that is not so verbose?
    return adata
