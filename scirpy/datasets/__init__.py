from anndata import AnnData
from pathlib import Path
from ..util import _doc_params, _read_to_str
from scanpy.readwrite import read
from scanpy import settings
from textwrap import indent
import tempfile
from ..io import upgrade_schema, AirrCell, from_airr_cells
import urllib.request
import zipfile
import pandas as pd
import scanpy as sc
from datetime import datetime
from ..util import tqdm
from scanpy import logging
import os.path

HERE = Path(__file__).parent


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/wu2020.py"), "   ")
)
def wu2020() -> AnnData:
    """\
    Return the dataset from :cite:`Wu2020` as AnnData object.

    200k cells, of which 100k have TCRs.

    This is how the dataset was processed:

    .. code-block:: python

    {processing_code}
    """
    url = "https://github.com/scverse/scirpy/releases/download/d0.1.0/wu2020.h5ad"
    filename = settings.datasetdir / "wu2020.h5ad"
    adata = read(filename, backup_url=url)
    upgrade_schema(adata)
    return adata


@_doc_params(
    processing_code=indent(
        _read_to_str(HERE / "_processing_scripts/wu2020_3k.py"), "   "
    )
)
def wu2020_3k() -> AnnData:
    """\
    Return the dataset from :cite:`Wu2020` as AnnData object, downsampled
    to 3000 TCR-containing cells.

    This is how the dataset was processed:

    .. code-block:: python

    {processing_code}
    """
    url = "https://github.com/scverse/scirpy/releases/download/d0.1.0/wu2020_3k.h5ad"
    filename = settings.datasetdir / "wu2020_3k.h5ad"
    adata = read(filename, backup_url=url)
    upgrade_schema(adata)
    return adata


@_doc_params(
    processing_code=indent(
        _read_to_str(HERE / "_processing_scripts/maynard2020.py"), "   "
    )
)
def maynard2020() -> AnnData:
    """\
    Return the dataset from :cite:`Maynard2020` as AnnData object.

    21k cells from NSCLC profiled with Smart-seq2, of which 3,500 have :term:`TCRs<TCR>`
    and 1,500 have :term:`BCRs<BCR>`.

    The raw FASTQ files have been obtained from `PRJNA591860 <https://www.ebi.ac.uk/ena/browser/view/PRJNA591860>`__
    and processed using the nf-core `Smart-seq2 pipeline <https://github.com/nf-core/smartseq2/>`__.

    The processed files have been imported and transformed into an :class:`anndata.AnnData`
    object using the following script:

    .. code-block:: python

    {processing_code}
    """
    url = "https://github.com/scverse/scirpy/releases/download/d0.1.0/maynard2020.h5ad"
    filename = settings.datasetdir / "maynard2020.h5ad"
    adata = read(filename, backup_url=url)
    upgrade_schema(adata)
    return adata


def vdjdb(cached: bool = True, *, cache_path="data/vdjdb.h5ad") -> AnnData:
    """\
    Download VDJdb and process it into an AnnData object.

    `VDJdb <https://vdjdb.cdr3.net/>`_ :cite:`vdjdb` is a curated database of
    T-cell receptor (TCR) sequences with known antigen specificities.

    Parameters
    ----------
    cached
        If `True`, attempt to read from the `data` directory before downloading

    Returns
    -------
    An anndata object containing all entries from VDJDB in `obs`.
    Each entry is represented as if it was a cell, but without gene expression.
    Metadata is stored in `adata.uns["DB"]`.
    """
    if cached:
        try:
            return sc.read_h5ad(cache_path)
        except OSError:
            pass

    logging.info("Downloading latest version of VDJDB")
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest-version.txt"
    ) as url:
        latest_versions = url.read().decode().split()
    url = latest_versions[0]

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        urllib.request.urlretrieve(url, d / "vdjdb.tar.gz")
        with zipfile.ZipFile(d / "vdjdb.tar.gz") as zf:
            zf.extractall(d)
        df = pd.read_csv(d / "vdjdb_full.txt", sep="\t")

    tcr_cells = []
    for idx, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Processing VDJDB entries"
    ):
        cell = AirrCell(cell_id=idx)
        if not pd.isnull(row["cdr3.alpha"]):
            alpha_chain = AirrCell.empty_chain_dict()
            alpha_chain.update(
                {
                    "locus": "TRA",
                    "junction_aa": row["cdr3.alpha"],
                    "v_call": row["v.alpha"],
                    "j_call": row["j.alpha"],
                    "consensus_count": 0,
                    "productive": True,
                }
            )
            cell.add_chain(alpha_chain)

        if not pd.isnull(row["cdr3.beta"]):
            beta_chain = AirrCell.empty_chain_dict()
            beta_chain.update(
                {
                    "locus": "TRB",
                    "junction_aa": row["cdr3.beta"],
                    "v_call": row["v.beta"],
                    "d_call": row["d.beta"],
                    "j_call": row["j.beta"],
                    "consensus_count": 0,
                    "productive": True,
                }
            )
            cell.add_chain(beta_chain)

        INCLUDE_CELL_METADATA_FIELDS = [
            "species",
            "mhc.a",
            "mhc.b",
            "mhc.class",
            "antigen.epitope",
            "antigen.gene",
            "antigen.species",
            "reference.id",
            "method.identification",
            "method.frequency",
            "method.singlecell",
            "method.sequencing",
            "method.verification",
            "meta.study.id",
            "meta.cell.subset",
            "meta.subject.cohort",
            "meta.subject.id",
            "meta.replica.id",
            "meta.clone.id",
            "meta.epitope.id",
            "meta.tissue",
            "meta.donor.MHC",
            "meta.donor.MHC.method",
            "meta.structure.id",
            "vdjdb.score",
        ]
        for f in INCLUDE_CELL_METADATA_FIELDS:
            cell[f] = row[f]
        tcr_cells.append(cell)

    logging.info("Converting to AnnData object")
    adata = from_airr_cells(tcr_cells)

    adata.uns["DB"] = {"name": "VDJDB", "date_downloaded": datetime.now().isoformat()}

    # store cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    adata.write_h5ad(cache_path)

    return adata

def iedb(cached: bool = True, *, cache_path="data/iedb.h5ad") -> AnnData:
    """\
    Download IEDBD and process it into an AnnData object.
    ----------
    cached
        If `True`, attempt to read from the `data` directory before downloading
    Returns
    -------
    An anndata object containing all entries from IEDB in `obs`.
    Each entry is represented as if it was a cell, but without gene expression.
    Metadata is stored in `adata.uns["DB"]`.
    """
    if cached:
        try:
            return sc.read_h5ad(cache_path)
        except OSError:
            pass

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        urllib.request.urlretrieve("https://www.iedb.org/downloader.php?file_name=doc/receptor_full_v3.zip", d / "receptor_full_v3.zip")
        with zipfile.ZipFile(d / "receptor_full_v3.zip") as zf:
            zf.extractall(d)
        tcr_table = pd.read_csv(d / "receptor_full_v3.csv", index_col=None, sep=",", na_values=["None"], true_values=["True"],)

    tcr_table.loc[tcr_table["Chain 1 CDR3 Curated"].isna(),'Chain 1 CDR3 Curated'] = tcr_table["Chain 1 CDR3 Calculated"]

    tcr_table.loc[tcr_table["Chain 2 CDR3 Curated"].isna(),'Chain 2 CDR3 Curated'] = tcr_table["Chain 2 CDR3 Calculated"]

    tcr_table_T = tcr_table[(tcr_table["Response Type"] == "T cell")]

    tcr_table_T["Chain 1 CDR3 Curated"] = tcr_table_T["Chain 1 CDR3 Curated"].str.upper()
    tcr_table_T["Chain 2 CDR3 Curated"] = tcr_table_T["Chain 2 CDR3 Curated"].str.upper()

    tcr_table = tcr_table_T

    tcr_cells = []
    for idx, row in tcr_table.iterrows():
        cell = AirrCell(cell_id=row["Receptor ID"])
        alpha_chain = AirrCell.empty_chain_dict()
        beta_chain = AirrCell.empty_chain_dict()
        alpha_chain.update(
            {
                "locus": "TRA",
                "junction_aa": row["Chain 1 CDR3 Curated"],
                "junction": "None",
                "consensus_count": "None",
                "v_call": row["Curated Chain 1 V Gene"],
                "j_call": row["Curated Chain 1 J Gene"],
                "productive": "True",
            }
        )
        beta_chain.update(
            {
                "locus": "TRB",
                "junction_aa": row["Chain 2 CDR3 Curated"],
                "junction": "None",
                "consensus_count": "None",
                "v_call": row["Curated Chain 2 V Gene"],
                "d_call": row["Curated Chain 2 D Gene"],
                "j_call": row["Curated Chain 2 J Gene"],
                "productive": "True",
            }
        )
        cell.add_chain(alpha_chain)
        cell.add_chain(beta_chain)
        tcr_cells.append(cell)

    logging.info("Converting to AnnData object")
    IEDB = from_airr_cells(tcr_cells)
    tcr_table = tcr_table.set_index(IEDB.obs.index)
    IEDB.obs["Antigen"] = tcr_table["Antigen"]
    IEDB.obs["Organism"] = tcr_table["Organism"]
    IEDB.obs["Chain 1 Type"] = tcr_table["Chain 1 Type"]
    IEDB.obs["Chain 2 Type"] = tcr_table["Chain 2 Type"]
    IEDB.obs["Response Type"] = tcr_table["Response Type"]
    IEDB.obs["Reference IRI"] = tcr_table["Reference IRI"]
    IEDB.obs["Epitope IRI"] = tcr_table["Epitope IRI"]

    adata = IEDB

    adata.uns["DB"] = {"name": "IEDB", "date_downloaded": datetime.now().isoformat()}

    # store cache
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    adata.write_h5ad(cache_path)

    return adata
