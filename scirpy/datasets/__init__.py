from anndata import AnnData
from pathlib import Path
from ..util import _doc_params, _read_to_str
from scanpy.readwrite import read
from scanpy import settings
from textwrap import indent
from ..io import upgrade_schema

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
    url = "https://github.com/icbi-lab/scirpy/releases/download/d0.1.0/wu2020.h5ad"
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
    url = "https://github.com/icbi-lab/scirpy/releases/download/d0.1.0/wu2020_3k.h5ad"
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
    url = "https://github.com/icbi-lab/scirpy/releases/download/d0.1.0/maynard2020.h5ad"
    filename = settings.datasetdir / "maynard2020.h5ad"
    adata = read(filename, backup_url=url)
    upgrade_schema(adata)
    return adata
