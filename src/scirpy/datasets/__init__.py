import gzip
import json
import os
import os.path
import tempfile
import urllib.request
import warnings
import zipfile
from datetime import datetime
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from textwrap import dedent, indent
from typing import cast

import mudata
import pandas as pd
import pooch
import requests
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scanpy import logging

from scirpy.io._convert_anndata import from_airr_cells
from scirpy.io._datastructures import AirrCell
from scirpy.io._io import _infer_locus_from_gene_names, _IOLogger
from scirpy.pp import index_chains
from scirpy.util import _doc_params, _read_to_str, tqdm

HERE = Path(__file__).parent
DATASET_ENV_VAR = "SCIRPY_DATA_DIR"

_AWS_EXAMPLEDATA = pooch.create(
    path=pooch.os_cache("scirpy"),
    base_url="https://exampledata.scverse.org/scirpy",
    version=version("scirpy"),
    version_dev="main",
    env=DATASET_ENV_VAR,
    registry={
        "wu2020.h5mu": "md5:ed30d9c1c44cae544f4c080a2451118b",
        "wu2020_3k.h5mu": "md5:12c57c790f8a403751304c9de5a18cbf",
        "maynard2020.h5mu": "md5:da64ac62e3e92c80eaf0e8eef6537ac7",
        "stephenson2021_5k.h5mu": "md5:6ea26f9d95525371ff9028f8e99ed474",
    },
)

_POOCH_INFO = dedent(
    f"""\
    .. note::
        Scirpy datasets are managed through `Pooch <https://github.com/fatiando/pooch>`_.

        By default, the dataset will be downloaded into your operating system's default
        cache directory (See :func:`pooch.os_cache` for more details). If it has already been
        downloaded, it will be retrieved from the cache.

        You can override the default cache dir by setting the `{DATASET_ENV_VAR}` environment variable
        to a path of your preference.
    """
)


def _get_iggytop_registry(tag: str = "latest") -> pooch.Pooch:
    """
    Create pooch registry based on iggytop metadata.json file obtained from GitHub.

    If 'latest' tag is specified, always fetch the latest metadata.json file.
    If an explicit tag is specified, rely on pooch to deal with caching.
    """
    iggytop_cache_dir = Path(os.environ.get(DATASET_ENV_VAR) or pooch.os_cache("scirpy")) / "iggytop"
    if tag == "latest":
        # in case of latest, always fetch the latest metadata file from GitHub
        response = requests.get("https://github.com/biocypher/iggytop/releases/latest/download/metadata.json")
        response.raise_for_status()
        metadata = response.json()
        logging.info(
            f"Requested 'latest' version of iggytop. We recommend setting `tag=\"{metadata['iggytop_version']}\"` for reproducibility."
        )
    else:
        # in case tag is specified, use pooch to download and cache the file
        metadata_json = pooch.retrieve(
            f"https://github.com/biocypher/iggytop/releases/download/{tag}/metadata.json",
            known_hash=None,
            path=iggytop_cache_dir,
        )
        with Path(metadata_json).open() as f:
            metadata = json.load(f)

    return pooch.create(
        path=iggytop_cache_dir,
        base_url=f"https://github.com/biocypher/iggytop/releases/download/{metadata['iggytop_version']}",
        version=metadata["iggytop_version"].replace("data-", ""),
        registry={k: f"sha256:{v}" for k, v in metadata["assets"].items()},
    )


@_doc_params(pooch_info=_POOCH_INFO)
def iggytop(*, deduplicated: bool = True, tag: str = "latest") -> AnnData:
    """\
    Return the `IggyTop <https://iggytop.readthedocs.io/en/latest/>`_ database as an AnnData object.

    IggyTop (**I**mmunological **G**raph **Y**ielding **T**op receptor-epitope pairings)
    is a harmonized database of immunoreceptor-epitope pairings integrating data from
    multiple sources: IEDB, VDJdb, McPAS-TCR, CEDAR, ITRAP, TRAIT, TCR3d, and NeoTCR.
    V(D)J genes are normalized to IMGT standards and CDR3 sequences are harmonized following
    `AIRR standards <https://docs.airr-community.org/en/stable/datarep/rearrangements.html>`_.
    Pre-built datasets are released bimonthly.

    By default, a deduplicated version of the dataset is returned. Use this version if you'd like
    to work with the integrated resource combining data from all source datasets. If you prefer to work
    with a single resource, set `deduplicated=False` and filter the resource of interest via
    `.obs["source"]`.

    {pooch_info}

    Parameters
    ----------
    deduplicated
        If `True`, return the deduplicated and 10X-filtered dataset. If `False`, return
        the full merged dataset including all source records.
    tag
        The IggyTop release tag to use. Defaults to ``"latest"``, which always fetches
        the most recent release. For reproducibility, pin a specific release tag
        (e.g. ``"data-2026.04.25.075304"``).

    Returns
    -------
    An AnnData object containing immunoreceptor-epitope pairings from IggyTop
    in `obsm["airr"]`. Each entry is represented as if it was a cell, but without
    gene expression data.
    """
    iggytop_registry = _get_iggytop_registry(tag)
    if deduplicated:
        return sc.read_h5ad(iggytop_registry.fetch("deduplicated_anndata.h5ad"))
    else:
        return sc.read_h5ad(iggytop_registry.fetch("merged_anndata.h5ad"))


@_doc_params(pooch_info=_POOCH_INFO)
def vdjdb(cached: bool | None = None, *, cache_path: None = None, tag: str = "latest") -> AnnData:
    """\
    Download VDJdb through IggyTop <https://iggytop.readthedocs.io/en/latest/>`_.

    `VDJdb <https://vdjdb.cdr3.net/>`_ :cite:`vdjdb` is a curated database of
    T-cell receptor (TCR) sequences with known antigen specificities.

    As of v0.24, this is a wrapper around :func:`~scirpy.datasets.iggytop`.

    {pooch_info}

    Parameters
    ----------
    cached
        Deprecated as of v0.24. Has no effect. Caching is handled through `pooch` now.
    cache_path
        Deprecated as of v0.24. Has no effect.
    tag
        The IggyTop release tag to use. Defaults to ``"latest"``, which always fetches
        the most recent release. For reproducibility, pin a specific release tag
        (e.g. ``"data-2026.04.25.075304"``).

    Returns
    -------
    An AnnData object containing all entries from VDJDB in `obsm["airr"]`.
    Each entry is represented as if it was a cell, but without gene expression.
    Metadata is stored in `adata.uns["DB"]`.
    """
    if cached is not None or cache_path is not None:
        warnings.warn(
            "The arguments `cached` and `cache_path` are deprecated since v0.24 and have no effect."
            "Caching is now handled through Pooch.",
            category=FutureWarning,
        )
    adata = iggytop(deduplicated=False, tag=tag)
    adata = adata[adata.obs["source"] == "VDJDB"].copy()
    adata.uns["DB"]["name"] = "VDJDB"

    return adata


@_doc_params(pooch_info=_POOCH_INFO)
def iedb(cached: bool | None = None, *, cache_path=None, tag: str = "latest") -> AnnData:
    """\
    Download IEDB through IggyTop <https://iggytop.readthedocs.io/en/latest/>`_.

    :cite:`iedb` is a curated database of
    T-cell receptor (TCR) sequences with known antigen specificities.

    As of v0.24, this is a wrapper around :func:`~scirpy.datasets.iggytop`.

    {pooch_info}

    Parameters
    ----------
    cached
        Deprecated as of v0.24. Has no effect.
    cache_path
        Deprecated as of v0.24. Has no effect.
    tag
        The IggyTop release tag to use. Defaults to ``"latest"``, which always fetches
        the most recent release. For reproducibility, pin a specific release tag
        (e.g. ``"data-2026.04.25.075304"``).

    Returns
    -------
    An AnnData object containing all entries from IEDB in `obsm["airr"]`.
    Each entry is represented as if it was a cell, but without gene expression.
    Metadata is stored in `adata.uns["DB"]`.
    """
    if cached is not None or cache_path is not None:
        warnings.warn(
            "The arguments `cached` and `cache_path` are deprecated since v0.24 and have no effect."
            "Caching is now handled through Pooch.",
            category=FutureWarning,
        )
    adata = iggytop(deduplicated=False, tag=tag)
    adata = adata[adata.obs["source"] == "IEDB"].copy()
    adata.uns["DB"]["name"] = "IEDB"

    return adata


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/wu2020.py"), " " * 8),
    pooch_info=_POOCH_INFO,
)
def wu2020() -> MuData:
    """\
    Return the dataset from :cite:`Wu2020` as MuData object.

    140k cells, of which 100k have TCRs.

    {pooch_info}

    This is how the dataset was processed:

    .. code-block:: python

        {processing_code}
    """
    fname = cast(PathLike, _AWS_EXAMPLEDATA.fetch("wu2020.h5mu", progressbar=True))
    return mudata.read_h5mu(fname)


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/wu2020_3k.py"), " " * 8),
    pooch_info=_POOCH_INFO,
)
def wu2020_3k() -> MuData:
    """\
    Return the dataset from :cite:`Wu2020` as AnnData object, downsampled
    to 3000 TCR-containing cells.

    {pooch_info}

    This is how the dataset was processed:

    .. code-block:: python

        {processing_code}
    """
    fname = cast(PathLike, _AWS_EXAMPLEDATA.fetch("wu2020_3k.h5mu", progressbar=True))
    return mudata.read_h5mu(fname)


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/maynard2020.py"), " " * 8),
    pooch_info=_POOCH_INFO,
)
def maynard2020() -> MuData:
    """\
    Return the dataset from :cite:`Maynard2020` as AnnData object.

    21k cells from NSCLC profiled with Smart-seq2, of which 3,500 have :term:`TCRs<TCR>`
    and 1,500 have :term:`BCRs<BCR>`.

    {pooch_info}

    The raw FASTQ files have been obtained from `PRJNA591860 <https://www.ebi.ac.uk/ena/browser/view/PRJNA591860>`__
    and processed using the nf-core `RNA-seq pipeline <https://github.com/nf-core/rnaseq>`_ to obtain
    gene expression and TraCeR/BraCeR to reconstruct receptors.

    The processed files have been imported and transformed into an :class:`anndata.AnnData`
    object using the following script:

    .. code-block:: python

        {processing_code}
    """
    fname = cast(PathLike, _AWS_EXAMPLEDATA.fetch("maynard2020.h5mu", progressbar=True))
    return mudata.read_h5mu(fname)


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/maynard2020.py"), " " * 8),
    pooch_info=_POOCH_INFO,
)
def stephenson2021_5k() -> MuData:
    """\
    Return the dataset from :cite:`Stephenson2021` as MuData object, downsampled
    to 5000 BCR-containing cells.

    The original study sequenced 1,141,860 cells from 143 PBMC samples collected from patients with different severity of COVID-19 and control groups.
    Gene expression, TCR-enriched and BCR-enriched libraries were prepared for each sample according to 10x Genomics protocol and NovaSeq 6000 was used for sequencing.

    A preprocessed dataset for the transciptome library was obtained from `Array Express <https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026>`__
    A preprocessed dataset for the BCR-enriched library was obtained from `clatworthylab's GitHub <https://github.com/clatworthylab/COVID_analysis>`__
    Both dataset have already passed quality control and all cells that didn't express BCR were discarded.

    To  speed up computation time, we solely included 5 samples from each of the COVID-19-positive groups and randomly subsampled down to a total of 5k cells.

    """
    fname = cast(PathLike, _AWS_EXAMPLEDATA.fetch("stephenson2021_5k.h5mu", progressbar=True))
    return mudata.read_h5mu(fname)
