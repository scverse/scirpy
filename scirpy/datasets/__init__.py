from anndata import AnnData
from pathlib import Path
from ..util import _doc_params, _read_to_str
from scanpy.readwrite import read
from scanpy import settings
from contextlib import contextmanager
import tqdm.auto
from tqdm import tqdm as tqdm_
from textwrap import indent

HERE = Path(__file__).parent


@contextmanager
def _monkey_patch_tqdm():
    """Monkey-patch tqdm.auto to allow download without ipywidgets installed. 
    
    See also https://github.com/theislab/scanpy/pull/1130.
    """
    tqdm_auto = tqdm.auto.tqdm
    tqdm.auto.tqdm = tqdm_
    try:
        yield
    finally:
        tqdm.auto.tqdm = tqdm_auto


@_doc_params(
    processing_code=indent(_read_to_str(HERE / "_processing_scripts/wu2020.py"), "   ")
)
def wu2020() -> AnnData:
    """\
    Return the dataset from [Wu2020]_ as AnnData object. 

    200k cells, of which 100k have TCRs.

    This is how the dataset was processed:

    .. code-block:: python 

    {processing_code}
    """
    # os.makedirs(settings.datasetdir, exist_ok=True)
    # TODO host it on github or similar
    url = "https://github.com/icbi-lab/scirpy/releases/download/v0.1/wu2020.h5ad"
    filename = settings.datasetdir / "wu2020.h5ad"
    with _monkey_patch_tqdm():
        adata = read(filename, backup_url=url)
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
    # os.makedirs(settings.datasetdir, exist_ok=True)
    # TODO host it on github or similar
    url = "https://github.com/icbi-lab/scirpy/releases/download/v0.1/wu2020_3k.h5ad"
    filename = settings.datasetdir / "wu2020_3k.h5ad"
    with _monkey_patch_tqdm():
        adata = read(filename, backup_url=url)
    return adata
