from anndata import AnnData
from pathlib import Path
from .._util import _doc_params, _read_to_str
from scanpy.readwrite import read
from scanpy import settings
import os
from contextlib import contextmanager
import tqdm
from tqdm import tqdm as tqdm_

HERE = Path(__file__).parent


@contextmanager
def _monkey_path_tqdm():
    """Monkey-patch tqdm.auto to allow download without ipywidgets installed. 
    
    See also https://github.com/theislab/scanpy/pull/1130.
    """
    tqdm_auto = tqdm.auto.tqdm
    tqdm.auto.tqdm = tqdm_
    try:
        yield
    finally:
        tqdm.auto.tqdm = tqdm_auto


@_doc_params(processing_code=_read_to_str(HERE / "_processing_scripts/wu2020.py"))
def wu2020() -> AnnData:
    """
    Return the dataset from [Wu2020]_ as AnnData object. 

    200k cells, of which 100k have TCRs.

    This is how the dataset was processed:
    
    ```python
    {processing_code}
    ```
    """
    # os.makedirs(settings.datasetdir, exist_ok=True)
    # TODO host it on github or similar
    url = "https://www.dropbox.com/s/m48f3uveb2lrrbq/wu2020.h5ad?dl=1"
    filename = settings.datasetdir / "wu2020.h5ad"
    with _monkey_path_tqdm():
        adata = read(filename, backup_url=url)
    return adata


@_doc_params(processing_code=_read_to_str(HERE / "_processing_scripts/wu2020_3k.py"))
def wu2020_3k() -> AnnData:
    """
    Return the dataset from :cite:`Wu2020` as AnnData object, downsampled 
    to 3000 TCR-containing cells. 

    This is how the dataset was processed:

    ```python
    {processing_code}
    ```
    """
    # os.makedirs(settings.datasetdir, exist_ok=True)
    # TODO host it on github or similar
    url = "https://www.dropbox.com/s/dcfadomspl2pl8n/wu2020_3k.h5ad?dl=1"
    filename = settings.datasetdir / "wu2020_3k.h5ad"
    with _monkey_path_tqdm():
        adata = read(filename, backup_url=url)
    return adata
