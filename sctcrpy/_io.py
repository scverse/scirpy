import pandas as pd
from scanpy import AnnData


def read_10x(path: str, into: AnnData = None):
    """Read TCR data from a 10x genomics sample.
    
    Parameters
    ----------
    path
        Path to 
    into
        anndata object to inject the TCR data into. Will be stored in `obs`. 
        If None is given, a new AnnData object is created. 

    Raises:
    -------
    Warning: if cell barcodes do not exist in AnnData object
    """
    contig_annotation = pd.read_csv(path)
    print(contig_annotation)
    pass


def read_tracer():
    pass
