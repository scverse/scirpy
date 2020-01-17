import pandas as pd
from scanpy import AnnData
import json


def _check_tcr_df(df):
    """Ensure that the data read with whatever method follows the 
    sctcrpy column definition. 
    """
    # TODO
    assert True


def read_10x(path: str):
    """Read TCR data from a 10x genomics sample.
    
    Parameters
    ----------
    path
        Path to filtered_contig_annotations.csv

    Returns
    -------
    pd.DataFrame
    """
    contig_annotation = pd.read_csv(path, na_values="None")
    contig_annotation = contig_annotation[
        [
            "barcode",
            "length",
            "chain",
            "productive",
            "v_gene",
            "d_gene",
            "j_gene",
            "c_gene",
            "cdr3",
            "cdr3_nt",
            "reads",
            "umis",
            "full_length",
            "is_cell",
            "high_confidence",
        ]
    ]
    _check_tcr_df(contig_annotation)
    return contig_annotation


def read_tracer():
    pass
