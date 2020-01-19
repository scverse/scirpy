import pandas as pd
from scanpy import AnnData
from typing import Iterable, Collection
import numpy as np


def _check_anndata(adata):
    """Sanity checks on loaded data. Should be executed by 
    every read_xxx function"""
    assert (
        len(adata.X.shape) == 2
    ), "X needs to have dimensions, otherwise concat doesn't work. "
    assert "has_tcr" in adata.obs.columns


def read_10x(path: str):
    """Read TCR data from a 10x genomics sample.
    
    Parameters
    ----------
    path
        Path to filtered_contig_annotations.csv

    Returns
    -------
    AnnData
        AnnData object with TCR data in `obs` for each cell.  
    """
    contig_annotation = pd.read_csv(path, na_values="None")

    # for now, limit to the essential attributes
    contig_annotation = contig_annotation[
        [
            "barcode",
            "length",
            "chain",
            # "productive",
            # "v_gene",
            # "d_gene",
            # "j_gene",
            # "c_gene",
            "cdr3",
            # "cdr3_nt",
            "reads",
            "umis",
            # "full_length",
            # "is_cell",
            # "high_confidence",
        ]
    ]

    # for now, limit to alpha and beta chains
    contig_annotation_filtered = contig_annotation.loc[
        contig_annotation["chain"].isin(["TRA", "TRB"]), :
    ]

    # spread the data, s.t. there is one line per barcode
    def _apply(df):
        row_dict = dict()

        df_a = df.loc[df["chain"] == "TRA", :]
        df_b = df.loc[df["chain"] == "TRB", :]

        return pd.Series(
            {
                # TRA1
                "TRA_1_cdr3": df_a["cdr3"].values[0] if df_a.shape[0] >= 1 else np.nan,
                "TRA_1_reads": df_a["reads"].values[0]
                if df_a.shape[0] >= 1
                else np.nan,
                "TRA_1_umis": df_a["umis"].values[0] if df_a.shape[0] >= 1 else np.nan,
                # TRB1
                "TRB_1_cdr3": df_b["cdr3"].values[0] if df_b.shape[0] >= 1 else np.nan,
                "TRB_1_reads": df_b["reads"].values[0]
                if df_b.shape[0] >= 1
                else np.nan,
                "TRB_1_umis": df_b["umis"].values[0] if df_b.shape[0] >= 1 else np.nan,
                # TRA2
                "TRA_2_cdr3": df_a["cdr3"].values[1] if df_a.shape[0] >= 2 else np.nan,
                "TRA_2_reads": df_a["reads"].values[1]
                if df_a.shape[0] >= 2
                else np.nan,
                "TRA_2_umis": df_a["umis"].values[1] if df_a.shape[0] >= 2 else np.nan,
                # TRB2
                "TRB_2_cdr3": df_b["cdr3"].values[1] if df_b.shape[0] >= 2 else np.nan,
                "TRB_2_reads": df_b["reads"].values[1]
                if df_b.shape[0] >= 2
                else np.nan,
                "TRB_2_umis": df_b["umis"].values[1] if df_b.shape[0] >= 2 else np.nan,
            }
        )

    tcr_df = (
        # sort first, and not in _apply for more performance.
        # assumes groupby is stable, which it should be
        # https://stackoverflow.com/questions/39373820/is-pandas-dataframe-groupby-guaranteed-to-be-stable
        contig_annotation_filtered.sort_values(
            ["barcode", "chain", "umis", "reads"], ascending=[True, True, False, False]
        )
        .groupby(["barcode"])
        .apply(_apply)
        .assign(has_tcr=True)
    )

    adata = AnnData(obs=tcr_df, X=np.empty([tcr_df.shape[0], 0]))

    _check_anndata(adata)

    return adata


def read_tracer():
    pass
