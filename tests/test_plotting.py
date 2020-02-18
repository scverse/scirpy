import pandas as pd
from anndata import AnnData
import pytest
from sctcrpy import pl


@pytest.fixture
def adata_clonotype():
    obs = pd.DataFrame.from_records(
        [
            [
                "cell1",
                "A",
                "ct1",
                13,
                "SANLGQLNTEAF",
                "TGCAGTGCTGATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell2",
                "A",
                "ct1",
                13,
                "SANLGQLNTEAF",
                "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
            ],
            [
                "cell3",
                "A",
                "ct1",
                13,
                "SANLGLNTEAF",
                "TGCAGTGCTGCCATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell3",
                "A",
                "NaN",
                14,
                "SANLGQLNTEAF",
                "TGCAGTGCTGATACCCTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell4",
                "B",
                "ct1",
                14,
                "SANLQLNTEAF",
                "TGCACCGTGCTGATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell5",
                "B",
                "ct2",
                15,
                "SANLGQLNTEAF",
                "TGCAGTGCTGATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell6",
                "B",
                "ct3",
                15,
                "SANLGQLNTEAF",
                "TGCAGTGCTGATACTCCAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell7",
                "B",
                "ct4",
                16,
                "SANLGQLTEAF",
                "TGCAGTGCTGATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
            [
                "cell8",
                "B",
                "ct4",
                17,
                "SANLGQLNTEAF",
                "TGCAGTGCCCTGATACTAGAGGCAAAAACATTCAGTACTTC",
            ],
        ],
        columns=[
            "cell_id",
            "group",
            "clonotype",
            "TRB_1_cdr3_len",
            "TRB_1_cdr3",
            "TRB_1_cdr3_nt",
        ],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


def test_alpha_diversity(adata_clonotype):
    pl.alpha_diversity(adata_clonotype, groupby="group")


def test_clonal_expansion(adata_clonotype):
    pl.clonal_expansion(adata_clonotype, groupby="group")


def test_cdr_convergence(adata_clonotype):
    pl.group_abundance(adata_clonotype, groupby="group")


def test_spectratype(adata_clonotype):
    pl.group_abundance(adata_clonotype, groupby="group")


def test_group_abundance(adata_clonotype):
    pl.group_abundance(adata_clonotype, groupby="group")
