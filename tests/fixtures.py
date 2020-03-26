import pytest
import pandas as pd
from anndata import AnnData
import numpy as np
from sctcrpy._util import _is_symmetric


@pytest.fixture
def adata_conn():
    """Adata with connectivities computed"""
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(cell_id=["cell1", "cell2", "cell3", "cell4"])
        .set_index("cell_id")
    )
    adata.uns["tcr_neighbors"] = {
        "connectivities": np.array(
            [[1, 0, 0.5, 0], [0, 1, 1, 0], [0.5, 1, 1, 0], [0, 0, 0, 1]]
        )
    }
    assert _is_symmetric(adata.uns["tcr_neighbors"]["connectivities"])
    return adata


@pytest.fixture
def adata_clonotype_network():
    """Adata with clonotype network computed"""
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(cell_id=["cell1", "cell2", "cell3", "cell4"])
        .set_index("cell_id")
    )
    adata.uns["tcr_neighbors"] = {
        "connectivities": np.array(
            [[1, 0, 0.5, 0], [0, 1, 1, 0], [0.5, 1, 1, 0], [0, 0, 0, 1]]
        )
    }
    adata.obsm["X_clonotype_network"] = np.array(
        [
            [2.41359095, 0.23412465],
            [np.nan, np.nan],
            [1.61680611, 0.80266963],
            [3.06104282, 2.14395562],
        ]
    )
    assert _is_symmetric(adata.uns["tcr_neighbors"]["connectivities"])
    return adata


@pytest.fixture
def adata_tra():
    obs = {
        "AAGGTTCCACCCAGTG-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": 3,
            "clonotype": "clonotype_458",
            "chain_pairing": "Extra alpha",
        },
        "ACTATCTAGGGCTTCC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAVDGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCCGTGGACGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": 1,
            "clonotype": "clonotype_739",
            "chain_pairing": "Extra alpha",
        },
        "CAGTAACAGGCATGTG-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CAVRDSNYQLIW",
            "TRA_1_cdr3_nt": "TGTGCTGTGAGAGATAGCAACTATCAGTTAATCTGG",
            "sample": 1,
            "clonotype": "clonotype_986",
            "chain_pairing": "Two full chains",
        },
        "CCTTACGGTCATCCCT-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CAVRDSNYQLIW",
            "TRA_1_cdr3_nt": "TGTGCTGTGAGGGATAGCAACTATCAGTTAATCTGG",
            "sample": 1,
            "clonotype": "clonotype_987",
            "chain_pairing": "Single pair",
        },
        "CGTCCATTCATAACCG-1": {
            "TRA_1_cdr3_len": 17.0,
            "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCAGCAAGTCGCAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": 5,
            "clonotype": "clonotype_158",
            "chain_pairing": "Single pair",
        },
        "CTTAGGAAGGGCATGT-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": 1,
            "clonotype": "clonotype_459",
            "chain_pairing": "Single pair",
        },
        "GCAAACTGTTGATTGC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAVDGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCCGTGGATGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": 1,
            "clonotype": "clonotype_738",
            "chain_pairing": "Single pair",
        },
        "GCTCCTACAAATTGCC-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGATCCCAACACCAATGCAGGCAAATCAACCTTT",
            "sample": 3,
            "clonotype": "clonotype_460",
            "chain_pairing": "Two full chains",
        },
        "GGAATAATCCGATATG-1": {
            "TRA_1_cdr3_len": 17.0,
            "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCAGCAAGTAGGAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": 5,
            "clonotype": "clonotype_157",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGATAGCAT-1": {
            "TRA_1_cdr3_len": 13.0,
            "TRA_1_cdr3": "CAGGGSGTYKYIF",
            "TRA_1_cdr3_nt": "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
            "sample": 3,
            "clonotype": "clonotype_330",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGTACGCCC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAMRVGGSQGNLIF",
            "TRA_1_cdr3_nt": "TGTGCAATGAGGGTCGGAGGAAGCCAAGGAAATCTCATCTTT",
            "sample": 5,
            "clonotype": "clonotype_592",
            "chain_pairing": "Two full chains",
        },
        "AAACCTGCATAGAAAC-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CAFMKPFTAGNQFYF",
            "TRA_1_cdr3_nt": "TGTGCTTTCATGAAGCCTTTTACCGCCGGTAACCAGTTCTATTTT",
            "sample": 5,
            "clonotype": "clonotype_284",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTCCGTTAA-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CALNTGGFKTIF",
            "TRA_1_cdr3_nt": "TGTGCTCTCAATACTGGAGGCTTCAAAACTATCTTT",
            "sample": 3,
            "clonotype": "clonotype_425",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTTTGTGTG-1": {
            "TRA_1_cdr3_len": 13.0,
            "TRA_1_cdr3": "CALRGGRDDKIIF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGAGGGGGTAGAGATGACAAGATCATCTTT",
            "sample": 3,
            "clonotype": "clonotype_430",
            "chain_pairing": "Single pair",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_clonotype():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
            ["cell8", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_diversity():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype_"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata
