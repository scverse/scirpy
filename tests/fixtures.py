import pytest
import pandas as pd
from anndata import AnnData
import numpy as np
from scirpy.util import _is_symmetric


@pytest.fixture
def adata_cdr3():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "AHA", "KKY", "KKK", "GCGGCGGCG"],
            ["cell2", "AHA", "nan", "KK", "KKK", "GCGAUGGCG"],
            ["cell3", "nan", "nan", "nan", "nan", "nan"],
            ["cell4", "AAA", "AAA", "LLL", "AAA", "GCUGCUGCU"],
            ["cell5", "nan", "AAA", "LLL", "nan", "nan"],
        ],
        columns=[
            "cell_id",
            "TRA_1_cdr3",
            "TRA_2_cdr3",
            "TRB_1_cdr3",
            "TRB_2_cdr3",
            "TRA_1_cdr3_nt",
        ],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_define_clonotypes():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "ATA", "GGC", "CCC"],
            ["cell2", "AAA", "ATA", "GGC", "CCC"],
            ["cell3", "GGG", "ATA", "GGC", "CCC"],
            ["cell4", "GGG", "ATA", "GGG", "CCC"],
            ["cell10", "nan", "nan", "nan", "nan"],
        ],
        columns=[
            "cell_id",
            "TRA_1_cdr3_nt",
            "TRA_2_cdr3_nt",
            "TRB_1_cdr3_nt",
            "TRB_2_cdr3_nt",
        ],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_define_clonotype_clusters():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "AHA", "KKY", "KKK"],
            ["cell2", "AAA", "AHA", "KKY", "KKK"],
            ["cell3", "BBB", "AHA", "KKY", "KKK"],
            ["cell4", "BBB", "AHA", "BBB", "KKK"],
            ["cell5", "AAA", "nan", "KKY", "KKK"],
            ["cell6", "AAA", "nan", "KKY", "CCC"],
            ["cell7", "AAA", "AHA", "ZZZ", "nan"],
            ["cell8", "AAA", "nan", "nan", "KKK"],
            ["cell9", "nan", "nan", "nan", "KKK"],
            ["cell10", "nan", "nan", "nan", "nan"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3", "TRB_1_cdr3", "TRB_2_cdr3",],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_conn():
    """Adata with connectivities computed"""
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(
            cell_id=["cell1", "cell2", "cell3", "cell4"],
            TRA_1_v_gene=["av1", "av1", "av2", "av1"],
            TRB_1_v_gene=["bv1", "bv1", "bv2", "bv1"],
            TRA_2_v_gene=["a2v1", "a2v2", "a2v2", "a2v1"],
            TRB_2_v_gene=["b2v1", "b2v2", "b2v2", "b2v1"],
        )
        .set_index("cell_id")
    )
    adata.uns["tcr_neighbors_aa_alignment"] = {
        "connectivities": np.array(
            [[1, 0, 0.5, 0], [0, 1, 1, 0], [0.5, 1, 1, 0], [0, 0, 0, 1]]
        )
    }
    assert _is_symmetric(adata.uns["tcr_neighbors_aa_alignment"]["connectivities"])
    return adata


@pytest.fixture
def adata_clonotype_network():
    """Adata with clonotype network computed"""
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(cell_id=["cell1", "cell2", "cell3", "cell4"])
        .set_index("cell_id")
    )
    adata.uns["foo_neighbors"] = {
        "connectivities": np.array(
            [[1, 0, 0.5, 0], [0, 1, 1, 0], [0.5, 1, 1, 0], [0, 0, 0, 1]]
        )
    }
    adata.uns["clonotype_network"] = {"neighbors_key": "foo_neighbors"}
    adata.obsm["X_clonotype_network"] = np.array(
        [
            [2.41359095, 0.23412465],
            [np.nan, np.nan],
            [1.61680611, 0.80266963],
            [3.06104282, 2.14395562],
        ]
    )
    assert _is_symmetric(adata.uns["foo_neighbors"]["connectivities"])
    return adata


@pytest.fixture
def adata_tra():
    obs = {
        "AAGGTTCCACCCAGTG-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clonotype": "clonotype_458",
            "chain_pairing": "Extra alpha",
        },
        "ACTATCTAGGGCTTCC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAVDGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCCGTGGACGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clonotype": "clonotype_739",
            "chain_pairing": "Extra alpha",
        },
        "CAGTAACAGGCATGTG-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CAVRDSNYQLIW",
            "TRA_1_cdr3_nt": "TGTGCTGTGAGAGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clonotype": "clonotype_986",
            "chain_pairing": "Two full chains",
        },
        "CCTTACGGTCATCCCT-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CAVRDSNYQLIW",
            "TRA_1_cdr3_nt": "TGTGCTGTGAGGGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clonotype": "clonotype_987",
            "chain_pairing": "Single pair",
        },
        "CGTCCATTCATAACCG-1": {
            "TRA_1_cdr3_len": 17.0,
            "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCAGCAAGTCGCAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clonotype": "clonotype_158",
            "chain_pairing": "Single pair",
        },
        "CTTAGGAAGGGCATGT-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "1",
            "clonotype": "clonotype_459",
            "chain_pairing": "Single pair",
        },
        "GCAAACTGTTGATTGC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAVDGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCCGTGGATGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clonotype": "clonotype_738",
            "chain_pairing": "Single pair",
        },
        "GCTCCTACAAATTGCC-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CALSDPNTNAGKSTF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGTGATCCCAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clonotype": "clonotype_460",
            "chain_pairing": "Two full chains",
        },
        "GGAATAATCCGATATG-1": {
            "TRA_1_cdr3_len": 17.0,
            "TRA_1_cdr3": "CAASRNAGGTSYGKLTF",
            "TRA_1_cdr3_nt": "TGTGCAGCAAGTAGGAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clonotype": "clonotype_157",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGATAGCAT-1": {
            "TRA_1_cdr3_len": 13.0,
            "TRA_1_cdr3": "CAGGGSGTYKYIF",
            "TRA_1_cdr3_nt": "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
            "sample": "3",
            "clonotype": "clonotype_330",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGTACGCCC-1": {
            "TRA_1_cdr3_len": 14.0,
            "TRA_1_cdr3": "CAMRVGGSQGNLIF",
            "TRA_1_cdr3_nt": "TGTGCAATGAGGGTCGGAGGAAGCCAAGGAAATCTCATCTTT",
            "sample": "5",
            "clonotype": "clonotype_592",
            "chain_pairing": "Two full chains",
        },
        "AAACCTGCATAGAAAC-1": {
            "TRA_1_cdr3_len": 15.0,
            "TRA_1_cdr3": "CAFMKPFTAGNQFYF",
            "TRA_1_cdr3_nt": "TGTGCTTTCATGAAGCCTTTTACCGCCGGTAACCAGTTCTATTTT",
            "sample": "5",
            "clonotype": "clonotype_284",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTCCGTTAA-1": {
            "TRA_1_cdr3_len": 12.0,
            "TRA_1_cdr3": "CALNTGGFKTIF",
            "TRA_1_cdr3_nt": "TGTGCTCTCAATACTGGAGGCTTCAAAACTATCTTT",
            "sample": "3",
            "clonotype": "clonotype_425",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTTTGTGTG-1": {
            "TRA_1_cdr3_len": 13.0,
            "TRA_1_cdr3": "CALRGGRDDKIIF",
            "TRA_1_cdr3_nt": "TGTGCTCTGAGAGGGGGTAGAGATGACAAGATCATCTTT",
            "sample": "3",
            "clonotype": "clonotype_430",
            "chain_pairing": "Single pair",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_vdj():
    obs = {
        "LT1_ACGGCCATCCGAGCCA-2-24": {
            "TRA_1_j_gene": "TRAJ42",
            "TRA_1_v_gene": "TRAV26-2",
            "TRB_1_v_gene": "TRBV7-2",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ2-5",
            "sample": "LT1",
        },
        "LT1_CGCTTCACAAGGTGTG-2-24": {
            "TRA_1_j_gene": "TRAJ45",
            "TRA_1_v_gene": "None",
            "TRB_1_v_gene": "None",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ2-3",
            "sample": "LT1",
        },
        "LT1_AGGGAGTTCCCAAGAT-2-24": {
            "TRA_1_j_gene": "TRAJ29",
            "TRA_1_v_gene": "TRAV12-1",
            "TRB_1_v_gene": "TRBV20-1",
            "TRB_1_d_gene": "TRBD2",
            "TRB_1_j_gene": "TRBJ1-1",
            "sample": "LT1",
        },
        "LT1_ATTACTCGTTGGACCC-2-24": {
            "TRA_1_j_gene": "TRAJ4",
            "TRA_1_v_gene": "TRAV12-1",
            "TRB_1_v_gene": "TRBV7-2",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ2-6",
            "sample": "LT1",
        },
        "LT1_GCAATCACAATGAATG-1-24": {
            "TRA_1_j_gene": "TRAJ52",
            "TRA_1_v_gene": "TRAV8-6",
            "TRB_1_v_gene": "TRBV30",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ2-2",
            "sample": "LT1",
        },
        "LT1_TCTCTAATCCACTGGG-2-24": {
            "TRA_1_j_gene": "TRAJ43",
            "TRA_1_v_gene": "TRAV8-3",
            "TRB_1_v_gene": "TRBV30",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ1-2",
            "sample": "LT1",
        },
        "LT1_TATTACCTCAACGGCC-2-24": {
            "TRA_1_j_gene": "TRAJ45",
            "TRA_1_v_gene": "TRAV20",
            "TRB_1_v_gene": "TRBV4-1",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ1-3",
            "sample": "LT1",
        },
        "LT1_CGTCAGGTCGAACTGT-1-24": {
            "TRA_1_j_gene": "TRAJ15",
            "TRA_1_v_gene": "TRAV17",
            "TRB_1_v_gene": "TRBV5-1",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ1-1",
            "sample": "LT1",
        },
        "LT1_GGGAATGGTTGCGTTA-2-24": {
            "TRA_1_j_gene": "None",
            "TRA_1_v_gene": "None",
            "TRB_1_v_gene": "TRBV30",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ2-2",
            "sample": "LT1",
        },
        "LT1_AGCTCCTGTAATCGTC-2-24": {
            "TRA_1_j_gene": "TRAJ13",
            "TRA_1_v_gene": "TRAV13-1",
            "TRB_1_v_gene": "TRBV18",
            "TRB_1_d_gene": "TRBD2",
            "TRB_1_j_gene": "TRBJ2-2",
            "sample": "LT1",
        },
        "LT1_CAGCTGGTCCGCGGTA-1-24": {
            "TRA_1_j_gene": "TRAJ30",
            "TRA_1_v_gene": "TRAV21",
            "TRB_1_v_gene": "TRBV30",
            "TRB_1_d_gene": "TRBD2",
            "TRB_1_j_gene": "TRBJ2-1",
            "sample": "LT1",
        },
        "LT1_CCTTTCTCAGCAGTTT-1-24": {
            "TRA_1_j_gene": "TRAJ23",
            "TRA_1_v_gene": "TRAV9-2",
            "TRB_1_v_gene": "TRBV3-1",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ1-2",
            "sample": "LT1",
        },
        "LT1_GTATCTTGTATATGAG-1-24": {
            "TRA_1_j_gene": "TRAJ40",
            "TRA_1_v_gene": "TRAV36DV7",
            "TRB_1_v_gene": "TRBV6-3",
            "TRB_1_d_gene": "TRBD1",
            "TRB_1_j_gene": "TRBJ2-5",
            "sample": "LT1",
        },
        "LT1_TGCGCAGAGGGCATGT-1-24": {
            "TRA_1_j_gene": "TRAJ39",
            "TRA_1_v_gene": "TRAV12-3",
            "TRB_1_v_gene": "TRBV11-2",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ2-7",
            "sample": "LT1",
        },
        "LT1_CAGCAGCAGCGCTCCA-2-24": {
            "TRA_1_j_gene": "TRAJ32",
            "TRA_1_v_gene": "TRAV38-2DV8",
            "TRB_1_v_gene": "None",
            "TRB_1_d_gene": "None",
            "TRB_1_j_gene": "TRBJ2-3",
            "sample": "LT1",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_clonotype():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1", "cc1"],
            ["cell2", "A", "ct1", "cc1"],
            ["cell3", "A", "ct1", "cc1"],
            ["cell3", "A", "NaN", "NaN"],
            ["cell4", "B", "ct1", "cc1"],
            ["cell5", "B", "ct2", "cc2"],
            ["cell6", "B", "ct3", "cc2"],
            ["cell7", "B", "ct4", "cc3"],
            ["cell8", "B", "ct4", "cc3"],
        ],
        columns=["cell_id", "group", "clonotype", "clonotype_cluster"],
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
