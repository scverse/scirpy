import pytest
import pandas as pd
from anndata import AnnData
import numpy as np
from scipy.sparse.csr import csr_matrix
from scirpy.util import _is_symmetric
import scirpy as ir


@pytest.fixture
def adata_cdr3():
    obs = pd.DataFrame(
        [
            [
                "cell1",
                "AAA",
                "AHA",
                "KKY",
                "KKK",
                "GCGGCGGCG",
                "TRA",
                "TRB",
                "TRA",
                "TRB",
            ],
            [
                "cell2",
                "AHA",
                "nan",
                "KK",
                "KKK",
                "GCGAUGGCG",
                "TRA",
                "TRB",
                "TRA",
                "TRB",
            ],
            # This row has no chains, but "has_ir" = True. That can happen if
            # the user does not filter the data.
            [
                "cell3",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
                "nan",
            ],
            [
                "cell4",
                "AAA",
                "AAA",
                "LLL",
                "AAA",
                "GCUGCUGCU",
                "TRA",
                "TRB",
                "TRA",
                "TRB",
            ],
            [
                "cell5",
                "AAA",
                "nan",
                "LLL",
                "nan",
                "nan",
                "nan",
                "TRB",
                "TRA",
                "nan",
            ],
        ],
        columns=[
            "cell_id",
            "IR_VJ_1_cdr3",
            "IR_VJ_2_cdr3",
            "IR_VDJ_1_cdr3",
            "IR_VDJ_2_cdr3",
            "IR_VJ_1_cdr3_nt",
            "IR_VJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_1_locus",
            "IR_VDJ_2_locus",
        ],
    ).set_index("cell_id")
    obs["has_ir"] = "True"
    adata = AnnData(obs=obs)
    adata._sanitize()
    return adata


@pytest.fixture
def adata_cdr3_2():
    obs = pd.DataFrame(
        [
            ["c1", "AAA", "AAA", "KKK", "KKK"],
            ["c2", "AAA", "AAA", "LLL", "LLL"],
            ["c3", "nan", "nan", "LLL", "LLL"],
        ],
        columns=[
            "cell_id",
            "IR_VJ_1_cdr3",
            "IR_VJ_2_cdr3",
            "IR_VDJ_1_cdr3",
            "IR_VDJ_2_cdr3",
        ],
    ).set_index("cell_id")
    obs["has_ir"] = "True"
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_define_clonotypes():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "ATA", "GGC", "CCC", "IGK", "IGH", "IGK", "IGH"],
            ["cell2", "AAA", "ATA", "GGC", "CCC", "IGL", "IGH", "IGL", "IGH"],
            ["cell3", "GGG", "ATA", "GGC", "CCC", "IGK", "IGH", "IGK", "IGH"],
            ["cell4", "GGG", "ATA", "GGG", "CCC", "IGK", "IGH", "IGK", "IGH"],
            ["cell10", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
        ],
        columns=[
            "cell_id",
            "IR_VJ_1_cdr3_nt",
            "IR_VJ_2_cdr3_nt",
            "IR_VDJ_1_cdr3_nt",
            "IR_VDJ_2_cdr3_nt",
            "IR_VJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_1_locus",
            "IR_VDJ_2_locus",
        ],
    ).set_index("cell_id")
    obs["has_ir"] = "True"
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_define_clonotype_clusters():
    obs = pd.DataFrame(
        [
            ["cell1", "AAA", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
            ["cell2", "AAA", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
            ["cell3", "BBB", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
            ["cell4", "BBB", "AHA", "BBB", "KKK", "TRA", "TRB", "TRA", "TRB"],
            ["cell5", "AAA", "nan", "KKY", "KKK", "TRA", "nan", "TRA", "TRB"],
            ["cell5.noir", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
            ["cell6", "AAA", "nan", "KKY", "CCC", "TRA", "nan", "TRA", "TRB"],
            ["cell7", "AAA", "AHA", "ZZZ", "nan", "TRA", "TRB", "TRA", "nan"],
            ["cell8", "AAA", "nan", "KKK", "nan", "TRA", "nan", "TRB", "nan"],
            ["cell9", "nan", "nan", "KKK", "nan", "nan", "nan", "TRB", "nan"],
            ["cell10", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
        ],
        columns=[
            "cell_id",
            "IR_VJ_1_cdr3",
            "IR_VJ_2_cdr3",
            "IR_VDJ_1_cdr3",
            "IR_VDJ_2_cdr3",
            "IR_VJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_1_locus",
            "IR_VDJ_2_locus",
        ],
    ).set_index("cell_id")
    obs["has_ir"] = ["True"] * 5 + ["False"] + ["True"] * 5
    adata = AnnData(obs=obs)
    return adata


@pytest.fixture
def adata_define_clonotype_clusters_singletons():
    """Adata where every cell belongs to a singleton clonotype.
    Required for a regression test for #236.
    """
    adata = AnnData(
        obs=pd.DataFrame()
        .assign(
            cell_id=["cell1", "cell2", "cell3", "cell4"],
            IR_VJ_1_cdr3=["AAA", "BBB", "CCC", "DDD"],
            IR_VDJ_1_cdr3=["AAA", "BBB", "CCC", "DDD"],
            IR_VJ_2_cdr3=["AAA", "BBB", "CCC", "DDD"],
            IR_VDJ_2_cdr3=["AAA", "BBB", "CCC", "DDD"],
            IR_VJ_1_v_gene=["A", "B", "C", "D"],
            IR_VDJ_1_v_gene=["A", "B", "C", "D"],
            IR_VJ_2_v_gene=["A", "B", "C", "D"],
            IR_VDJ_2_v_gene=["A", "B", "C", "D"],
            receptor_type=["TCR", "TCR", "TCR", "TCR"],
            has_ir=["True", "True", "True", "True"],
        )
        .set_index("cell_id")
    )
    ir.pp.ir_dist(adata, metric="identity", sequence="aa")
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
            "IR_VJ_1_cdr3_len": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CALSDPNTNAGKSTF",
            "IR_VJ_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clonotype": "clonotype_458",
            "chain_pairing": "Extra alpha",
        },
        "ACTATCTAGGGCTTCC-1": {
            "IR_VJ_1_cdr3_len": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAVDGGTSYGKLTF",
            "IR_VJ_1_cdr3_nt": "TGTGCCGTGGACGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clonotype": "clonotype_739",
            "chain_pairing": "Extra alpha",
        },
        "CAGTAACAGGCATGTG-1": {
            "IR_VJ_1_cdr3_len": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAVRDSNYQLIW",
            "IR_VJ_1_cdr3_nt": "TGTGCTGTGAGAGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clonotype": "clonotype_986",
            "chain_pairing": "Two full chains",
        },
        "CCTTACGGTCATCCCT-1": {
            "IR_VJ_1_cdr3_len": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAVRDSNYQLIW",
            "IR_VJ_1_cdr3_nt": "TGTGCTGTGAGGGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clonotype": "clonotype_987",
            "chain_pairing": "Single pair",
        },
        "CGTCCATTCATAACCG-1": {
            "IR_VJ_1_cdr3_len": 17.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAASRNAGGTSYGKLTF",
            "IR_VJ_1_cdr3_nt": "TGTGCAGCAAGTCGCAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clonotype": "clonotype_158",
            "chain_pairing": "Single pair",
        },
        "CTTAGGAAGGGCATGT-1": {
            "IR_VJ_1_cdr3_len": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CALSDPNTNAGKSTF",
            "IR_VJ_1_cdr3_nt": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "1",
            "clonotype": "clonotype_459",
            "chain_pairing": "Single pair",
        },
        "GCAAACTGTTGATTGC-1": {
            "IR_VJ_1_cdr3_len": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAVDGGTSYGKLTF",
            "IR_VJ_1_cdr3_nt": "TGTGCCGTGGATGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clonotype": "clonotype_738",
            "chain_pairing": "Single pair",
        },
        "GCTCCTACAAATTGCC-1": {
            "IR_VJ_1_cdr3_len": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CALSDPNTNAGKSTF",
            "IR_VJ_1_cdr3_nt": "TGTGCTCTGAGTGATCCCAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clonotype": "clonotype_460",
            "chain_pairing": "Two full chains",
        },
        "GGAATAATCCGATATG-1": {
            "IR_VJ_1_cdr3_len": 17.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAASRNAGGTSYGKLTF",
            "IR_VJ_1_cdr3_nt": "TGTGCAGCAAGTAGGAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clonotype": "clonotype_157",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGATAGCAT-1": {
            "IR_VJ_1_cdr3_len": 13.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAGGGSGTYKYIF",
            "IR_VJ_1_cdr3_nt": "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
            "sample": "3",
            "clonotype": "clonotype_330",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGTACGCCC-1": {
            "IR_VJ_1_cdr3_len": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAMRVGGSQGNLIF",
            "IR_VJ_1_cdr3_nt": "TGTGCAATGAGGGTCGGAGGAAGCCAAGGAAATCTCATCTTT",
            "sample": "5",
            "clonotype": "clonotype_592",
            "chain_pairing": "Two full chains",
        },
        "AAACCTGCATAGAAAC-1": {
            "IR_VJ_1_cdr3_len": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CAFMKPFTAGNQFYF",
            "IR_VJ_1_cdr3_nt": "TGTGCTTTCATGAAGCCTTTTACCGCCGGTAACCAGTTCTATTTT",
            "sample": "5",
            "clonotype": "clonotype_284",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTCCGTTAA-1": {
            "IR_VJ_1_cdr3_len": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CALNTGGFKTIF",
            "IR_VJ_1_cdr3_nt": "TGTGCTCTCAATACTGGAGGCTTCAAAACTATCTTT",
            "sample": "3",
            "clonotype": "clonotype_425",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTTTGTGTG-1": {
            "IR_VJ_1_cdr3_len": 13.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_cdr3": "CALRGGRDDKIIF",
            "IR_VJ_1_cdr3_nt": "TGTGCTCTGAGAGGGGGTAGAGATGACAAGATCATCTTT",
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
            "IR_VJ_1_j_gene": "TRAJ42",
            "IR_VJ_1_v_gene": "TRAV26-2",
            "IR_VDJ_1_v_gene": "TRBV7-2",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ2-5",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CGCTTCACAAGGTGTG-2-24": {
            "IR_VJ_1_j_gene": "TRAJ45",
            "IR_VJ_1_v_gene": "None",
            "IR_VDJ_1_v_gene": "None",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ2-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_AGGGAGTTCCCAAGAT-2-24": {
            "IR_VJ_1_j_gene": "TRAJ29",
            "IR_VJ_1_v_gene": "TRAV12-1",
            "IR_VDJ_1_v_gene": "TRBV20-1",
            "IR_VDJ_1_d_gene": "TRBD2",
            "IR_VDJ_1_j_gene": "TRBJ1-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_ATTACTCGTTGGACCC-2-24": {
            "IR_VJ_1_j_gene": "TRAJ4",
            "IR_VJ_1_v_gene": "TRAV12-1",
            "IR_VDJ_1_v_gene": "TRBV7-2",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ2-6",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GCAATCACAATGAATG-1-24": {
            "IR_VJ_1_j_gene": "TRAJ52",
            "IR_VJ_1_v_gene": "TRAV8-6",
            "IR_VDJ_1_v_gene": "TRBV30",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TCTCTAATCCACTGGG-2-24": {
            "IR_VJ_1_j_gene": "TRAJ43",
            "IR_VJ_1_v_gene": "TRAV8-3",
            "IR_VDJ_1_v_gene": "TRBV30",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ1-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TATTACCTCAACGGCC-2-24": {
            "IR_VJ_1_j_gene": "TRAJ45",
            "IR_VJ_1_v_gene": "TRAV20",
            "IR_VDJ_1_v_gene": "TRBV4-1",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ1-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CGTCAGGTCGAACTGT-1-24": {
            "IR_VJ_1_j_gene": "TRAJ15",
            "IR_VJ_1_v_gene": "TRAV17",
            "IR_VDJ_1_v_gene": "TRBV5-1",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ1-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GGGAATGGTTGCGTTA-2-24": {
            "IR_VJ_1_j_gene": "None",
            "IR_VJ_1_v_gene": "None",
            "IR_VDJ_1_v_gene": "TRBV30",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_AGCTCCTGTAATCGTC-2-24": {
            "IR_VJ_1_j_gene": "TRAJ13",
            "IR_VJ_1_v_gene": "TRAV13-1",
            "IR_VDJ_1_v_gene": "TRBV18",
            "IR_VDJ_1_d_gene": "TRBD2",
            "IR_VDJ_1_j_gene": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CAGCTGGTCCGCGGTA-1-24": {
            "IR_VJ_1_j_gene": "TRAJ30",
            "IR_VJ_1_v_gene": "TRAV21",
            "IR_VDJ_1_v_gene": "TRBV30",
            "IR_VDJ_1_d_gene": "TRBD2",
            "IR_VDJ_1_j_gene": "TRBJ2-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CCTTTCTCAGCAGTTT-1-24": {
            "IR_VJ_1_j_gene": "TRAJ23",
            "IR_VJ_1_v_gene": "TRAV9-2",
            "IR_VDJ_1_v_gene": "TRBV3-1",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ1-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GTATCTTGTATATGAG-1-24": {
            "IR_VJ_1_j_gene": "TRAJ40",
            "IR_VJ_1_v_gene": "TRAV36DV7",
            "IR_VDJ_1_v_gene": "TRBV6-3",
            "IR_VDJ_1_d_gene": "TRBD1",
            "IR_VDJ_1_j_gene": "TRBJ2-5",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TGCGCAGAGGGCATGT-1-24": {
            "IR_VJ_1_j_gene": "TRAJ39",
            "IR_VJ_1_v_gene": "TRAV12-3",
            "IR_VDJ_1_v_gene": "TRBV11-2",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ2-7",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CAGCAGCAGCGCTCCA-2-24": {
            "IR_VJ_1_j_gene": "TRAJ32",
            "IR_VJ_1_v_gene": "TRAV38-2DV8",
            "IR_VDJ_1_v_gene": "None",
            "IR_VDJ_1_d_gene": "None",
            "IR_VDJ_1_j_gene": "TRBJ2-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
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
