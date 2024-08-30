import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData

import scirpy as ir

from .util import _make_adata


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_cdr3(request):
    obs = pd.DataFrame(
        # fmt: off
        [
            ["cell1", "AAA", "AHA", "KKY", "KKK", "GCGGCGGCG", "TRA", "TRB", "TRA", "TRB"],
            ["cell2", "AHA", "nan", "KK", "KKK", "GCGAUGGCG", "TRA", "TRB", "TRA", "TRB"],
            ["cell3", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
            ["cell4", "AAA", "AAA", "LLL", "AAA", "GCUGCUGCU", "TRA", "TRB", "TRA", "TRB"],
            ["cell5", "AAA", "nan", "LLL", "nan", "nan", "nan", "TRB", "TRA", "nan"],
        ],
        # fmt: on
        columns=[
            "cell_id",
            "IR_VJ_1_junction_aa",
            "IR_VJ_2_junction_aa",
            "IR_VDJ_1_junction_aa",
            "IR_VDJ_2_junction_aa",
            "IR_VJ_1_junction",
            "IR_VJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_1_locus",
            "IR_VDJ_2_locus",
        ],
    ).set_index("cell_id")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_cdr3_2(request):
    obs = pd.DataFrame(
        [
            ["c1", "AAA", "AAA", "KKK", "KKK"],
            ["c2", "AAA", "AAA", "LLL", "LLL"],
            ["c3", "nan", "nan", "LLL", "LLL"],
        ],
        columns=[
            "cell_id",
            "IR_VJ_1_junction_aa",
            "IR_VJ_2_junction_aa",
            "IR_VDJ_1_junction_aa",
            "IR_VDJ_2_junction_aa",
        ],
    ).set_index("cell_id")
    adata = _make_adata(obs, request.param)
    uns_ = adata.mod["airr"].uns if isinstance(adata, MuData) else adata.uns
    uns_["DB"] = {"name": "TESTDB"}
    return adata


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_define_clonotypes(request):
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
            "IR_VJ_1_junction",
            "IR_VJ_2_junction",
            "IR_VDJ_1_junction",
            "IR_VDJ_2_junction",
            "IR_VJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_1_locus",
            "IR_VDJ_2_locus",
        ],
    ).set_index("cell_id")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_define_clonotype_clusters(request):
    obs = (
        pd.DataFrame(
            [
                ["cell1", "AAA", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
                ["cell2", "AAA", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
                ["cell3", "BBB", "AHA", "KKY", "KKK", "TRA", "TRB", "TRA", "TRB"],
                ["cell4", "BBB", "AHA", "BBB", "KKK", "TRA", "TRB", "TRA", "TRB"],
                ["cell5", "AAA", "nan", "KKY", "KKK", "TRA", "nan", "TRA", "TRB"],
                # cell5 has no receptor data whatsoever
                ["cell5.noir", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
                ["cell6", "AAA", "nan", "KKY", "CCC", "TRA", "nan", "TRA", "TRB"],
                ["cell7", "AAA", "AHA", "ZZZ", "nan", "TRA", "TRB", "TRA", "nan"],
                ["cell8", "AAA", "nan", "KKK", "nan", "TRA", "nan", "TRB", "nan"],
                ["cell9", "nan", "nan", "KKK", "nan", "nan", "nan", "TRB", "nan"],
                # while cell 10 has no CDR3 sequences, but v-calls and a receptor type.
                ["cell10", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"],
            ],
            columns=[
                "cell_id",
                "IR_VJ_1_junction_aa",
                "IR_VJ_2_junction_aa",
                "IR_VDJ_1_junction_aa",
                "IR_VDJ_2_junction_aa",
                "IR_VJ_1_locus",
                "IR_VJ_2_locus",
                "IR_VDJ_1_locus",
                "IR_VDJ_2_locus",
            ],
        )
        .set_index("cell_id")
        .join(
            pd.DataFrame(
                [
                    ["cell1", "A", "B", "A", "B", "TCR"],
                    ["cell2", "A", "A", "A", "A", "TCR"],
                    ["cell3", "A", "A", "A", "A", "TCR"],
                    ["cell4", "C", "C", "C", "C", "BCR"],
                    ["cell5", "A", "A", "A", "A", "BCR"],
                    ["cell5.noir", "nan", "nan", "nan", "nan", "nan"],
                    ["cell6", "A", "A", "A", "A", "TCR"],
                    ["cell7", "A", "A", "A", "A", "TCR"],
                    ["cell8", "A", "A", "X", "A", "TCR"],
                    ["cell9", "A", "A", "A", "A", "BCR"],
                    ["cell10", "A", "A", "A", "A", "BCR"],
                ],
                columns=[
                    "cell_id",
                    "IR_VJ_1_v_call",
                    "IR_VJ_2_v_call",
                    "IR_VDJ_1_v_call",
                    "IR_VDJ_2_v_call",
                    "receptor_type",
                ],
            ).set_index("cell_id")
        )
    )
    return _make_adata(obs, request.param)


@pytest.fixture
def adata_clonotype_modularity(adata_define_clonotypes):
    data = adata_define_clonotypes
    adata = data.mod["airr"] if isinstance(data, MuData) else data
    adata.obs["clone_id"] = ["0", "1", "2", "2", "nan"]
    # Since the results depend on both GEX and TCR data, the results are stored in the mudata object directly.
    data.obs["clonotype_modularity_x"] = [0, 0, 4, 4, np.nan]
    data.obs["clonotype_modularity_x_fdr"] = [1, 1, 1e-6, 1e-6, np.nan]
    data.uns["clonotype_modularity_x"] = {
        "target_col": "clone_id",
        "fdr_correction": True,
    }
    return data


@pytest.fixture
def adata_conn(adata_define_clonotype_clusters):
    """Stub adata to test the clonotype_network functions"""
    adata = adata_define_clonotype_clusters
    ir.pp.ir_dist(adata, sequence="aa", metric="alignment")
    ir.tl.define_clonotype_clusters(adata, sequence="aa", metric="alignment", receptor_arms="any", dual_ir="any")
    return adata


@pytest.fixture
def adata_define_clonotype_clusters_singletons():
    """Adata where every cell belongs to a singleton clonotype.
    Required for a regression test for #236.
    """
    obs = (
        pd.DataFrame()
        .assign(
            cell_id=["cell1", "cell2", "cell3", "cell4"],
            IR_VJ_1_junction_aa=["AAA", "BBB", "CCC", "DDD"],
            IR_VDJ_1_junction_aa=["AAA", "BBB", "CCC", "DDD"],
            IR_VJ_2_junction_aa=["AAA", "BBB", "CCC", "DDD"],
            IR_VDJ_2_junction_aa=["AAA", "BBB", "CCC", "DDD"],
            IR_VJ_1_v_call=["A", "B", "C", "D"],
            IR_VDJ_1_v_call=["A", "B", "C", "D"],
            IR_VJ_2_v_call=["A", "B", "C", "D"],
            IR_VDJ_2_v_call=["A", "B", "C", "D"],
            receptor_type=["TCR", "TCR", "TCR", "TCR"],
        )
        .set_index("cell_id")
    )
    adata = _make_adata(obs)
    ir.pp.ir_dist(adata, metric="identity", sequence="aa")
    return adata


@pytest.fixture
def adata_clonotype_network(adata_conn, request):
    """Adata with clonotype network computed.

    adata derived from adata_conn that also contains some gene expression data
    for plotting.
    """
    try:
        kwargs = request.param
    except AttributeError:
        kwargs = {}
    if isinstance(adata_conn, AnnData):
        adata = AnnData(
            var=pd.DataFrame().assign(gene_symbol=["CD8A", "CD4"]).set_index("gene_symbol"),
            X=np.array(
                [
                    [3, 4, 0, 0, 3, 3, 1, 0, 2, 2, 0],
                    [0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0],
                ]
            ).T,
            obs=adata_conn.obs,
            uns=adata_conn.uns,
            obsm=adata_conn.obsm,
        )
        adata.obs["continuous"] = [3, 4, 0, 0, 7, 14, 1, 0, 2, 2, 0]
        ir.tl.clonotype_network(adata, sequence="aa", metric="alignment", **kwargs)
        return adata
    else:
        adata_gex = AnnData(
            var=pd.DataFrame().assign(gene_symbol=["CD8A", "CD4"]).set_index("gene_symbol"),
            X=np.array(
                [
                    [3, 4, 0, 0, 3, 3, 1, 0, 2, 2, 0],
                    [0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0],
                ]
            ).T,
            obs=adata_conn.obs.loc[:, []],
        )
        mdata = MuData({"gex": adata_gex, "airr": adata_conn.mod["airr"]})
        mdata.obs["continuous"] = [3, 4, 0, 0, 7, 14, 1, 0, 2, 2, 0]
        ir.tl.clonotype_network(mdata, sequence="aa", metric="alignment", **kwargs)
        return mdata


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_tra(request):
    obs = {
        "AAGGTTCCACCCAGTG-1": {
            "IR_VJ_1_junction_aa_length": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CALSDPNTNAGKSTF",
            "IR_VJ_1_junction": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clone_id": "clonotype_458",
            "chain_pairing": "Extra alpha",
        },
        "ACTATCTAGGGCTTCC-1": {
            "IR_VJ_1_junction_aa_length": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAVDGGTSYGKLTF",
            "IR_VJ_1_junction": "TGTGCCGTGGACGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clone_id": "clonotype_739",
            "chain_pairing": "Extra alpha",
        },
        "CAGTAACAGGCATGTG-1": {
            "IR_VJ_1_junction_aa_length": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAVRDSNYQLIW",
            "IR_VJ_1_junction": "TGTGCTGTGAGAGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clone_id": "clonotype_986",
            "chain_pairing": "Two full chains",
        },
        "CCTTACGGTCATCCCT-1": {
            "IR_VJ_1_junction_aa_length": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAVRDSNYQLIW",
            "IR_VJ_1_junction": "TGTGCTGTGAGGGATAGCAACTATCAGTTAATCTGG",
            "sample": "1",
            "clone_id": "clonotype_987",
            "chain_pairing": "Single pair",
        },
        "CGTCCATTCATAACCG-1": {
            "IR_VJ_1_junction_aa_length": 17.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAASRNAGGTSYGKLTF",
            "IR_VJ_1_junction": "TGTGCAGCAAGTCGCAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clone_id": "clonotype_158",
            "chain_pairing": "Single pair",
        },
        "CTTAGGAAGGGCATGT-1": {
            "IR_VJ_1_junction_aa_length": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CALSDPNTNAGKSTF",
            "IR_VJ_1_junction": "TGTGCTCTGAGTGACCCTAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "1",
            "clone_id": "clonotype_459",
            "chain_pairing": "Single pair",
        },
        "GCAAACTGTTGATTGC-1": {
            "IR_VJ_1_junction_aa_length": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAVDGGTSYGKLTF",
            "IR_VJ_1_junction": "TGTGCCGTGGATGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "1",
            "clone_id": "clonotype_738",
            "chain_pairing": "Single pair",
        },
        "GCTCCTACAAATTGCC-1": {
            "IR_VJ_1_junction_aa_length": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CALSDPNTNAGKSTF",
            "IR_VJ_1_junction": "TGTGCTCTGAGTGATCCCAACACCAATGCAGGCAAATCAACCTTT",
            "sample": "3",
            "clone_id": "clonotype_460",
            "chain_pairing": "Two full chains",
        },
        "GGAATAATCCGATATG-1": {
            "IR_VJ_1_junction_aa_length": 17.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAASRNAGGTSYGKLTF",
            "IR_VJ_1_junction": "TGTGCAGCAAGTAGGAATGCTGGTGGTACTAGCTATGGAAAGCTGACATTT",
            "sample": "5",
            "clone_id": "clonotype_157",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGATAGCAT-1": {
            "IR_VJ_1_junction_aa_length": 13.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAGGGSGTYKYIF",
            "IR_VJ_1_junction": "TGTGCAGGGGGGGGCTCAGGAACCTACAAATACATCTTT",
            "sample": "3",
            "clone_id": "clonotype_330",
            "chain_pairing": "Single pair",
        },
        "AAACCTGAGTACGCCC-1": {
            "IR_VJ_1_junction_aa_length": 14.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAMRVGGSQGNLIF",
            "IR_VJ_1_junction": "TGTGCAATGAGGGTCGGAGGAAGCCAAGGAAATCTCATCTTT",
            "sample": "5",
            "clone_id": "clonotype_592",
            "chain_pairing": "Two full chains",
        },
        "AAACCTGCATAGAAAC-1": {
            "IR_VJ_1_junction_aa_length": 15.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CAFMKPFTAGNQFYF",
            "IR_VJ_1_junction": "TGTGCTTTCATGAAGCCTTTTACCGCCGGTAACCAGTTCTATTTT",
            "sample": "5",
            "clone_id": "clonotype_284",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTCCGTTAA-1": {
            "IR_VJ_1_junction_aa_length": 12.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CALNTGGFKTIF",
            "IR_VJ_1_junction": "TGTGCTCTCAATACTGGAGGCTTCAAAACTATCTTT",
            "sample": "3",
            "clone_id": "clonotype_425",
            "chain_pairing": "Extra alpha",
        },
        "AAACCTGGTTTGTGTG-1": {
            "IR_VJ_1_junction_aa_length": 13.0,
            "IR_VJ_1_locus": "TRA",
            "IR_VJ_1_junction_aa": "CALRGGRDDKIIF",
            "IR_VJ_1_junction": "TGTGCTCTGAGAGGGGGTAGAGATGACAAGATCATCTTT",
            "sample": "3",
            "clone_id": "clonotype_430",
            "chain_pairing": "Single pair",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_vdj(request):
    obs = {
        "LT1_ACGGCCATCCGAGCCA-2-24": {
            "IR_VJ_1_j_call": "TRAJ42",
            "IR_VJ_1_v_call": "TRAV26-2",
            "IR_VDJ_1_v_call": "TRBV7-2",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ2-5",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CGCTTCACAAGGTGTG-2-24": {
            "IR_VJ_1_j_call": "TRAJ45",
            "IR_VJ_1_v_call": "None",
            "IR_VDJ_1_v_call": "None",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ2-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_AGGGAGTTCCCAAGAT-2-24": {
            "IR_VJ_1_j_call": "TRAJ29",
            "IR_VJ_1_v_call": "TRAV12-1",
            "IR_VDJ_1_v_call": "TRBV20-1",
            "IR_VDJ_1_d_call": "TRBD2",
            "IR_VDJ_1_j_call": "TRBJ1-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_ATTACTCGTTGGACCC-2-24": {
            "IR_VJ_1_j_call": "TRAJ4",
            "IR_VJ_1_v_call": "TRAV12-1",
            "IR_VDJ_1_v_call": "TRBV7-2",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ2-6",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GCAATCACAATGAATG-1-24": {
            "IR_VJ_1_j_call": "TRAJ52",
            "IR_VJ_1_v_call": "TRAV8-6",
            "IR_VDJ_1_v_call": "TRBV30",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TCTCTAATCCACTGGG-2-24": {
            "IR_VJ_1_j_call": "TRAJ43",
            "IR_VJ_1_v_call": "TRAV8-3",
            "IR_VDJ_1_v_call": "TRBV30",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ1-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TATTACCTCAACGGCC-2-24": {
            "IR_VJ_1_j_call": "TRAJ45",
            "IR_VJ_1_v_call": "TRAV20",
            "IR_VDJ_1_v_call": "TRBV4-1",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ1-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CGTCAGGTCGAACTGT-1-24": {
            "IR_VJ_1_j_call": "TRAJ15",
            "IR_VJ_1_v_call": "TRAV17",
            "IR_VDJ_1_v_call": "TRBV5-1",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ1-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GGGAATGGTTGCGTTA-2-24": {
            "IR_VJ_1_j_call": "None",
            "IR_VJ_1_v_call": "None",
            "IR_VDJ_1_v_call": "TRBV30",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_AGCTCCTGTAATCGTC-2-24": {
            "IR_VJ_1_j_call": "TRAJ13",
            "IR_VJ_1_v_call": "TRAV13-1",
            "IR_VDJ_1_v_call": "TRBV18",
            "IR_VDJ_1_d_call": "TRBD2",
            "IR_VDJ_1_j_call": "TRBJ2-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CAGCTGGTCCGCGGTA-1-24": {
            "IR_VJ_1_j_call": "TRAJ30",
            "IR_VJ_1_v_call": "TRAV21",
            "IR_VDJ_1_v_call": "TRBV30",
            "IR_VDJ_1_d_call": "TRBD2",
            "IR_VDJ_1_j_call": "TRBJ2-1",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CCTTTCTCAGCAGTTT-1-24": {
            "IR_VJ_1_j_call": "TRAJ23",
            "IR_VJ_1_v_call": "TRAV9-2",
            "IR_VDJ_1_v_call": "TRBV3-1",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ1-2",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_GTATCTTGTATATGAG-1-24": {
            "IR_VJ_1_j_call": "TRAJ40",
            "IR_VJ_1_v_call": "TRAV36DV7",
            "IR_VDJ_1_v_call": "TRBV6-3",
            "IR_VDJ_1_d_call": "TRBD1",
            "IR_VDJ_1_j_call": "TRBJ2-5",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_TGCGCAGAGGGCATGT-1-24": {
            "IR_VJ_1_j_call": "TRAJ39",
            "IR_VJ_1_v_call": "TRAV12-3",
            "IR_VDJ_1_v_call": "TRBV11-2",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ2-7",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
        "LT1_CAGCAGCAGCGCTCCA-2-24": {
            "IR_VJ_1_j_call": "TRAJ32",
            "IR_VJ_1_v_call": "TRAV38-2DV8",
            "IR_VDJ_1_v_call": "None",
            "IR_VDJ_1_d_call": "None",
            "IR_VDJ_1_j_call": "TRBJ2-3",
            "IR_VJ_1_locus": "TRA",
            "IR_VDJ_1_locus": "TRB",
            "sample": "LT1",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_clonotype(request):
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1", "cc1"],
            ["cell2", "A", "ct1", "cc1"],
            ["cell3", "A", "ct1", "cc1"],
            ["cell3.1", "A", np.nan, np.nan],
            ["cell3.2", "A", np.nan, np.nan],
            ["cell4", "B", "ct1", "cc1"],
            ["cell5", "B", "ct2", "cc2"],
            ["cell6", "B", "ct3", "cc2"],
            ["cell7", "B", "ct4", "cc3"],
            ["cell8", "B", "ct4", "cc3"],
        ],
        columns=["cell_id", "group", "clone_id", "clonotype_cluster"],
    ).set_index("cell_id")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_diversity(request):
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3.1", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype_"],
    ).set_index("cell_id")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_mutation(request):
    # real data from Stephenson et al.2021, but germline alignments copy from sequence alignments and manually manipulated
    obs = {
        "AAACGGGCACGACTCG-MH9179822": {
            # 1
            # no mutation
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTACAGCAGTGGGGCGCA...GGACTGTTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTC............AGTGGTTACTACTGGAGCTGGATCCGCCAGCCCCCAGGGAAGGGGCTGGAGTGGATTGGGGAAATCAATCATAGT.........GGAAGCACCAACTACAACCCGTCCCTCAAG...AGTCGAGTCACCATATCAGTAGACACGTCCAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACCGCCGCGGACACGGCTGTGTATTACTGTGCGAGAGGCTTCTGTAGTAGTACCAGCTGCTATACGGGGCGGGGTGGTAGGTACTACTACTACATGGACGTCTGGGGCAAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_germline_alignment": "CAGGTGCAGCTACAGCAGTGGGGCGCA...GGACTGTTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCGCTGTCTATGGTGGGTCCTTC............AGTGGTTACTACTGGAGCTGGATCCGCCAGCCCCCAGGGAAGGGGCTGGAGTGGATTGGGGAAATCAATCATAGT.........GGAAGCACCAACTACAACCCGTCCCTCAAG...AGTCGAGTCACCATATCAGTAGACACGTCCAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACCGCCGCGGACACGGCTGTGTATTACTGTGCGAGAGGCTTCTGTAGTAGTACCAGCTGCTATACGGGGCGGGGTGGTAGGTACTACTACTACATGGACGTCTGGGGCAAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_junction": "TGTGCGAGAGGCTTCTGTAGTAGTACCAGCTGCTATACGGGGCGGGGTGGTAGGTACTACTACTACATGGACGTCTGG",
            "IR_VJ_1_sequence_alignment": "CAGTCTGCCCTGACTCAGCCTGCCTCC...GTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGACGTTGGT.........GGTTATAACTATGTCTCCTGGTACCAACAGCACCCAGGCAAAGCCCCCAAACTCATGATTTATGATGTC.....................AGTAATCGGCCCTCAGGGGTTTCT...AATCGCTTCTCTGGCTCCAAG......TCTGGCAACACGGCCTCCCTGACCATCTCTGGGCTCCAGGCTGAGGACGAGGCTGATTATTACTGCAGCTCATATACAAGCAGCAGCACCCCTAATGTCTTCGGAACTGGGACCAAGGTCACCGTCCTAG",
            "IR_VJ_1_germline_alignment": "CAGTCTGCCCTGACTCAGCCTGCCTCC...GTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGACGTTGGT.........GGTTATAACTATGTCTCCTGGTACCAACAGCACCCAGGCAAAGCCCCCAAACTCATGATTTATGATGTC.....................AGTAATCGGCCCTCAGGGGTTTCT...AATCGCTTCTCTGGCTCCAAG......TCTGGCAACACGGCCTCCCTGACCATCTCTGGGCTCCAGGCTGAGGACGAGGCTGATTATTACTGCAGCTCATATACAAGCAGCAGCACCCCTAATGTCTTCGGAACTGGGACCAAGGTCACCGTCCTAG",
            "IR_VJ_1_junction": "TGCAGCTCATATACAAGCAGCAGCACCCCTAATGTCTTC",
            "IR_VJ_1_locus": "IGL",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AACCATGAGAGCAATT-MH9179822": {
            # 2
            # no mutation, but germline cdr3 masked with 35 "N" in VDJ and 5 "N" in VJ
            "IR_VDJ_1_sequence_alignment": "GACGTGCACCTGTTGGAGTCTGGGGGA...GGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGAAGCCTCTGGATTCACCTTT............AGCAACTATGCCATGAATTGGGTCCGCCAGGCTCCAGGAAAGGGGCTGGAGTGGGTCTCAACTATTAGTGGCAGT......GGTGGTAGCGCATACTACGGCGATTCCGTGAAG...GGCCGCTTCACCATCTCCAGAGACAATTCCAAGAGCACGCTGTTTCTGCAAATGAGCAGCTTGAGAGTCGACGACACGGCCCTATATTTCTGTGCGAAAGGCCGCCAATATGAAGATATTTTGACTGCATTTGACGACTGGGGCCAGGGTACCCTGGTTACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": "GACGTGCACCTGTTGGAGTCTGGGGGA...GGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGAAGCCTCTGGATTCACCTTT............AGCAACTATGCCATGAATTGGGTCCGCCAGGCTCCAGGAAAGGGGCTGGAGTGGGTCTCAACTATTAGTGGCAGT......GGTGGTAGCGCATACTACGGCGATTCCGTGAAG...GGCCGCTTCACCATCTCCAGAGACAATTCCAAGAGCACGCTGTTTCTGCAAATGAGCAGCTTGAGAGTCGACGACACGGCCCTATATTTCTGTGCGAAAGGCCGCCAATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAGGGTACCCTGGTTACCGTCTCCTCAG",
            "IR_VDJ_1_junction": "TGTGCGAAAGGCCGCCAATATGAAGATATTTTGACTGCATTTGACGACTGG",
            "IR_VJ_1_sequence_alignment": "GACATCGTGATGACCCAGTCTCCAGACTCCCTGGCTGTGTCTCTGGGCGAGAGGGCCACCATCAAGTGCAAGTCCAGCCAGAGTGTTTTATACAAGTCCAACAATAAGAACTACTTAGCTTGGTACCAGCAGAAACCAGGACAGCCTCCTAAATTGCTCATTTCCTGGGCC.....................TCTACCCGGGAATCCGGAGTCCCT...GACCGGTTCAGTGGCAGCGGG......TCTGGGACAGATTTCACTCTCACCATCAGCAGCCTGCAGGCTGAGGATGTGGCAGTTTATTACTGTCAGCAGTATTACAGTCTTCCTCCGGTCACTTTCGGCGGAGGGACCAAGGTGGAGATCA",
            "IR_VJ_1_germline_alignment": "GACATCGTGATGACCCAGTCTCCAGACTCCCTGGCTGTGTCTCTGGGCGAGAGGGCCACCATCAAGTGCAAGTCCAGCCAGAGTGTTTTATACAAGTCCAACAATAAGAACTACTTAGCTTGGTACCAGCAGAAACCAGGACAGCCTCCTAAATTGCTCATTTCCTGGGCC.....................TCTACCCGGGAATCCGGAGTCCCT...GACCGGTTCAGTGGCAGCGGG......TCTGGGACAGATTTCACTCTCACCATCAGCAGCCTGCAGGCTGAGGATGTGGCAGTTTATTACTGTCAGCAGTATTACAGTCTTCCNNNNNTCACTTTCGGCGGAGGGACCAAGGTGGAGATCA",
            "IR_VJ_1_junction": "TGTCAGCAGTATTACAGTCTTCCTCCGGTCACTTTC",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AACCATGCAGTCACTA-MH9179822": {
            # 3
            # no mutation, but sequence alignment poor sequence quality at beginning: 15 '.'
            "IR_VDJ_1_sequence_alignment": "...............GAATCTGGGGGA...GGCGTGGTCCAGCCCGGGAGGTCCCTGAGACTCTCCTGTGTAACCTCTGGATTCAACATC............AATAATTATGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGACTGGAATGGGTGGCACTTATTTCATACGAA......GGAAGTAAAAAGGTCTATGCAGACTCCTTGAAG...GGCCGATTCATTATCTCCAGAGACAATTCCAAGAACACGGTGTTTCTGCAGATGGACAGCCTGAGACCTGAGGACACGGCCGTCTATTATTGTGCGAAAGGGGGTCAGATCTTTCATTTTTCGAGTGGTTTTTATTTTGACTTCTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": "CAAGTACAATTGGTGGAATCTGGGGGA...GGCGTGGTCCAGCCCGGGAGGTCCCTGAGACTCTCCTGTGTAACCTCTGGATTCAACATC............AATAATTATGGCATGCACTGGGTCCGCCAGGCTCCAGGCAAGGGACTGGAATGGGTGGCACTTATTTCATACGAA......GGAAGTAAAAAGGTCTATGCAGACTCCTTGAAG...GGCCGATTCATTATCTCCAGAGACAATTCCAAGAACACGGTGTTTCTGCAGATGGACAGCCTGAGACCTGAGGACACGGCCGTCTATTATTGTGCGAAAGGGGGTCAGATCTTTCATTTTTCGAGTGGTTTTTATTTTGACTTCTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_junction": "TGTGCGAAAGGGGGTCAGATCTTTCATTTTTCGAGTGGTTTTTATTTTGACTTCTGG",
            "IR_VJ_1_sequence_alignment": "...............CAGTCTCCAGGCACCCTGTCTTTGTCTCCAGGGCAAAGAGCCACCCTCTCTTGCAGGGCCAGTCAGACTGTTGAC...............AACAACTACTTAGCCTGGTATCGACACAAACCTGGCCAGGCTCCCAGCCTCCTCATTTATGGTGCA.....................TCCAGCAGGGCCACTGGCATCCCA...GACAGGTTCAGTGGCAGTGGA......TCTGAGACAGACTTCACTCTCACCATCAGCAGACTGGAGGCTGCAGATTTTGCAGTGTATTACTGTCAACAGTATGGTAGCTCACCGCTCACTTTCGGCGGAGGGACCAAGGTGGAGATCAAAC",
            "IR_VJ_1_germline_alignment": "GAAATTGTGTTGACGCAGTCTCCAGGCACCCTGTCTTTGTCTCCAGGGCAAAGAGCCACCCTCTCTTGCAGGGCCAGTCAGACTGTTGAC...............AACAACTACTTAGCCTGGTATCGACACAAACCTGGCCAGGCTCCCAGCCTCCTCATTTATGGTGCA.....................TCCAGCAGGGCCACTGGCATCCCA...GACAGGTTCAGTGGCAGTGGA......TCTGAGACAGACTTCACTCTCACCATCAGCAGACTGGAGGCTGCAGATTTTGCAGTGTATTACTGTCAACAGTATGGTAGCTCACCGCTCACTTTCGGCGGAGGGACCAAGGTGGAGATCAAAC",
            "IR_VJ_1_junction": "TGTCAACAGTATGGTAGCTCACCGCTCACTTTC",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AACGTTGGTATAAACG-MH9179822": {
            # 4
            # no mutation, but gaps ('-') in sequence alignment: 3 in FWR1, 3 in FWR2 and 5 in FWR4
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTGCAGGAGTCGGGCCCA...GGACTGGTGAAGC---CGGAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATC............AGTGGTTACTACTGGACCTGGA---GGCAGCCCCCAGGGAAGGGACTGGAATGGATTGGATATATCTATTACAGT.........GGGACCACCAACTACAACCCCTCCCTCAAG...AGTCGAGTCACCTTATCAGTGGACACGTCCAAGAACCAGTTCTCCCTGAGGCTGAGTTCTGTGACCGCTGCGGACACGGCCGTGTATTACTGTGCGAGAGACAATTTGTTCTACTACCAGATGGACGTC-----CAAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_germline_alignment": "CAGGTGCAGCTGCAGGAGTCGGGCCCA...GGACTGGTGAAGCCTTCGGAGACCCTGTCCCTCACCTGCACTGTCTCTGGTGGCTCCATC............AGTGGTTACTACTGGACCTGGATCCGGCAGCCCCCAGGGAAGGGACTGGAATGGATTGGATATATCTATTACAGT.........GGGACCACCAACTACAACCCCTCCCTCAAG...AGTCGAGTCACCTTATCAGTGGACACGTCCAAGAACCAGTTCTCCCTGAGGCTGAGTTCTGTGACCGCTGCGGACACGGCCGTGTATTACTGTGCGAGAGACAATTTGTTCTACTACCAGATGGACGTCTGGGGCAAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_junction": "TGTGCGAGAGACAATTTGTTCTACTACCAGATGGACGTCTGG",
            "IR_VJ_1_sequence_alignment": "GACATCCAGATGACCCAGTCTCCAT---CCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGAACATT..................AACAGCTATTTAAAT---TATCAACAAAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATGCTGCA.....................TCCAGTTTGCAAAGTGGAGTCCCA...TCAAGGTTCAGTGGCAGTGGA......TCTGGGACAGATTTCACTCTCACCATCAGTAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGACTTACAGTACCCCGTGGACG-----CCAAGGGACCAAGGTGGAAATCAAAC",
            "IR_VJ_1_germline_alignment": "GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGAACATT..................AACAGCTATTTAAATTGGTATCAACAAAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATGCTGCA.....................TCCAGTTTGCAAAGTGGAGTCCCA...TCAAGGTTCAGTGGCAGTGGA......TCTGGGACAGATTTCACTCTCACCATCAGTAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGACTTACAGTACCCCGTGGACGTTCGGCCAAGGGACCAAGGTGGAAATCAAAC",
            "IR_VJ_1_junction": "TGTCAACAGACTTACAGTACCCCGTGGACGTTC",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AACTCTTGTTTGGCGC-MH9179822": {
            # 6
            # few mutations: 1 in each subregion of sequence_alignment (= 7 in total)
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCGTGGTCCAGCCTGGCAGGTCCCTGAGAATCTCCTGTGCAGCCTCTGGATTCAGCTTC............AGTATCTATGGCGTGCACTGGGTCCGCCAAGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGATATATCATATGAA......GGTAGTCTTTAAAACTATGACGACTCCGTGAAG...GGCCGATTCACCATCTCCAGAGACAATTCCAAGAAGACGGATTATCTGCAAATGGACAGCCTGAGAAGTGAGGACACGGCTGTATATTACTGCGCGAAGCGCCGGCCTGTTTTTGCCTTGAGTGGTGGTTTTGTCGACTACTCGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": "CAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCGTGGTCCAGCCTGGGAGGTCCCTGAGAATCTCCTGTGCAGCCTCTGGATTCAGCTTC............AGTAACTATGGCGTGCACTGGGTCCGCCAGGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGATATATCATATGAA......GGTAGTCTTAAAAACTATGACGACTCCGTGAAG...GGCCGATTCACCATCTCCAGAGACAATTCCAAGAAGACGGTTTATCTGCAAATGGACAGCCTGAGAAGTGAGGACACGGCTGTATATTACTGCGCGAAGCGCCGGTCTGTTTTTGCCTTGAGTGGTGGTTTTGTCGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_junction": "TGCGCGAAGCGCCGGTCTGTTTTTGCCTTGAGTGGTGGTTTTGTCGACTACTGG",
            "IR_VJ_1_sequence_alignment": "CAATCTGCCCTGACTCAGCCTGCCTCC...GTGTCTGGGTCTCCTGGACAGTCGATCACCTTCTCCTGCACTGGAACCAGCAGTGAGATTGGT.........GATTATAACTATGTCTCCTGCTACCAACAACACCCAGGCAATGCCCCCAAACTCATGATTTATGATGGC.....................AGTAATCGGCCCTCAGGGGTTTCT...ATTCGCTTCTCTGGCTCCAAG......TCTGGCAACACGGCCTCCCTGACCATCTCTGGGCTCCAGGCTGAGGACGAGGCTGATTATTTCTGCTCCTCATATACAACCATCAACACTTGGGTGTTCGGCGGAGGGAGCAAGGTGACCGTCCTA",
            "IR_VJ_1_germline_alignment": "CAATCTGCCCTGACTCAGCCTGCCTCC...GTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGACATTGGT.........GATTATAACTATGTCTCCTGGTACCAACAACACCCAGGCAATGCCCCCAAACTCATGATTTATGATGTC.....................AGTAATCGGCCCTCAGGGGTTTCT...ATTCGCTTCTCTGGCTCCAAG......TCTGGCAACACGGCCTCCCTGACCATCTCTGGGCTCCAGGCTGAGGACGAGGCTGATTATTACTGCACCTCATATACAACCATCAACACTTGGGTGTTCGGCGGAGGGACCAAGGTGACCGTCCTA",
            "IR_VJ_1_junction": "TGCACCTCATATACAACCATCAACACTTGGGTGTTC",
            "IR_VJ_1_locus": "IGL",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AACTGGTCAATTGCTG-MH9179822": {
            # 7
            # some mutations: 3 in each subregion of germline alignment (= 21 in total)
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTGGTGCAGTCTGGGGCT...GAGGTGAAGAAGCCTGGGTCCTCGGTGAAGGTCTCCTGCAAGGCTTCTGGAGGCACCTTC............AGCAGCTATGCTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGTGGATGGGAGGGATCATCCCTATC......TTTGGTACAGCAAACTACGCACAGAAGTTCCAG...GGCAGAGTCACGATTACCGCGGACGAATCCACGAGCACAGCCTACATGGAGCTGAGCAGCCTGAGATCTGAGGACACGGCCGTGTATTACTGTGCGAGAGAAGATGGTTCGGGGGTGTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": "CAGGTGCAGCTGGTGCAGTCTGGGGCT...GAGGTATTGAAGCCTGGGTCCTCGGTGAAGGTCTCCTGCAAGGCTTCTGGAGGCACCTTC............TTAAGCTATGCTATCAGCTGGGTGCGACAGGCCCCTGGACAAGGGCTTGAGATAATGGGAGGGATCATCCCTATC......AAAGGTACAGCAAACTACGCACAGAAGTTCCAG...GGCAGAGTCACGATTACCGCGGACGAATCCACGAGCACAGCCTACATGGAGCTGAGCAGCCACTGATCTGAGGACACGGCCGTGTATTACTGTGCGAGAGAAGATGGAAGGGGGGTGTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTTTC",
            "IR_VDJ_1_junction": "TGTGCGAGAGAAGATGGTTCGGGGGTGTTTGACTACTGG",
            "IR_VJ_1_sequence_alignment": "CAGTCTGTGCTGACTCAGCCACCCTCA...GCGTCTGGGACCCCCGGGCAGAGGGTCACCATCTCTTGTTCTGGAAGCAGCTCCAACATC............GGAAGTAATTATGTATACTGGTACCAGCAGCTCCCAGGAACGGCCCCCAAACTCCTCATCTATAGGAAT.....................AATCAGCGGCCCTCAGGGGTCCCT...GACCGATTCTCTGGCTCCAAG......TCTGGCACCTCAGCCTCCCTGGCCATCAGTGGGCTCCGGTCCGAGGATGAGGCTGATTATTACTGTGCAGCATGGGATGACAGCCTGAGTGGTCCGGTGTTCGGCGGAGGGACCAAGCTGACCGTCCTAG",
            "IR_VJ_1_germline_alignment": "CAGTCTGTGCTGACTCAGCCACCCTCA...GCGTCTGGGACCCCGCCGCAGAGGGTCACCATCTCTTGTTCTGGAAGCAGCAGGAACATC............GGAAGTAATTATGTATACTAACACCAGCAGCTCCCAGGAACGGCCCCCAAACTCCTCATCTATAGGTTG.....................AATCAGCGGCCCTCAGGGGTCCCT...GACCGATTCTCTGGCTCCAAG......TCTGGCACCTCAGCCTCCCAAACCATCAGTGGGCTCCGGTCCGAGGATGAGGCTGATTATTACTGTGCAGCATTTTATGACAGCCTGAGTGGTCCGGTGTTCGGCGGAGGGACCAAGCTTTTCGTCCTAG",
            "IR_VJ_1_junction": "TGTGCAGCATGGGATGACAGCCTGAGTGGTCCGGTGTTC",
            "IR_VJ_1_locus": "IGL",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AAGCCGCAGATATACG-MH9179822": {
            # 8
            # a lot mutation: 5 in each subregion of germline alignment (= 35 in total)
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCGTGGTCCAGCCTGGGAGGTCCCTGAGACTCTCCTGTATAGCCTCTGGATTCACCTTC............AATAATTATGGCATGCACTGGGTCCGCCGGGCTCCAGGCAAGGGGCTGGAGTGGGTGGCAGTTATATCATATGAA......GGAAGTAATAAAAATTATGGAGACTCCGTGAAG...GGCCGATTCACCATCTCCAGAGACGGTTCCAAGAGCACACTGTATCTGCAAATGAACAGCCTGAGAGCTGAGGACACGGCTGTGTATTACTGTGTGAAAGCCGGTCAGATTTTCGATAATTCGAGTGGTTATTATTTTGAGTACTGGGGCCAGGGAACTCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": "CAGGTGCAGCTGGTGGAGTCTGGGGGA...AACGTGGTCCAGCCTGGGAGGTCCCTGAGACAGACCTGTATAGCCTCTGGATTCACCAAG............TTTAATTATGGCATGCACTGGGTCCGCCGGGCTCCAGGCCCGGGGCTGGAGTGGGTGGCACAAATATCATATGCC......TTAAATAATAAAAATTATGGAGACTCCGTGAAG...GGCCGATTCACCATCTCCAGAGACGGTTCCAAGAGCACACTGTATCTGCCCCTGAACAGCCTGAGAGCTGATTACACGGCTGTGTATTACTGTGTGAAAGGGGGTCAGATTTTCGATTTTTCGAGTGGTTATTATTTTGACTACTGGGGCCAGGGAACTCTGGTCACACGCTCCTCGC",
            "IR_VDJ_1_junction": "TGTGTGAAAGGGGGTCAGATTTTCGATTTTTCGAGTGGTTATTATTTTGACTACTGG",
            "IR_VJ_1_sequence_alignment": "GAAATTGTGTTGACGCAGTCTCCAGACACCCTGTCTTTGTCTCCAGGGGAAAGAGCCACCCTCTCCTGCAGGGCCAGTCAGAGTGGTACC...............AGCAACTACTTAGCCTGGTACCAGCAGAAACCTGGCCAGCCTCCCAGACTCCTCATCTATGGTGCA.....................TCCAGCAGGGCCTCTGGCATCGCA...GACAGGTTCAGTGCCAGTGGA......TCTGGGACAGACTTCACTCTCACCATCAGCAGACTGGAGCCTGAAGATTTTGCAGTGTATTACTGTCAGCAGTATGGTAGTTTACCGCTCACTTTCGGCGGAGGGACCAAGGTAGATATCAAAC",
            "IR_VJ_1_germline_alignment": "GAAATTGTGTTGACGCTGTCTCCAGACACCCTGTCGGTGTCTCCAGGCCAAAGAGCCACCCTCTCCTGCAGGGCCAGTCAGAGTAATACC...............TCGAACTACTTAGCCTGGTACCAGCAGAAACCTGGCCAGCCTGGGAGACTAATCATCTATCCAGCA.....................AACAGCAGGGCCTCTGGCATCGCA...GACAGGTTCAGTGCCAGTGGA......TCTGGGACAGACTTCACTCTCAGGATCAGCAGACTTTAGCCTGAAGATTTTGCAGTGTATTAGTGTCAGCAGTATGGTAGCCCACCGCTCAGGTTCGGCGGAGGGACCGGGGTAGTAATCAAAG",
            "IR_VJ_1_junction": "TGTCAGCAGTATGGTAGTTTACCGCTCACTTTC",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AAGCCGCAGCGATGAC-MH9179822": {
            # 9
            # No germline alignment
            "IR_VDJ_1_sequence_alignment": "CAGGTGCAGCTGCAGCAGTCGGGCCCA...CGACTGGTGAAGCCTTCACAGACCCTGTCCCTCACCTGCACTGTCTCCGGTGACTCCATTAGC......AGTGAAAATTTCTACTGGAGCTGGGTCCGGCAGCCCGCCGGGGGGGGCCTGGAGTGGATTGGGCGCATCCATATCGCT.........GGGAGCACCGACTACAACCCCTCCTTCATC...AGTCGAGTCACCATATCACGAGACACGTCCAAGAGCCAGTTCTCCCTGAGGCTGCGTTCTGTGACCGCCACAGACACCGCCATATATTACTGTGCGACAGGTGGCTACAAATCAGATTTTGACCACTGGGGTCAGGGAATCGATGTCACCGTCTCCTCAG",
            "IR_VDJ_1_germline_alignment": None,
            "IR_VDJ_1_junction": "TGTGCGACAGGTGGCTACAAATCAGATTTTGACCACTGG",
            "IR_VJ_1_sequence_alignment": "GACATCGTGATGACCCAGTCTCCAGACTGCCTGGCTGTGTCTCTGGGCGAGAGGGCCGCCATCAACTGCAAGTCCAGCCAGAATATTGTGGCCAGCCCCGACAACAAGAACTGCTTGGCTTGGTTCCAGCAAAAACCAGGGCAGCCTCCTAAATTACTCATTTACCGGGCG.....................TCTACCCGGGCATCCGGGGTCCCT...GACCGGTTCAGTGGCAGCGGG......TCTGGGTCAGATTTTACTCTCACCATCAGCAACCTGCAGGCAGAAGATGTGGCAGTTTATTTCTGTCAACAATATTTTACTACTCCGCTCACCTTCGGCGGCGGGACCAGGGTGGAGATCAAAC",
            "IR_VJ_1_germline_alignment": None,
            "IR_VJ_1_junction": "TGTCAACAATATTTTACTACTCCGCTCACCTTC",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
        "AAGCCGCGTCAGATAA-MH9179822": {
            # 10
            # No sequence_alignment
            "IR_VDJ_1_sequence_alignment": None,
            "IR_VDJ_1_germline_alignment": "CAGCTGCAGCTGCAGGAGTCGGGCCCA...GGACTGGTGAAGCCCTCGGAGACCCTGTCCCTCACCTGCAGTGTCTCTGGTGGCTCTATCAGT......AGTAATAGTTATTACTGGGGCTGGATCCGCCAGCCCCCAGGGAAGAGCCTGGAGTGGATTGGGAGTATCCATTATAGT.........GGGAGCACCAACTACAACCCGTCCCTCAAG...AGTCGAGTCACCATATCCGTAGACACGTCCAAGAACCAGTTCTCCCTGAAGCTGAGCTCTGTGACCGCCGCAGACACGGCTGTGTATTACTGTGCGAGACGTGGCAATTATTATGATAGAAGTGGTTATGGGCTTGAGAACTTTGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
            "IR_VDJ_1_junction": "TGTGCGAGACGTGGCAATTATTATGATAGAAGTGGTTATGGGCTTGAGAACTTTGACTACTGG",
            "IR_VJ_1_sequence_alignment": None,
            "IR_VJ_1_germline_alignment": "GAAATTGTGTTGACGCAGTCTCCAGGCACCCTGTCTTTGTCTCCAGGGGAAAGAGCCACCCTCTCCTGCAGGGCCAGTCAGACTATTATC...............GACAGCTACTTAGCCTGGTACCAGCAGAAACCTGGCCAGGCTCCCAGGCTCCTCATCTATGATGCA.....................TCCAGCAGGGCCACTGGCATCCCA...GACAGGTTCAGTGGCAGTGGG......TCTGGGACAGACTTCACTCTCACCATCAGCAGACTGGAGGCTGAAGATTTTGCAGTGTATTACTGTCAGCACTATGGTAGCTCACCTCCATACACTTTTGGCCAGGGGACCAAGGTGGAGATCAAAC",
            "IR_VJ_1_junction": "TGTCAGCACTATGGTAGCTCACCTCCATACACTTTT",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    return _make_adata(obs, request.param)


@pytest.fixture(params=[False, True], ids=["AnnData", "MuData"])
def adata_not_aligned(request):
    obs = {
        "AACTCAGTCCTTTACA-MH9179822": {
            # 5
            # no mutations, but sequence alignment not IMGT-gapped
            "IR_VDJ_1_sequence_alignment": "GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTAAAGCCTGGGGGGTCCCTTAGACTCTCCTGTGCAGCCTCTGGATTCACTTTCAGTAACGCCTGGATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTTGGCCGTATTAAAAGCAAAACTGATGGTGGGACAACAGACTACGCTGCACCCGTGAAAGGCAGATTCACCATCTCAAGAGATGATTCAAAAAACACGCTGTATCTGCAAATGAACAGCCTGAAAACCGAGGACACAGCCGTGTATTACTGTACCACAGGAATTGTAGTAGTACCAGCTGCTATCCAATATTACTACTACTACTACGGTATGGACGTCTGGGGCCAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_germline_alignment": "GAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCTTGGTAAAGCCTGGGGGGTCCCTTAGACTCTCCTGTGCAGCCTCTGGATTCACTTTC............AGTAACGCCTGGATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTTGGCCGTATTAAAAGCAAAACTGATGGTGGGACAACAGACTACGCTGCACCCGTGAAA...GGCAGATTCACCATCTCAAGAGATGATTCAAAAAACACGCTGTATCTGCAAATGAACAGCCTGAAAACCGAGGACACAGCCGTGTATTACTGTACCACAGGAATTGTAGTAGTACCAGCTGCTATCCAATATTACTACTACTACTACGGTATGGACGTCTGGGGCCAAGGGACCACGGTCACCGTCTCCTCA",
            "IR_VDJ_1_junction": "TGTACCACAGGAATTGTAGTAGTACCAGCTGCTATCCAATATTACTACTACTACTACGGTATGGACGTCTGG",
            "IR_VJ_1_sequence_alignment": "GATATTGTGATGACTCAGTCTCCACTCTCCCTGCCCGTCACCCCTGGAGAGCCGGCCTCCATCTCCTGCAGGTCTAGTCAGAGCCTCCTGCATAGTAATGGATACAACTATTTGGATTGGTACCTGCAGAAGCCAGGGCAGTCTCCACAGCTCCTGATCTATTTGGGTTCTAATCGGGCCTCCGGGGTCCCTGACAGGTTCAGTGGCAGTGGATCAGGCACAGATTTTACACTGAAAATCAGCAGAGTGGAGGCTGAGGATGTTGGGGTTTATTACTGCATGCAAGCTCTACAAACTCCTCGAACTTTTGGCCAGGGGACCAAGCTGGAGATCAAAC",
            "IR_VJ_1_germline_alignment": "GATATTGTGATGACTCAGTCTCCACTCTCCCTGCCCGTCACCCCTGGAGAGCCGGCCTCCATCTCCTGCAGGTCTAGTCAGAGCCTCCTGCATAGT...AATGGATACAACTATTTGGATTGGTACCTGCAGAAGCCAGGGCAGTCTCCACAGCTCCTGATCTATTTGGGT.....................TCTAATCGGGCCTCCGGGGTCCCT...GACAGGTTCAGTGGCAGTGGA......TCAGGCACAGATTTTACACTGAAAATCAGCAGAGTGGAGGCTGAGGATGTTGGGGTTTATTACTGCATGCAAGCTCTACAAACTCCTCGAACTTTTGGCCAGGGGACCAAGCTGGAGATCAAAC",
            "IR_VJ_1_junction": "TGCATGCAAGCTCTACAAACTCCTCGAACTTTT",
            "IR_VJ_1_locus": "IGK",
            "IR_VDJ_1_locus": "IGH",
            "sample": "MH9179822",
        },
    }
    obs = pd.DataFrame.from_dict(obs, orient="index")
    return _make_adata(obs, request.param)
