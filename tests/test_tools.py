import pandas as pd
import sctcrpy as st
from scanpy import AnnData
import pytest
import numpy.testing as npt
import pandas.testing as pdt
import numpy as np
from sctcrpy._util import _get_from_uns


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


def test_chain_pairing():
    obs = pd.DataFrame.from_records(
        [
            ["False", "nan", "nan", "nan", "nan", "nan"],
            ["True", "True", "AAAA", "BBBB", "CCCC", "DDDD"],
            ["True", "False", "AAAA", "BBBB", "CCCC", "DDDD"],
            ["True", "nan", "AAAA", "nan", "nan", "nan"],
            ["True", "False", "AAAA", "nan", "CCCC", "nan"],
            ["True", "False", "AAAA", "BBBB", "nan", "nan"],
            ["True", "False", "AAAA", "BBBB", "CCCC", "nan"],
            ["True", "False", "nan", "nan", "CCCC", "nan"],
            ["True", "False", "nan", "nan", "CCCC", "DDDD"],
            ["True", "False", "AAAA", "nan", "CCCC", "DDDD"],
        ],
        columns=[
            "has_tcr",
            "multi_chain",
            "TRA_1_cdr3",
            "TRA_2_cdr3",
            "TRB_1_cdr3",
            "TRB_2_cdr3",
        ],
    )
    adata = AnnData(obs=obs)
    res = st.tl.chain_pairing(adata, inplace=False)
    npt.assert_equal(
        res,
        [
            "No TCR",
            "Multichain",
            "Two full chains",
            "Orphan alpha",
            "Single pair",
            "Orphan alpha",
            "Extra alpha",
            "Orphan beta",
            "Orphan beta",
            "Extra beta",
        ],
    )


def test_clip_and_count_clonotypes():
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

    res = st.tl.clip_and_count(
        adata, groupby="group", target_col="clonotype", clip_at=2, fraction=False
    )
    assert res == {"A": {"1": 0, ">= 2": 1}, "B": {"1": 3, ">= 2": 1}}

    res_frac = st.tl.clip_and_count(
        adata, groupby="group", target_col="clonotype", clip_at=2, fraction=True
    )
    assert res_frac == {"A": {"1": 0, ">= 2": 1.0}, "B": {"1": 0.75, ">= 2": 0.25}}

    # check if target_col works
    adata.obs["new_col"] = adata.obs["clonotype"]
    adata.obs.drop("clonotype", axis="columns", inplace=True)

    # check if it raises value error if target_col does not exist
    with pytest.raises(ValueError):
        st.tl.clip_and_count(
            adata, groupby="group", target_col="clonotype", clip_at=2, fraction=False,
        )


def test_clip_and_count_convergence(adata_tra):
    # Check counts
    res = st.tl.clip_and_count(
        adata_tra, target_col="TRA_1_cdr3", groupby="sample", fraction=False,
    )
    assert res == {
        1: {"1": 1, "2": 2, ">= 3": 0},
        3: {"1": 3, "2": 1, ">= 3": 0},
        5: {"1": 2, "2": 1, ">= 3": 0},
    }

    # Check fractions
    res = st.tl.clip_and_count(adata_tra, target_col="TRA_1_cdr3", groupby="sample")
    assert res == {
        1: {"1": 0.3333333333333333, "2": 0.6666666666666666, ">= 3": 0.0},
        3: {"1": 0.75, "2": 0.25, ">= 3": 0.0},
        5: {"1": 0.6666666666666666, "2": 0.3333333333333333, ">= 3": 0.0},
    }


def test_define_clonotypes():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "AAAA", "nan", "nan", "nan"],
            ["cell2", "nan", "nan", "nan", "nan"],
            ["cell3", "AAAA", "nan", "nan", "nan"],
            ["cell4", "AAAA", "BBBB", "nan", "nan"],
            ["cell5", "nan", "nan", "CCCC", "DDDD"],
        ],
        columns=["cell_id", "TRA_1_cdr3", "TRA_2_cdr3", "TRB_1_cdr3", "TRB_2_cdr3"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    res = st.tl.define_clonotypes(adata, inplace=False)
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res,
        ["clonotype_1", np.nan, "clonotype_1", "clonotype_0", "clonotype_2"],
    )

    res_primary_only = st.tl.define_clonotypes(
        adata, flavor="primary_only", inplace=False
    )
    npt.assert_equal(
        # order is by alphabet: BBBB < nan
        # we don't care about the order of numbers, so this is ok.
        res_primary_only,
        ["clonotype_0", np.nan, "clonotype_0", "clonotype_0", "clonotype_1"],
    )

    # test inplace
    st.tl.define_clonotypes(adata, key_added="clonotype_")
    npt.assert_equal(res, adata.obs["clonotype_"].values)


def test_alpha_diversity():
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

    res = st.tl.alpha_diversity(adata, groupby="group", target_col="clonotype_")
    assert res.to_dict(orient="index") == {"A": {0: 0.0}, "B": {0: 2.0}}


def test_group_abundance():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
        ],
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    # Check counts
    res = st.tl.group_abundance(adata, groupby="group", fraction=False)
    expected_count = pd.DataFrame.from_dict(
        {"ct1": {"A": 3.0, "B": 1.0}, "ct2": {"A": 0.0, "B": 1.0},}, orient="index",
    )
    npt.assert_equal(res.values, expected_count.values)

    # Check fractions
    res = st.tl.group_abundance(adata, groupby="group", fraction=True)
    expected_frac = pd.DataFrame.from_dict(
        {"ct1": {"A": 1.0, "B": 0.5}, "ct2": {"A": 0.0, "B": 0.5},}, orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)

    # Check swapped
    res = st.tl.group_abundance(
        adata, groupby="clonotype", target_col="group", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {"B": {"ct1": 0.25, "ct2": 1.0}, "A": {"ct1": 0.75, "ct2": 0.0}},
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)


def test_spectratype(adata_tra):
    # Check numbers
    res1 = st.tl.spectratype(
        adata_tra, target_col="TRA_1_cdr3_len", groupby="sample", fraction=False,
    )
    res2 = st.tl.spectratype(
        adata_tra, target_col=("TRA_1_cdr3_len",), groupby="sample", fraction=False,
    )
    expected_count = pd.DataFrame.from_dict(
        {
            0: {1: 0.0, 3: 0.0, 5: 0.0},
            1: {1: 0.0, 3: 0.0, 5: 0.0},
            2: {1: 0.0, 3: 0.0, 5: 0.0},
            3: {1: 0.0, 3: 0.0, 5: 0.0},
            4: {1: 0.0, 3: 0.0, 5: 0.0},
            5: {1: 0.0, 3: 0.0, 5: 0.0},
            6: {1: 0.0, 3: 0.0, 5: 0.0},
            7: {1: 0.0, 3: 0.0, 5: 0.0},
            8: {1: 0.0, 3: 0.0, 5: 0.0},
            9: {1: 0.0, 3: 0.0, 5: 0.0},
            10: {1: 0.0, 3: 0.0, 5: 0.0},
            11: {1: 0.0, 3: 0.0, 5: 0.0},
            12: {1: 1.0, 3: 2.0, 5: 0.0},
            13: {1: 2.0, 3: 0.0, 5: 0.0},
            14: {1: 0.0, 3: 2.0, 5: 1.0},
            15: {1: 2.0, 3: 1.0, 5: 1.0},
            16: {1: 0.0, 3: 0.0, 5: 0.0},
            17: {1: 0.0, 3: 0.0, 5: 2.0},
        },
        orient="index",
    )
    npt.assert_equal(res1.values, expected_count.values)
    npt.assert_equal(res2.values, expected_count.values)

    # Check fractions
    res = st.tl.spectratype(
        adata_tra, target_col="TRA_1_cdr3_len", groupby="sample", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {
            0: {1: 0.0, 3: 0.0, 5: 0.0},
            1: {1: 0.0, 3: 0.0, 5: 0.0},
            2: {1: 0.0, 3: 0.0, 5: 0.0},
            3: {1: 0.0, 3: 0.0, 5: 0.0},
            4: {1: 0.0, 3: 0.0, 5: 0.0},
            5: {1: 0.0, 3: 0.0, 5: 0.0},
            6: {1: 0.0, 3: 0.0, 5: 0.0},
            7: {1: 0.0, 3: 0.0, 5: 0.0},
            8: {1: 0.0, 3: 0.0, 5: 0.0},
            9: {1: 0.0, 3: 0.0, 5: 0.0},
            10: {1: 0.0, 3: 0.0, 5: 0.0},
            11: {1: 0.0, 3: 0.0, 5: 0.0},
            12: {1: 0.2, 3: 0.4, 5: 0.0},
            13: {1: 0.4, 3: 0.0, 5: 0.0},
            14: {1: 0.0, 3: 0.4, 5: 0.25},
            15: {1: 0.4, 3: 0.2, 5: 0.25},
            16: {1: 0.0, 3: 0.0, 5: 0.0},
            17: {1: 0.0, 3: 0.0, 5: 0.5},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)
