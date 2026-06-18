# pylama:ignore=W0611,W0404
import itertools

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import scanpy as sc
from mudata import MuData
from pytest import approx

import scirpy as ir
from scirpy.util import DataHandler

from . import TESTDATA
from .util import _make_adata


def test_chain_qc():
    obs = pd.DataFrame.from_records(
        [
            [False, "nan", "nan", "nan", "nan"],
            [True, "TRA", "TRB", "TRA", "TRB"],
            # multichain takes precedencee over ambiguous
            [True, "TRA", "IGH", "nan", "nan"],
            [False, "TRA", "TRB", "nan", "nan"],
            [False, "TRA", "TRB", "TRA", "nan"],
            [False, "TRA", "TRB", "nan", "TRB"],
            [False, "TRA", "TRB", "TRA", "TRB"],
            [False, "IGK", "IGH", "nan", "nan"],
            [False, "IGL", "IGH", "IGL", "IGH"],
            [False, "IGL", "IGH", "IGK", "IGH"],
            [False, "IGK", "IGH", "IGL", "IGH"],
            [False, "nan", "IGH", "nan", "IGH"],
            [False, "TRA", "TRB", "TRG", "TRB"],
            [False, "IGK", "TRB", "nan", "nan"],
            [False, "TRA", "nan", "nan", "nan"],
            [False, "IGL", "nan", "nan", "nan"],
            [False, "nan", "TRD", "nan", "nan"],
        ],
        columns=[
            "_multi_chain",
            "IR_VJ_1_locus",
            "IR_VDJ_1_locus",
            "IR_VJ_2_locus",
            "IR_VDJ_2_locus",
        ],
    )
    # fake chains
    for chain, chain_number in itertools.product(["VJ", "VDJ"], ["1", "2"]):
        obs[f"IR_{chain}_{chain_number}_junction_aa"] = [
            "AAA" if x != "nan" else "nan" for x in obs[f"IR_{chain}_{chain_number}_locus"]
        ]
    adata = _make_adata(obs)

    ir.tl.chain_qc(adata, key_added=("rec_type", "rec_subtype", "ch_pairing"))

    npt.assert_equal(
        adata.obs["rec_type"],
        np.array(
            [
                "no IR",
                "multichain",
                "multichain",
                "TCR",
                "TCR",
                "TCR",
                "TCR",
                "BCR",
                "BCR",
                "BCR",
                "BCR",
                "BCR",
                "TCR",
                "ambiguous",
                "TCR",
                "BCR",
                "TCR",
            ]
        ),
    )
    npt.assert_equal(
        adata.obs["rec_subtype"],
        np.array(
            [
                "no IR",
                "multichain",
                #
                "multichain",
                "TRA+TRB",
                "TRA+TRB",
                "TRA+TRB",
                "TRA+TRB",
                "IGH+IGK",
                "IGH+IGL",
                "IGH+IGK/L",
                "IGH+IGK/L",
                "IGH",
                "ambiguous",
                "ambiguous",
                "TRA+TRB",
                "IGH+IGL",
                "TRG+TRD",
            ]
        ),
    )


def test_clip_and_count_clonotypes(adata_clonotype):
    adata = adata_clonotype

    res = ir.tl._clonal_expansion._clip_and_count(
        adata, groupby="group", target_col="clone_id", breakpoints=(1,), inplace=False
    )
    npt.assert_equal(res, np.array(["> 1"] * 3 + ["nan"] * 2 + ["<= 1"] * 3 + ["> 1"] * 2))

    # check without group
    res = ir.tl._clonal_expansion._clip_and_count(adata, target_col="clone_id", breakpoints=(1, 2, 4), inplace=False)
    npt.assert_equal(res, np.array(["<= 4"] * 3 + ["nan"] * 2 + ["<= 4"] + ["<= 1"] * 2 + ["<= 2"] * 2))

    # check if target_col works
    params = DataHandler.default(adata)
    params.adata.obs["new_col"] = params.adata.obs["clone_id"]
    params.adata.obs.drop("clone_id", axis="columns", inplace=True)

    ir.tl._clonal_expansion._clip_and_count(
        adata,
        groupby="group",
        target_col="new_col",
        breakpoints=(1,),
    )
    npt.assert_equal(
        params.adata.obs["new_col_clipped_count"],
        np.array(["> 1"] * 3 + ["nan"] * 2 + ["<= 1"] * 3 + ["> 1"] * 2),
    )


@pytest.mark.parametrize(
    "expanded_in,expected",
    [
        ("group", ["> 1"] * 3 + ["nan"] * 2 + ["<= 1"] * 3 + ["> 1"] * 2),
        (None, ["> 1"] * 3 + ["nan"] * 2 + ["> 1"] + ["<= 1"] * 2 + ["> 1"] * 2),
    ],
)
def test_clonal_expansion(adata_clonotype, expanded_in, expected):
    res = ir.tl.clonal_expansion(adata_clonotype, expanded_in=expanded_in, breakpoints=(1,), inplace=False)
    npt.assert_equal(res, np.array(expected))


def test_clonal_expansion_summary(adata_clonotype):
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clone_id", breakpoints=(1,), normalize=True
    )
    assert res.reset_index().to_dict(orient="list") == {
        "group": ["A", "B"],
        "<= 1": [0, approx(0.4)],
        "> 1": [1.0, approx(0.6)],
    }

    # test the `expanded_in` parameter.
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clone_id",
        clip_at=2,
        normalize=True,
        expanded_in="group",
    )
    assert res.reset_index().to_dict(orient="list") == {
        "group": ["A", "B"],
        "<= 1": [0, approx(0.6)],
        "> 1": [1.0, approx(0.4)],
    }

    # test the `summarize_by` parameter.
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clone_id",
        clip_at=2,
        normalize=True,
        summarize_by="clone_id",
    )
    assert res.reset_index().to_dict(orient="list") == {
        "group": ["A", "B"],
        "<= 1": [0, approx(0.5)],
        "> 1": [1.0, approx(0.5)],
    }

    res_counts = ir.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clone_id", clip_at=2, normalize=False
    )
    assert res_counts.reset_index().to_dict(orient="list") == {"group": ["A", "B"], "<= 1": [0, 2], "> 1": [3, 3]}


@pytest.mark.extra
def test_alpha_diversity(adata_diversity):
    # normalized_shannon_entropy by default
    res = ir.tl.alpha_diversity(adata_diversity, groupby="group", target_col="clonotype_", inplace=False)
    assert res.to_dict(orient="index") == {"A": {0: 0.0}, "B": {0: 1.0}}

    # D50
    res = ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric="D50",
        inplace=False,
    )
    assert res.to_dict(orient="index") == {"A": {0: 100.0}, "B": {0: 50.0}}

    # observed_otus from skbio.diversity.alpha that calculates the number of distinct OTUs.
    res = ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric="observed_otus",
        inplace=False,
    )
    assert res.to_dict(orient="index") == {"A": {0: 1}, "B": {0: 4}}

    # custom metric function simply returns the # of unique clonotypes,
    # same as the observed_otus function
    def metric_func(counts):
        return len(counts)

    res = ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric=metric_func,
        inplace=False,
    )
    assert res.to_dict(orient="index") == {"A": {0: 1}, "B": {0: 4}}

    ir.tl.alpha_diversity(adata_diversity, groupby="group", target_col="clonotype_", inplace=True)
    ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric="D50",
        inplace=True,
    )
    ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric="observed_otus",
        inplace=True,
    )

    ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric="observed_otus",
        inplace=True,
    )

    ir.tl.alpha_diversity(
        adata_diversity,
        groupby="group",
        target_col="clonotype_",
        metric=metric_func,
        inplace=True,
    )

    mdata_modifier = "airr:" if isinstance(adata_diversity, MuData) else ""
    npt.assert_equal(
        adata_diversity.obs[mdata_modifier + "normalized_shannon_entropy_clonotype_"].values,
        np.array([0.0] * 4 + [1.0] * 4),
    )
    npt.assert_equal(
        adata_diversity.obs[mdata_modifier + "D50_clonotype_"].values,
        np.array([100.0] * 4 + [50.0] * 4),
    )
    npt.assert_equal(
        adata_diversity.obs[mdata_modifier + "observed_otus_clonotype_"].values,
        np.array([1] * 4 + [4] * 4),
    )
    npt.assert_equal(
        adata_diversity.obs[mdata_modifier + "metric_func_clonotype_"].values,
        np.array([1] * 4 + [4] * 4),
    )


def test_group_abundance():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell4", "A", "NaN"],
            ["cell5", "B", "ct1"],
            ["cell6", "B", "ct2"],
        ],
        columns=["cell_id", "group", "clone_id"],
    ).set_index("cell_id")
    adata = _make_adata(obs)

    # Check counts
    res = ir.tl.group_abundance(adata, groupby="clone_id", target_col="group", fraction=False)
    expected_count = pd.DataFrame.from_dict(
        {
            "ct1": {"A": 3.0, "B": 1.0},
            "ct2": {"A": 0.0, "B": 1.0},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_count.values)

    # Check fractions
    res = ir.tl.group_abundance(adata, groupby="clone_id", target_col="group", fraction=True)
    expected_frac = pd.DataFrame.from_dict(
        {
            "ct1": {"A": 0.75, "B": 0.25},
            "ct2": {"A": 0.0, "B": 1.0},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)

    # Check swapped
    res = ir.tl.group_abundance(adata, groupby="group", target_col="clone_id", fraction=True)
    expected_frac = pd.DataFrame.from_dict(
        {
            "A": {"ct1": 1.0, "ct2": 0.0},
            "B": {"ct1": 0.5, "ct2": 0.5},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)


def test_group_abundance_non_overlapping_gex_airr():
    """Regression test for #641"""
    mdata = ir.io.read_h5mu(TESTDATA / "wu2020_200.h5mu")
    ir.tl.chain_qc(mdata)
    res = ir.tl.group_abundance(mdata, groupby="airr:receptor_subtype")
    assert res.to_dict(orient="records") == [
        {"True": 149.0, "False": 0.0},
        {"True": 0.0, "False": 50.0},
        {"True": 1.0, "False": 0.0},
    ]


def test_spectratype(adata_tra):
    # Check numbers
    adata_tra.obs["IR_VJ_1_junction_aa"] = ir.get.airr(adata_tra, "junction_aa", "VJ_1")

    # Old API calls should raise a value error
    with pytest.raises(ValueError):
        res1 = ir.tl.spectratype(
            adata_tra,
            chain="IR_VJ_1_junction_aa",
            target_col="sample",
            fraction=False,
        )
    with pytest.raises(ValueError):
        res2 = ir.tl.spectratype(
            adata_tra,
            chain=("IR_VJ_1_junction_aa",),
            target_col="sample",
            fraction=False,
        )

    res1 = ir.tl.spectratype(adata_tra, chain="VJ_1", target_col="sample", fraction=False)
    res2 = ir.tl.spectratype(adata_tra, chain=["VJ_1"], target_col="sample", fraction=False)
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
            12: {1: 2.0, 3: 1.0, 5: 0.0},
            13: {1: 0.0, 3: 2.0, 5: 0.0},
            14: {1: 2.0, 3: 0.0, 5: 1.0},
            15: {1: 1.0, 3: 2.0, 5: 1.0},
            16: {1: 0.0, 3: 0.0, 5: 0.0},
            17: {1: 0.0, 3: 0.0, 5: 2.0},
        },
        orient="index",
    )
    npt.assert_equal(res1.values, expected_count.values)
    npt.assert_equal(res2.values, expected_count.values)

    # Check fractions
    res = ir.tl.spectratype(adata_tra, chain="VJ_1", target_col="sample", fraction="sample")
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
            12: {1: 0.4, 3: 0.2, 5: 0.0},
            13: {1: 0.0, 3: 0.4, 5: 0.0},
            14: {1: 0.4, 3: 0.0, 5: 0.25},
            15: {1: 0.2, 3: 0.4, 5: 0.25},
            16: {1: 0.0, 3: 0.0, 5: 0.0},
            17: {1: 0.0, 3: 0.0, 5: 0.5},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)


def test_repertoire_overlap(adata_tra):
    res, d, l = ir.tl.repertoire_overlap(adata_tra, "sample", inplace=False)
    expected_cnt = pd.DataFrame.from_dict(
        {
            1: {
                "clonotype_157": 0.0,
                "clonotype_158": 0.0,
                "clonotype_284": 0.0,
                "clonotype_330": 0.0,
                "clonotype_425": 0.0,
                "clonotype_430": 0.0,
                "clonotype_458": 0.0,
                "clonotype_459": 1.0,
                "clonotype_460": 0.0,
                "clonotype_592": 0.0,
                "clonotype_738": 1.0,
                "clonotype_739": 1.0,
                "clonotype_986": 1.0,
                "clonotype_987": 1.0,
            },
            3: {
                "clonotype_157": 0.0,
                "clonotype_158": 0.0,
                "clonotype_284": 0.0,
                "clonotype_330": 1.0,
                "clonotype_425": 1.0,
                "clonotype_430": 1.0,
                "clonotype_458": 1.0,
                "clonotype_459": 0.0,
                "clonotype_460": 1.0,
                "clonotype_592": 0.0,
                "clonotype_738": 0.0,
                "clonotype_739": 0.0,
                "clonotype_986": 0.0,
                "clonotype_987": 0.0,
            },
            5: {
                "clonotype_157": 1.0,
                "clonotype_158": 1.0,
                "clonotype_284": 1.0,
                "clonotype_330": 0.0,
                "clonotype_425": 0.0,
                "clonotype_430": 0.0,
                "clonotype_458": 0.0,
                "clonotype_459": 0.0,
                "clonotype_460": 0.0,
                "clonotype_592": 1.0,
                "clonotype_738": 0.0,
                "clonotype_739": 0.0,
                "clonotype_986": 0.0,
                "clonotype_987": 0.0,
            },
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_cnt.values)


@pytest.mark.extra
@pytest.mark.parametrize(
    "permutation_test,fdr_correction,expected_scores,expected_pvalues",
    [
        (
            "approx",
            True,
            {"0": -0.12697285625776547, "1": -0.22712493444992585},
            {"0": 0.7373836252618164, "1": 0.7373836252618164},
        ),
        (
            "approx",
            False,
            {"0": -0.12697285625776547, "1": -0.22712493444992585},
            {"0": 0.6417000000890838, "1": 0.7373836252618164},
        ),
        (
            "exact",
            False,
            {"0": -0.1302531239444782, "1": -0.22422553993839395},
            {"0": 0.8583, "1": 0.929},
        ),
    ],
)
def test_clonotype_modularity(
    adata_clonotype_network,
    permutation_test,
    fdr_correction,
    expected_scores,
    expected_pvalues,
):
    if isinstance(adata_clonotype_network, MuData):
        sc.pp.neighbors(adata_clonotype_network.mod["gex"])
    else:
        sc.pp.neighbors(adata_clonotype_network)
    scores, pvalues = ir.tl.clonotype_modularity(
        adata_clonotype_network,
        target_col="cc_aa_alignment",
        permutation_test=permutation_test,
        fdr_correction=fdr_correction,
        inplace=False,
    )  # type: ignore
    # print(scores)
    # print(pvalues)
    assert scores == pytest.approx(expected_scores, abs=0.02)
    assert pvalues == pytest.approx(expected_pvalues, abs=0.02)


def test_clonotype_imbalance(adata_tra):
    freq, stat = ir.tl.clonotype_imbalance(
        adata_tra[
            adata_tra.obs.index.isin(
                [
                    "AAGGTTCCACCCAGTG-1",
                    "ACTATCTAGGGCTTCC-1",
                    "CAGTAACAGGCATGTG-1",
                    "CCTTACGGTCATCCCT-1",
                    "AAACCTGAGATAGCAT-1",
                ]
            )
        ],
        replicate_col="sample",
        groupby="chain_pairing",
        case_label="Single pair",
        inplace=False,
    )
    expected_freq = pd.DataFrame.from_dict(
        {
            0: {
                "clone_id": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            1: {
                "clone_id": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            2: {
                "clone_id": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            3: {
                "clone_id": "clonotype_330",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            4: {
                "clone_id": "clonotype_330",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            5: {
                "clone_id": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            6: {
                "clone_id": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            7: {
                "clone_id": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            8: {
                "clone_id": "clonotype_458",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            9: {
                "clone_id": "clonotype_458",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            10: {
                "clone_id": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            11: {
                "clone_id": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            12: {
                "clone_id": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            13: {
                "clone_id": "clonotype_739",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            14: {
                "clone_id": "clonotype_739",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            15: {
                "clone_id": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            16: {
                "clone_id": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            17: {
                "clone_id": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            18: {
                "clone_id": "clonotype_986",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            19: {
                "clone_id": "clonotype_986",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            20: {
                "clone_id": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            21: {
                "clone_id": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            22: {
                "clone_id": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            23: {
                "clone_id": "clonotype_987",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            24: {
                "clone_id": "clonotype_987",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
        },
        orient="index",
    )
    freq = freq.sort_values(by=["clone_id", "chain_pairing", "sample"])
    freq = freq.reset_index().iloc[:, 1:6]
    print(freq)
    print(stat)
    pdt.assert_frame_equal(freq, expected_freq, check_names=False, check_dtype=False)
    expected_stat = pd.DataFrame.from_dict(
        {
            0: {
                "clone_id": "clonotype_330",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": 0.5849625007211562,
            },
            1: {
                "clone_id": "clonotype_458",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": 0.5849625007211562,
            },
            2: {
                "clone_id": "clonotype_739",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
            3: {
                "clone_id": "clonotype_986",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
            4: {
                "clone_id": "clonotype_987",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
        },
        orient="index",
    )
    stat = stat.sort_values(by="clone_id")
    stat = stat.reset_index().iloc[:, 1:5]
    pdt.assert_frame_equal(stat, expected_stat, check_names=False, check_dtype=False)


@pytest.mark.parametrize(
    "region_vars,expected",
    [
        [
            "mutation_count",
            pd.DataFrame.from_dict(
                {
                    "AAACGGGCACGACTCG-MH9179822": {
                        # 1
                        # no mutation
                        "VDJ_1_mutation_count": 0,
                        "VJ_1_mutation_count": 0,
                    },
                    "AACCATGAGAGCAATT-MH9179822": {
                        # 2
                        # no mutation, but germline cdr3 masked with 35 "N" in VDJ and 5 "N" in VJ
                        "VDJ_1_mutation_count": 0,
                        "VJ_1_mutation_count": 0,
                    },
                    "AACCATGCAGTCACTA-MH9179822": {
                        # 3
                        # no mutation, but sequence alignment poor sequence quality at beginning: 15 '.'
                        "VDJ_1_mutation_count": 0,
                        "VJ_1_mutation_count": 0,
                    },
                    "AACGTTGGTATAAACG-MH9179822": {
                        # 4
                        # no mutation, but gaps ('-') in sequence alignment: 3 in FWR1, 3 in FWR2 and 5 in FWR4
                        "VDJ_1_mutation_count": 11,
                        "VJ_1_mutation_count": 11,
                    },
                    "AACTCTTGTTTGGCGC-MH9179822": {
                        # 6
                        # few mutations: 1 in each subregion of sequence_alignment (= 7 in total)
                        "VDJ_1_mutation_count": 7,
                        "VJ_1_mutation_count": 7,
                    },
                    "AACTGGTCAATTGCTG-MH9179822": {
                        # 7
                        # some mutations: 3 in each subregion of germline alignment (= 21 in total)
                        "VDJ_1_mutation_count": 21,
                        "VJ_1_mutation_count": 21,
                    },
                    "AAGCCGCAGATATACG-MH9179822": {
                        # 8
                        # a lot mutation: 5 in each subregion of germline alignment (= 35 in total)
                        "VDJ_1_mutation_count": 35,
                        "VJ_1_mutation_count": 35,
                    },
                    "AAGCCGCAGCGATGAC-MH9179822": {
                        # 9
                        # No germline alignment
                        "VDJ_1_mutation_count": None,
                        "VJ_1_mutation_count": None,
                    },
                    "AAGCCGCGTCAGATAA-MH9179822": {
                        "VDJ_1_mutation_count": None,
                        "VJ_1_mutation_count": None,
                        # 10
                        # No sequence_alignment
                    },
                },
                orient="index",
            ),
        ],
        [
            "v_mutation_count",
            pd.DataFrame.from_dict(
                {
                    "AAACGGGCACGACTCG-MH9179822": {
                        # 1
                        # no mutation
                        "VDJ_1_v_mutation_count": 0,
                        "VJ_1_v_mutation_count": 0,
                    },
                    "AACCATGAGAGCAATT-MH9179822": {
                        # 2
                        # no mutation, but germline cdr3 masked with 35 "N" in VDJ and 5 "N" in VJ
                        "VDJ_1_v_mutation_count": 0,
                        "VJ_1_v_mutation_count": 0,
                    },
                    "AACCATGCAGTCACTA-MH9179822": {
                        # 3
                        # no mutation, but sequence alignment poor sequence quality at beginning: 15 '.'
                        "VDJ_1_v_mutation_count": 0,
                        "VJ_1_v_mutation_count": 0,
                    },
                    "AACGTTGGTATAAACG-MH9179822": {
                        # 4
                        # no mutation, but gaps ('-') in sequence alignment: 3 in FWR1, 3 in FWR2 and 5 in FWR4
                        "VDJ_1_v_mutation_count": 6,
                        "VJ_1_v_mutation_count": 6,
                    },
                    "AACTCTTGTTTGGCGC-MH9179822": {
                        # 6
                        # few mutations: 1 in each subregion of sequence_alignment (= 7 in total)
                        "VDJ_1_v_mutation_count": 5,
                        "VJ_1_v_mutation_count": 5,
                    },
                    "AACTGGTCAATTGCTG-MH9179822": {
                        # 7
                        # some mutations: 3 in each subregion of germline alignment (= 21 in total)
                        "VDJ_1_v_mutation_count": 15,
                        "VJ_1_v_mutation_count": 15,
                    },
                    "AAGCCGCAGATATACG-MH9179822": {
                        # 8
                        # a lot mutation: 5 in each subregion of germline alignment (= 35 in total)
                        "VDJ_1_v_mutation_count": 25,
                        "VJ_1_v_mutation_count": 25,
                    },
                    "AAGCCGCAGCGATGAC-MH9179822": {
                        # 9
                        # No germline alignment
                        "VDJ_1_v_mutation_count": None,
                        "VJ_1_v_mutation_count": None,
                    },
                    "AAGCCGCGTCAGATAA-MH9179822": {
                        # 10
                        # No sequence_alignment
                        "VDJ_1_v_mutation_count": None,
                        "VJ_1_v_mutation_count": None,
                    },
                },
                orient="index",
            ),
        ],
        [
            (
                "fwr1_mutation_count",
                "cdr1_mutation_count",
                "fwr2_mutation_count",
                "cdr2_mutation_count",
                "fwr3_mutation_count",
                "cdr3_mutation_count",
                "fwr4_mutation_count",
            ),
            pd.DataFrame.from_dict(
                {
                    "AAACGGGCACGACTCG-MH9179822": {
                        # 1
                        # no mutation
                        "VDJ_1_fwr1_mutation_count": 0,
                        "VDJ_1_cdr1_mutation_count": 0,
                        "VDJ_1_fwr2_mutation_count": 0,
                        "VDJ_1_cdr2_mutation_count": 0,
                        "VDJ_1_fwr3_mutation_count": 0,
                        "VDJ_1_cdr3_mutation_count": 0,
                        "VDJ_1_fwr4_mutation_count": 0,
                        "VJ_1_fwr1_mutation_count": 0,
                        "VJ_1_cdr1_mutation_count": 0,
                        "VJ_1_fwr2_mutation_count": 0,
                        "VJ_1_cdr2_mutation_count": 0,
                        "VJ_1_fwr3_mutation_count": 0,
                        "VJ_1_cdr3_mutation_count": 0,
                        "VJ_1_fwr4_mutation_count": 0,
                    },
                    "AACCATGAGAGCAATT-MH9179822": {
                        # 2
                        # no mutation, but germline cdr3 masked with 35 "N" in VDJ and 5 "N" in VJ
                        "VDJ_1_fwr1_mutation_count": 0,
                        "VDJ_1_cdr1_mutation_count": 0,
                        "VDJ_1_fwr2_mutation_count": 0,
                        "VDJ_1_cdr2_mutation_count": 0,
                        "VDJ_1_fwr3_mutation_count": 0,
                        "VDJ_1_cdr3_mutation_count": 0,
                        "VDJ_1_fwr4_mutation_count": 0,
                        "VJ_1_fwr1_mutation_count": 0,
                        "VJ_1_cdr1_mutation_count": 0,
                        "VJ_1_fwr2_mutation_count": 0,
                        "VJ_1_cdr2_mutation_count": 0,
                        "VJ_1_fwr3_mutation_count": 0,
                        "VJ_1_cdr3_mutation_count": 0,
                        "VJ_1_fwr4_mutation_count": 0,
                    },
                    "AACCATGCAGTCACTA-MH9179822": {
                        # 3
                        # no mutation, but sequence alignment poor sequence quality at beginning: 15 '.'
                        "VDJ_1_fwr1_mutation_count": 0,
                        "VDJ_1_cdr1_mutation_count": 0,
                        "VDJ_1_fwr2_mutation_count": 0,
                        "VDJ_1_cdr2_mutation_count": 0,
                        "VDJ_1_fwr3_mutation_count": 0,
                        "VDJ_1_cdr3_mutation_count": 0,
                        "VDJ_1_fwr4_mutation_count": 0,
                        "VJ_1_fwr1_mutation_count": 0,
                        "VJ_1_cdr1_mutation_count": 0,
                        "VJ_1_fwr2_mutation_count": 0,
                        "VJ_1_cdr2_mutation_count": 0,
                        "VJ_1_fwr3_mutation_count": 0,
                        "VJ_1_cdr3_mutation_count": 0,
                        "VJ_1_fwr4_mutation_count": 0,
                    },
                    "AACGTTGGTATAAACG-MH9179822": {
                        # 4
                        # no mutation, but gaps ('-') in sequence alignment: 3 in FWR1, 3 in FWR2 and 5 in FWR4
                        "VDJ_1_fwr1_mutation_count": 3,
                        "VDJ_1_cdr1_mutation_count": 0,
                        "VDJ_1_fwr2_mutation_count": 3,
                        "VDJ_1_cdr2_mutation_count": 0,
                        "VDJ_1_fwr3_mutation_count": 0,
                        "VDJ_1_cdr3_mutation_count": 0,
                        "VDJ_1_fwr4_mutation_count": 5,
                        "VJ_1_fwr1_mutation_count": 3,
                        "VJ_1_cdr1_mutation_count": 0,
                        "VJ_1_fwr2_mutation_count": 3,
                        "VJ_1_cdr2_mutation_count": 0,
                        "VJ_1_fwr3_mutation_count": 0,
                        "VJ_1_cdr3_mutation_count": 0,
                        "VJ_1_fwr4_mutation_count": 5,
                    },
                    "AACTCTTGTTTGGCGC-MH9179822": {
                        # 6
                        # few mutations: 1 in each subregion of sequence_alignment (= 7 in total)
                        "VDJ_1_fwr1_mutation_count": 1,
                        "VDJ_1_cdr1_mutation_count": 1,
                        "VDJ_1_fwr2_mutation_count": 1,
                        "VDJ_1_cdr2_mutation_count": 1,
                        "VDJ_1_fwr3_mutation_count": 1,
                        "VDJ_1_cdr3_mutation_count": 1,
                        "VDJ_1_fwr4_mutation_count": 1,
                        "VJ_1_fwr1_mutation_count": 1,
                        "VJ_1_cdr1_mutation_count": 1,
                        "VJ_1_fwr2_mutation_count": 1,
                        "VJ_1_cdr2_mutation_count": 1,
                        "VJ_1_fwr3_mutation_count": 1,
                        "VJ_1_cdr3_mutation_count": 1,
                        "VJ_1_fwr4_mutation_count": 1,
                    },
                    "AACTGGTCAATTGCTG-MH9179822": {
                        # 7
                        # some mutations: 3 in each subregion of germline alignment (= 21 in total)
                        "VDJ_1_fwr1_mutation_count": 3,
                        "VDJ_1_cdr1_mutation_count": 3,
                        "VDJ_1_fwr2_mutation_count": 3,
                        "VDJ_1_cdr2_mutation_count": 3,
                        "VDJ_1_fwr3_mutation_count": 3,
                        "VDJ_1_cdr3_mutation_count": 3,
                        "VDJ_1_fwr4_mutation_count": 3,
                        "VJ_1_fwr1_mutation_count": 3,
                        "VJ_1_cdr1_mutation_count": 3,
                        "VJ_1_fwr2_mutation_count": 3,
                        "VJ_1_cdr2_mutation_count": 3,
                        "VJ_1_fwr3_mutation_count": 3,
                        "VJ_1_cdr3_mutation_count": 3,
                        "VJ_1_fwr4_mutation_count": 3,
                    },
                    "AAGCCGCAGATATACG-MH9179822": {
                        # 8
                        # a lot mutation: 5 in each subregion of germline alignment (= 35 in total)
                        "VDJ_1_fwr1_mutation_count": 5,
                        "VDJ_1_cdr1_mutation_count": 5,
                        "VDJ_1_fwr2_mutation_count": 5,
                        "VDJ_1_cdr2_mutation_count": 5,
                        "VDJ_1_fwr3_mutation_count": 5,
                        "VDJ_1_cdr3_mutation_count": 5,
                        "VDJ_1_fwr4_mutation_count": 5,
                        "VJ_1_fwr1_mutation_count": 5,
                        "VJ_1_cdr1_mutation_count": 5,
                        "VJ_1_fwr2_mutation_count": 5,
                        "VJ_1_cdr2_mutation_count": 5,
                        "VJ_1_fwr3_mutation_count": 5,
                        "VJ_1_cdr3_mutation_count": 5,
                        "VJ_1_fwr4_mutation_count": 5,
                    },
                    "AAGCCGCAGCGATGAC-MH9179822": {
                        # 9
                        # No germline alignment
                        "VDJ_1_fwr1_mutation_count": None,
                        "VDJ_1_cdr1_mutation_count": None,
                        "VDJ_1_fwr2_mutation_count": None,
                        "VDJ_1_cdr2_mutation_count": None,
                        "VDJ_1_fwr3_mutation_count": None,
                        "VDJ_1_cdr3_mutation_count": None,
                        "VDJ_1_fwr4_mutation_count": None,
                        "VJ_1_fwr1_mutation_count": None,
                        "VJ_1_cdr1_mutation_count": None,
                        "VJ_1_fwr2_mutation_count": None,
                        "VJ_1_cdr2_mutation_count": None,
                        "VJ_1_fwr3_mutation_count": None,
                        "VJ_1_cdr3_mutation_count": None,
                        "VJ_1_fwr4_mutation_count": None,
                    },
                    "AAGCCGCGTCAGATAA-MH9179822": {
                        # 10
                        # No sequence_alignment
                        "VDJ_1_fwr1_mutation_count": None,
                        "VDJ_1_cdr1_mutation_count": None,
                        "VDJ_1_fwr2_mutation_count": None,
                        "VDJ_1_cdr2_mutation_count": None,
                        "VDJ_1_fwr3_mutation_count": None,
                        "VDJ_1_cdr3_mutation_count": None,
                        "VDJ_1_fwr4_mutation_count": None,
                        "VJ_1_fwr1_mutation_count": None,
                        "VJ_1_cdr1_mutation_count": None,
                        "VJ_1_fwr2_mutation_count": None,
                        "VJ_1_cdr2_mutation_count": None,
                        "VJ_1_fwr3_mutation_count": None,
                        "VJ_1_cdr3_mutation_count": None,
                        "VJ_1_fwr4_mutation_count": None,
                    },
                },
                orient="index",
            ),
        ],
    ],
)
def test_mutational_load(adata_mutation, region_vars, expected):
    ir.tl.mutational_load(
        adata_mutation,
        germline_key="germline_alignment",
    )
    pdt.assert_frame_equal(
        ir.get.airr(adata_mutation, region_vars, ["VDJ_1", "VJ_1"]),
        expected,
        check_names=False,
        check_dtype=False,
    )


def test_mutational_load_adata_not_aligned(adata_not_aligned):
    with npt.assert_raises(ValueError):
        ir.tl.mutational_load(adata_not_aligned, germline_key="germline_alignment")


def _adata_from_counts(counts: dict[str, list[int]], mudata: bool = False):
    """Build a minimal AnnData/MuData whose obs encodes the given per-group clonotype counts.

    ``counts`` maps a group label to a list of clonotype abundances. Each abundance
    is expanded into that many cells sharing one clonotype id, so that collapsing
    ``obs`` back by (group, clonotype) recovers the original counts exactly.
    """
    records = []
    for group, abundances in counts.items():
        for clone_idx, n in enumerate(abundances):
            for _ in range(n):
                records.append([f"{group}_ct{clone_idx}", group])
    obs = pd.DataFrame(records, columns=["clonotype_", "group"])
    obs.index = [f"cell{i}" for i in range(len(obs))]
    return _make_adata(obs, mudata)


# two well-sampled groups of different depth: comparable at a common coverage < 1
_HILL_COUNTS = {
    "A": [30, 20, 14, 10, 7, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1],
    "B": [90, 60, 42, 30, 22, 16, 12, 9, 7, 5, 4, 3, 2, 2, 1, 1, 1, 1],
}


@pytest.mark.extra
@pytest.mark.parametrize("mudata", [False, True], ids=["AnnData", "MuData"])
def test_hill_diversity_profile(mudata):
    import warnings

    import hillrep

    adata = _adata_from_counts(_HILL_COUNTS, mudata)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the happy path must not warn
        res = ir.tl.hill_diversity_profile(adata, groupby="group", target_col="clonotype_", q_min=0, q_max=2, q_step=1)

    # the adapter must reproduce hillrep's coverage-standardized estimate exactly
    ref = hillrep.compare(_HILL_COUNTS, level="coverage", q=[0.0, 1.0, 2.0], n_boot=0)
    expected = ref.pivot(index="order_q", columns="assemblage", values="qD")

    assert list(res.columns) == ["A", "B"]
    for q in (0.0, 1.0, 2.0):
        for grp in ("A", "B"):
            npt.assert_allclose(res.loc[q, grp], expected.loc[q, grp], rtol=1e-9)


@pytest.mark.extra
def test_hill_diversity_profile_warns_when_not_comparable():
    # group "A" is heavily undersampled with no doubletons: a fair comparison is not supported
    counts = {"A": [3, 1, 1, 1, 1], "B": [120, 60, 40, 30, 20, 12, 8, 5, 3, 2, 1, 1]}
    adata = _adata_from_counts(counts)
    with pytest.warns(UserWarning, match="cannot be compared"):
        ir.tl.hill_diversity_profile(adata, groupby="group", target_col="clonotype_")


@pytest.mark.extra
def test_convert_hill_table():
    adata = _adata_from_counts(_HILL_COUNTS)
    profile = ir.tl.hill_diversity_profile(adata, groupby="group", target_col="clonotype_", q_min=0, q_max=2, q_step=1)

    div = ir.tl.convert_hill_table(profile, convert_to="diversity")
    assert list(div.index) == ["Observed richness", "Shannon entropy", "Inverse Simpson", "Gini-Simpson"]
    npt.assert_allclose(div.loc["Observed richness"].to_numpy(dtype=float), profile.loc[0].to_numpy(dtype=float))
    npt.assert_allclose(div.loc["Shannon entropy"].to_numpy(dtype=float), np.log(profile.loc[1].to_numpy(dtype=float)))
    npt.assert_allclose(div.loc["Inverse Simpson"].to_numpy(dtype=float), profile.loc[2].to_numpy(dtype=float))
    npt.assert_allclose(div.loc["Gini-Simpson"].to_numpy(dtype=float), 1 - 1 / profile.loc[2].to_numpy(dtype=float))

    with pytest.raises(ValueError):
        ir.tl.convert_hill_table(profile, convert_to="not_a_mode")


def test_convert_hill_table_modes():
    # operates on a plain DataFrame, so it needs no optional dependency
    profile = pd.DataFrame({"A": [10.0, 6.0, 4.0], "B": [20.0, 12.0, 8.0]}, index=[0, 1, 2])

    div = ir.tl.convert_hill_table(profile, convert_to="diversity")
    npt.assert_allclose(div.loc["Shannon entropy"].to_numpy(dtype=float), np.log(profile.loc[1].to_numpy(dtype=float)))
    npt.assert_allclose(div.loc["Gini-Simpson"].to_numpy(dtype=float), 1 - 1 / profile.loc[2].to_numpy(dtype=float))

    ef = ir.tl.convert_hill_table(profile, convert_to="evenness_factor")
    npt.assert_allclose(ef.to_numpy(dtype=float), (profile / profile.loc[0]).to_numpy(dtype=float))

    rel = ir.tl.convert_hill_table(profile, convert_to="relative_evenness")
    npt.assert_allclose(rel.to_numpy(dtype=float), (np.log(profile) / np.log(profile.loc[0])).to_numpy(dtype=float))

    with pytest.raises(ValueError, match="Invalid"):
        ir.tl.convert_hill_table(profile, convert_to="nope")

    with pytest.raises(ValueError, match="missing diversity order"):
        ir.tl.convert_hill_table(profile.drop(index=2), convert_to="diversity")


def test_hill_diversity_profile_requires_hillrep(monkeypatch):
    import builtins

    from scirpy.tl import _diversity

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hillrep":
            raise ImportError("simulated missing hillrep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install hillrep"):
        _diversity._import_hillrep()
