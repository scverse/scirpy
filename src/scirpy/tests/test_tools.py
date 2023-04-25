# pylama:ignore=W0611,W0404
import itertools

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import scanpy as sc
from mudata import MuData

import scirpy as ir
from scirpy.util import DataHandler

from .fixtures import (  # NOQA
    adata_clonotype,
    adata_clonotype_network,
    adata_conn,
    adata_define_clonotype_clusters,
    adata_diversity,
    adata_tra,
)
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
            "AAA" if x != "nan" else "nan"
            for x in obs[f"IR_{chain}_{chain_number}_locus"]
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
                "ambiguous",
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
        adata, groupby="group", target_col="clone_id", clip_at=2, inplace=False
    )
    npt.assert_equal(
        res, np.array([">= 2"] * 3 + ["nan"] * 2 + ["1"] * 3 + [">= 2"] * 2)
    )

    # check without group
    res = ir.tl._clonal_expansion._clip_and_count(
        adata, target_col="clone_id", clip_at=5, inplace=False
    )
    npt.assert_equal(
        res, np.array(["4"] * 3 + ["nan"] * 2 + ["4"] + ["1"] * 2 + ["2"] * 2)
    )

    # check if target_col works
    params = DataHandler.default(adata)
    params.adata.obs["new_col"] = params.adata.obs["clone_id"]
    params.adata.obs.drop("clone_id", axis="columns", inplace=True)

    ir.tl._clonal_expansion._clip_and_count(
        adata,
        groupby="group",
        target_col="new_col",
        clip_at=2,
    )
    npt.assert_equal(
        params.adata.obs["new_col_clipped_count"],
        np.array([">= 2"] * 3 + ["nan"] * 2 + ["1"] * 3 + [">= 2"] * 2),
    )

    # check if it raises value error if target_col does not exist
    with pytest.raises(ValueError):
        ir.tl._clonal_expansion._clip_and_count(
            adata,
            groupby="group",
            target_col="clone_id",
            clip_at=2,
            fraction=False,
        )


@pytest.mark.parametrize(
    "expanded_in,expected",
    [
        ("group", [">= 2"] * 3 + ["nan"] * 2 + ["1"] * 3 + [">= 2"] * 2),
        (None, [">= 2"] * 3 + ["nan"] * 2 + [">= 2"] + ["1"] * 2 + [">= 2"] * 2),
    ],
)
def test_clonal_expansion(adata_clonotype, expanded_in, expected):
    res = ir.tl.clonal_expansion(
        adata_clonotype, expanded_in=expanded_in, clip_at=2, inplace=False
    )
    npt.assert_equal(res, np.array(expected))


def test_clonal_expansion_summary(adata_clonotype):
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clone_id", clip_at=2, normalize=True
    )
    pdt.assert_frame_equal(
        res,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2 / 5], ">= 2": [1.0, 3 / 5]}
        ).set_index("group"),
        check_names=False,
        check_index_type=False,
        check_categorical=False,
    )

    # test the `expanded_in` parameter.
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clone_id",
        clip_at=2,
        normalize=True,
        expanded_in="group",
    )
    pdt.assert_frame_equal(
        res,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 3 / 5], ">= 2": [1.0, 2 / 5]}
        ).set_index("group"),
        check_names=False,
        check_index_type=False,
        check_categorical=False,
    )

    # test the `summarize_by` parameter.
    res = ir.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clone_id",
        clip_at=2,
        normalize=True,
        summarize_by="clone_id",
    )
    pdt.assert_frame_equal(
        res,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2 / 4], ">= 2": [1.0, 2 / 4]}
        ).set_index("group"),
        check_names=False,
        check_index_type=False,
        check_categorical=False,
    )

    res_counts = ir.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clone_id", clip_at=2, normalize=False
    )
    print(res_counts)
    pdt.assert_frame_equal(
        res_counts,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2], ">= 2": [3, 3]}
        ).set_index("group"),
        check_names=False,
        check_dtype=False,
        check_index_type=False,
        check_categorical=False,
    )


@pytest.mark.extra
def test_alpha_diversity(adata_diversity):
    # normalized_shannon_entropy by default
    res = ir.tl.alpha_diversity(
        adata_diversity, groupby="group", target_col="clonotype_", inplace=False
    )
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

    ir.tl.alpha_diversity(
        adata_diversity, groupby="group", target_col="clonotype_", inplace=True
    )
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
        adata_diversity.obs[
            mdata_modifier + "normalized_shannon_entropy_clonotype_"
        ].values,
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
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
        ],
        columns=["cell_id", "group", "clone_id"],
    ).set_index("cell_id")
    adata = _make_adata(obs)

    # Check counts
    res = ir.tl.group_abundance(
        adata, groupby="clone_id", target_col="group", fraction=False
    )
    expected_count = pd.DataFrame.from_dict(
        {
            "ct1": {"A": 3.0, "B": 1.0},
            "ct2": {"A": 0.0, "B": 1.0},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_count.values)

    # Check fractions
    res = ir.tl.group_abundance(
        adata, groupby="clone_id", target_col="group", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {
            "ct1": {"A": 0.75, "B": 0.25},
            "ct2": {"A": 0.0, "B": 1.0},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)

    # Check swapped
    res = ir.tl.group_abundance(
        adata, groupby="group", target_col="clone_id", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {
            "A": {"ct1": 1.0, "ct2": 0.0},
            "B": {"ct1": 0.5, "ct2": 0.5},
        },
        orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)


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

    res1 = ir.tl.spectratype(
        adata_tra, chain="VJ_1", target_col="sample", fraction=False
    )
    res2 = ir.tl.spectratype(
        adata_tra, chain=["VJ_1"], target_col="sample", fraction=False
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
    res = ir.tl.spectratype(
        adata_tra, chain="VJ_1", target_col="sample", fraction="sample"
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
