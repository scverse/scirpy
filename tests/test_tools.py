# pylama:ignore=W0611,W0404
import pandas as pd
import scirpy as st
from scanpy import AnnData
import pytest
import numpy.testing as npt
import pandas.testing as pdt
import numpy as np
from scirpy.util import _get_from_uns
from .fixtures import adata_clonotype, adata_tra, adata_vdj, adata_diversity


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


def test_clip_and_count_clonotypes(adata_clonotype):
    adata = adata_clonotype

    res = st._tools._clonal_expansion._clip_and_count(
        adata, groupby="group", target_col="clonotype", clip_at=2, inplace=False
    )
    npt.assert_equal(res, np.array([">= 2"] * 3 + ["1"] * 4 + [">= 2"] * 2))

    # check without group
    res = st._tools._clonal_expansion._clip_and_count(
        adata, target_col="clonotype", clip_at=5, inplace=False
    )
    npt.assert_equal(res, np.array(["4"] * 3 + ["1"] + ["4"] + ["1"] * 2 + ["2"] * 2))

    # check if target_col works
    adata.obs["new_col"] = adata.obs["clonotype"]
    adata.obs.drop("clonotype", axis="columns", inplace=True)

    st._tools._clonal_expansion._clip_and_count(
        adata, groupby="group", target_col="new_col", clip_at=2,
    )
    npt.assert_equal(
        adata.obs["new_col_clipped_count"],
        np.array([">= 2"] * 3 + ["1"] * 4 + [">= 2"] * 2),
    )

    # check if it raises value error if target_col does not exist
    with pytest.raises(ValueError):
        st._tools._clonal_expansion._clip_and_count(
            adata, groupby="group", target_col="clonotype", clip_at=2, fraction=False,
        )


def test_clonal_expansion(adata_clonotype):
    res = st.tl.clonal_expansion(
        adata_clonotype, expanded_in="group", clip_at=2, inplace=False
    )
    npt.assert_equal(res, np.array([">= 2"] * 3 + ["1"] * 4 + [">= 2"] * 2))

    res = st.tl.clonal_expansion(adata_clonotype, clip_at=2, inplace=False)
    npt.assert_equal(
        res, np.array([">= 2"] * 3 + ["1"] + [">= 2"] + ["1"] * 2 + [">= 2"] * 2)
    )


def test_clonal_expansion_summary(adata_clonotype):
    res = st.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clonotype", clip_at=2, normalize=True
    )
    pdt.assert_frame_equal(
        res,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2 / 5], ">= 2": [1.0, 3 / 5]}
        ).set_index("group"),
        check_names=False,
    )

    # test the `expanded_in` parameter.
    res = st.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clonotype",
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
    )

    # test the `summarize_by` parameter.
    res = st.tl.summarize_clonal_expansion(
        adata_clonotype,
        "group",
        target_col="clonotype",
        clip_at=2,
        normalize=True,
        summarize_by="clonotype",
    )
    pdt.assert_frame_equal(
        res,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2 / 4], ">= 2": [1.0, 2 / 4]}
        ).set_index("group"),
        check_names=False,
    )

    res_counts = st.tl.summarize_clonal_expansion(
        adata_clonotype, "group", target_col="clonotype", clip_at=2, normalize=False
    )
    print(res_counts)
    pdt.assert_frame_equal(
        res_counts,
        pd.DataFrame.from_dict(
            {"group": ["A", "B"], "1": [0, 2], ">= 2": [3, 3]}
        ).set_index("group"),
        check_names=False,
        check_dtype=False,
    )


def test_alpha_diversity(adata_diversity):
    res = st.tl.alpha_diversity(
        adata_diversity, groupby="group", target_col="clonotype_", inplace=False
    )
    assert res.to_dict(orient="index") == {"A": {0: 0.0}, "B": {0: 1.0}}

    st.tl.alpha_diversity(
        adata_diversity, groupby="group", target_col="clonotype_", inplace=True
    )
    npt.assert_equal(
        adata_diversity.obs["alpha_diversity_clonotype_"].values,
        np.array([0.0] * 4 + [1.0] * 4),
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
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)

    # Check counts
    res = st.tl.group_abundance(
        adata, groupby="clonotype", target_col="group", fraction=False
    )
    expected_count = pd.DataFrame.from_dict(
        {"ct1": {"A": 3.0, "B": 1.0}, "ct2": {"A": 0.0, "B": 1.0},}, orient="index",
    )
    npt.assert_equal(res.values, expected_count.values)

    # Check fractions
    res = st.tl.group_abundance(
        adata, groupby="clonotype", target_col="group", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {"ct1": {"A": 0.75, "B": 0.25}, "ct2": {"A": 0.0, "B": 1.0},}, orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)

    # Check swapped
    res = st.tl.group_abundance(
        adata, groupby="group", target_col="clonotype", fraction=True
    )
    expected_frac = pd.DataFrame.from_dict(
        {"A": {"ct1": 1.0, "ct2": 0.0}, "B": {"ct1": 0.5, "ct2": 0.5},}, orient="index",
    )
    npt.assert_equal(res.values, expected_frac.values)


def test_spectratype(adata_tra):
    # Check numbers
    res1 = st.tl.spectratype(
        adata_tra, groupby="TRA_1_cdr3", target_col="sample", fraction=False,
    )
    res2 = st.tl.spectratype(
        adata_tra, groupby=("TRA_1_cdr3",), target_col="sample", fraction=False,
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
        adata_tra, groupby="TRA_1_cdr3", target_col="sample", fraction="sample"
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


def test_repertoire_overlap(adata_tra):
    res, d, l = st.tl.repertoire_overlap(adata_tra, "sample", inplace=False)
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


def test_clonotype_imbalance(adata_tra):
    freq, stat = st.tl.clonotype_imbalance(
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
                "clonotype": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            1: {
                "clonotype": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            2: {
                "clonotype": "clonotype_330",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            3: {
                "clonotype": "clonotype_330",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            4: {
                "clonotype": "clonotype_330",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            5: {
                "clonotype": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            6: {
                "clonotype": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            7: {
                "clonotype": "clonotype_458",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            8: {
                "clonotype": "clonotype_458",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 0.0,
            },
            9: {
                "clonotype": "clonotype_458",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 1.0,
            },
            10: {
                "clonotype": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            11: {
                "clonotype": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            12: {
                "clonotype": "clonotype_739",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            13: {
                "clonotype": "clonotype_739",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            14: {
                "clonotype": "clonotype_739",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            15: {
                "clonotype": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            16: {
                "clonotype": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            17: {
                "clonotype": "clonotype_986",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            18: {
                "clonotype": "clonotype_986",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            19: {
                "clonotype": "clonotype_986",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            20: {
                "clonotype": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            21: {
                "clonotype": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            22: {
                "clonotype": "clonotype_987",
                None: "All",
                "chain_pairing": "Background",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
            23: {
                "clonotype": "clonotype_987",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "1",
                "Normalized abundance": 1.0,
            },
            24: {
                "clonotype": "clonotype_987",
                None: "All",
                "chain_pairing": "Single pair",
                "sample": "3",
                "Normalized abundance": 0.0,
            },
        },
        orient="index",
    )
    freq = freq.sort_values(by=["clonotype", "chain_pairing", "sample"])
    freq = freq.reset_index().iloc[:, 1:6]
    print(freq)
    print(stat)
    pdt.assert_frame_equal(freq, expected_freq, check_names=False, check_dtype=False)
    expected_stat = pd.DataFrame.from_dict(
        {
            0: {
                "clonotype": "clonotype_330",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": 0.5849625007211562,
            },
            1: {
                "clonotype": "clonotype_458",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": 0.5849625007211562,
            },
            2: {
                "clonotype": "clonotype_739",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
            3: {
                "clonotype": "clonotype_986",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
            4: {
                "clonotype": "clonotype_987",
                "pValue": 1.0,
                "logpValue": -0.0,
                "logFC": -0.4150374992788438,
            },
        },
        orient="index",
    )
    stat = stat.sort_values(by="clonotype")
    stat = stat.reset_index().iloc[:, 1:5]
    pdt.assert_frame_equal(stat, expected_stat, check_names=False, check_dtype=False)
