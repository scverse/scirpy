from scirpy.io import read_10x_vdj, read_tracer, read_airr
from scirpy.util import _is_na, _is_false
import numpy as np
import pytest
import pandas.testing as pdt
from . import TESTDATA


@pytest.mark.parametrize(
    "path",
    [
        TESTDATA / "10x/vdj_nextgem_hs_pbmc3_t_filtered_contig_annotations.csv.gz",
        TESTDATA / "10x/sc5p_v2_hs_melanoma_10k_b_filtered_contig_annotations.csv.gz",
    ],
)
def test_read_10x_example(path):
    """Test that a full 10x CSV table can be imported without errors.

    Test-dataset from https://support.10xgenomics.com/single-cell-vdj/datasets/3.1.0/vdj_nextgem_hs_pbmc3
    and https://support.10xgenomics.com/single-cell-vdj/datasets/4.0.0/sc5p_v2_hs_melanoma_10k
    under CC-BY-4.0
    """
    anndata = read_10x_vdj(path)
    assert anndata.shape[0] > 0


def test_read_10x_csv():
    anndata = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    obs = anndata.obs
    assert obs.shape[0] == 5
    cell1 = obs.iloc[1, :]
    cell2 = obs.iloc[3, :]
    cell3 = obs.iloc[4, :]

    assert cell1.name == "AAACCTGAGTACGCCC-1"
    assert cell1["IR_VDJ_1_cdr3"] == "CASSLGPSTDTQYF"
    assert cell1["IR_VDJ_1_cdr3_nt"] == "TGTGCCAGCAGCTTGGGACCTAGCACAGATACGCAGTATTTT"
    assert _is_na(cell1["IR_VDJ_1_junction_ins"])
    assert cell1["IR_VDJ_1_expr"] == 55
    assert cell1["IR_VDJ_1_v_gene"] == "TRBV7-2"
    assert cell1["IR_VDJ_1_d_gene"] == "TRBD2"
    assert cell1["IR_VDJ_1_j_gene"] == "TRBJ2-3"
    assert cell1["IR_VDJ_1_c_gene"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])
    assert cell1["IR_VJ_1_locus"] == "TRA"
    assert cell1["IR_VDJ_1_locus"] == "TRB"

    assert cell2.name == "AAACCTGGTCCGTTAA-1"
    assert cell2["IR_VJ_1_cdr3"] == "CALNTGGFKTIF"
    assert cell2["IR_VJ_2_cdr3"] == "CAVILDARLMF"
    assert cell2["IR_VJ_1_expr"] == 5
    assert cell2["IR_VJ_2_expr"] == 5
    assert cell2["IR_VJ_1_locus"] == "TRA"
    assert cell2["IR_VDJ_1_locus"] == "TRB"
    assert cell2["IR_VJ_2_locus"] == "TRA"
    assert _is_na(cell2["IR_VDJ_2_cdr3"])

    assert cell3.name == "AAACTTGGTCCGTTAA-1"
    assert cell3["IR_VJ_1_locus"] == "IGK"
    assert cell3["IR_VDJ_1_locus"] == "IGH"


@pytest.mark.conda
def test_read_10x():
    anndata = read_10x_vdj(TESTDATA / "10x/all_contig_annotations.json")
    obs = anndata.obs
    # this has `is_cell=false` and should be filtered out
    assert "AAACCTGAGACCTTTG-1" not in anndata.obs_names
    assert obs.shape[0] == 3
    cell1 = obs.iloc[0, :]
    cell2 = obs.iloc[1, :]
    cell3 = obs.iloc[2, :]

    assert cell1.name == "AAACCTGAGACCTTTG-2"
    assert cell1["IR_VDJ_1_cdr3"] == "CASSPPSQGLSTGELFF"
    assert (
        cell1["IR_VDJ_1_cdr3_nt"]
        == "TGTGCCAGCTCACCACCGAGCCAGGGCCTTTCTACCGGGGAGCTGTTTTTT"
    )
    assert cell1["IR_VDJ_1_junction_ins"] == 4 + 7
    assert cell1["IR_VDJ_1_expr"] == 1
    assert cell1["IR_VDJ_1_v_gene"] == "TRBV18"
    assert cell1["IR_VDJ_1_d_gene"] == "TRBD1"
    assert cell1["IR_VDJ_1_j_gene"] == "TRBJ2-2"
    assert cell1["IR_VDJ_1_c_gene"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])
    assert np.all(
        _is_na(cell1[["IR_VJ_1_cdr3", "IR_VDJ_2_cdr3", "IR_VJ_1_junction_ins"]])
    )

    assert cell2.name == "AAACCTGAGTACGCCC-1"
    assert cell2["IR_VJ_1_cdr3"] == "CAMRVGGSQGNLIF"
    assert cell2["IR_VJ_2_cdr3"] == "CATDAKDSNYQLIW"
    assert cell2["IR_VJ_1_expr"] == 9
    assert cell2["IR_VJ_2_expr"] == 4
    assert np.all(_is_na(cell2[["IR_VDJ_1_cdr3", "IR_VDJ_2_cdr3"]]))
    assert cell2["IR_VJ_1_junction_ins"] == 4
    assert cell2["IR_VJ_2_junction_ins"] == 4

    assert cell3.name == "CAGGTGCTCGTGGTCG-1"
    assert cell3["IR_VJ_1_locus"] == "IGK"
    assert _is_na(cell3["IR_VJ_2_locus"])  # non-productive
    assert cell3["IR_VDJ_1_locus"] == "IGH"
    assert _is_na(cell3["IR_VDJ_2_locus"])  # non-productive


@pytest.mark.conda
def test_read_tracer():
    with pytest.raises(IOError):
        anndata = read_tracer(TESTDATA / "10x")

    anndata = read_tracer(TESTDATA / "tracer")
    assert "cell1" in anndata.obs_names and "cell2" in anndata.obs_names
    assert anndata.obs.shape[0] == 2

    cell1 = anndata.obs.loc["cell1", :]
    cell2 = anndata.obs.loc["cell2", :]

    assert cell1.name == "cell1"
    assert cell1["IR_VJ_1_cdr3"] == "AESTGTSGTYKYI"
    assert cell1["IR_VDJ_1_cdr3"] == "ASSYSVSRSGELF"

    assert cell2.name == "cell2"
    assert cell2["IR_VJ_1_cdr3"] == "ALSEAEGGSEKLV"
    assert cell2["IR_VDJ_1_cdr3"] == "ASSYNRGPGGTQY"
    assert cell2["IR_VDJ_1_j_gene"] == "TRBJ2-5"


@pytest.mark.conda
def test_read_airr():
    # Test that reading the files one-by-one or at once yields the same results
    anndata_tra = read_airr(TESTDATA / "airr/rearrangement_tra.tsv")
    anndata_trb = read_airr(TESTDATA / "airr/rearrangement_trb.tsv")
    anndata_ig = read_airr(TESTDATA / "airr/rearrangement_ig.tsv")
    anndata = read_airr(
        [
            TESTDATA / "airr/rearrangement_tra.tsv",
            TESTDATA / "airr/rearrangement_trb.tsv",
            TESTDATA / "airr/rearrangement_ig.tsv",
        ]
    )
    tra_cols = [
        "IR_VJ_1_cdr3",
        "IR_VJ_1_cdr3_nt",
        "IR_VJ_1_v_gene",
        "IR_VJ_1_d_gene",
        "IR_VJ_1_j_gene",
        "IR_VJ_1_c_gene",
        "IR_VJ_1_expr",
    ]
    trb_cols = [x.replace("IR_VJ", "IR_VDJ") for x in tra_cols]
    ig_cols = tra_cols + trb_cols
    pdt.assert_frame_equal(
        anndata.obs.loc[anndata.obs["IR_VJ_1_locus"] == "TRA", tra_cols],
        anndata_tra.obs.loc[:, tra_cols],
        check_categorical=False,  # categories differ, obviously
    )
    pdt.assert_frame_equal(
        anndata.obs.loc[anndata.obs["IR_VDJ_1_locus"] == "TRB", trb_cols],
        anndata_trb.obs.loc[:, trb_cols],
        check_categorical=False,  # categories differ, obviously
    )
    pdt.assert_frame_equal(
        anndata.obs.loc[anndata.obs["IR_VDJ_1_locus"] == "IGH", ig_cols],
        anndata_ig.obs.loc[:, ig_cols],
        check_categorical=False,  # categories differ, obviously
    )

    # test some fundamental values
    assert anndata.obs.shape[0] == 5

    cell1 = anndata.obs.loc["cell1", :]
    cell2 = anndata.obs.loc["cell2", :]
    cell3 = anndata.obs.loc["AAACCTGCAGCGTAAG-1", :]

    assert cell1.name == "cell1"
    assert cell1["IR_VJ_1_cdr3"] == "CTRPKWESPMVDAFDIW"
    assert cell1["IR_VDJ_2_cdr3"] == "CQQYDNLQITF"
    assert cell1["IR_VDJ_1_cdr3"] == "CQQYYHTPYSF"
    assert cell1["IR_VJ_1_locus"] == "TRA"
    assert cell1["IR_VDJ_1_locus"] == "TRB"

    assert cell2.name == "cell2"

    assert cell3["IR_VJ_1_locus"] == "IGL"
    assert cell3["IR_VDJ_1_locus"] == "IGH"
