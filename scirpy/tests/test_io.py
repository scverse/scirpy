from anndata._core.anndata import AnnData
from scirpy.io import (
    read_10x_vdj,
    read_tracer,
    read_airr,
    read_bracer,
    from_ir_objs,
    to_ir_objs,
)
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
        TESTDATA / "10x/filtered_contig_annotations.csv",
    ],
)
@pytest.mark.parametrize("omit_cols", [True, False])
def test_read_and_convert_10x_example(path, omit_cols):
    """Test that a full 10x CSV table can be imported without errors.

    Additionally test that the round-trip conversion using `to_ir_objs` and
    `from_ir_objs` is the identity. Doing this here to avoid loading the data twice
    since this is already one of the longer-running tests.

    Test-dataset from https://support.10xgenomics.com/single-cell-vdj/datasets/3.1.0/vdj_nextgem_hs_pbmc3
    and https://support.10xgenomics.com/single-cell-vdj/datasets/4.0.0/sc5p_v2_hs_melanoma_10k
    under CC-BY-4.0
    """
    anndata = read_10x_vdj(path)
    if omit_cols:
        del adata.obs["IR_VJ_1_expr_raw"]
    assert anndata.shape[0] > 0

    # Test that round-trip conversion succeeds
    ir_objs = to_ir_objs(anndata)
    anndata2 = from_ir_objs(ir_objs)
    pdt.assert_frame_equal(anndata.obs, anndata2.obs)


@pytest.mark.conda
def test_read_10x_csv():
    anndata = read_10x_vdj(TESTDATA / "10x/filtered_contig_annotations.csv")
    obs = anndata.obs
    assert obs.shape[0] == 5
    cell1 = obs.iloc[1, :]
    cell2 = obs.iloc[3, :]
    cell3 = obs.iloc[4, :]

    assert cell1.name == "AAACCTGAGTACGCCC-1"
    assert cell1["IR_VDJ_1_junction_aa"] == "CASSLGPSTDTQYF"
    assert cell1["IR_VDJ_1_junction"] == "TGTGCCAGCAGCTTGGGACCTAGCACAGATACGCAGTATTTT"
    assert cell1["IR_VDJ_1_duplicate_count"] == 55
    assert cell1["IR_VDJ_1_consensus_count"] == 18021
    assert cell1["IR_VDJ_1_v_call"] == "TRBV7-2"
    assert cell1["IR_VDJ_1_d_call"] == "TRBD2"
    assert cell1["IR_VDJ_1_j_call"] == "TRBJ2-3"
    assert cell1["IR_VDJ_1_c_call"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])
    assert cell1["IR_VJ_1_locus"] == "TRA"
    assert cell1["IR_VDJ_1_locus"] == "TRB"

    assert cell2.name == "AAACCTGGTCCGTTAA-1"
    assert cell2["IR_VJ_1_junction_aa"] == "CALNTGGFKTIF"
    assert cell2["IR_VJ_2_junction_aa"] == "CAVILDARLMF"
    assert cell2["IR_VJ_1_duplicate_count"] == 5
    assert cell2["IR_VJ_2_duplicate_count"] == 5
    assert cell2["IR_VJ_1_locus"] == "TRA"
    assert cell2["IR_VDJ_1_locus"] == "TRB"
    assert cell2["IR_VJ_2_locus"] == "TRA"
    assert _is_na(cell2["IR_VDJ_2_junction_aa"])

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
    assert cell1["IR_VDJ_1_junction_aa"] == "CASSPPSQGLSTGELFF"
    assert (
        cell1["IR_VDJ_1_junction"]
        == "TGTGCCAGCTCACCACCGAGCCAGGGCCTTTCTACCGGGGAGCTGTTTTTT"
    )
    assert cell1["IR_VDJ_1_np1_length"] == 4
    assert cell1["IR_VDJ_1_np2_length"] == 7
    assert cell1["IR_VDJ_1_duplicate_count"] == 1
    assert cell1["IR_VDJ_1_consensus_count"] == 494
    assert cell1["IR_VDJ_1_v_call"] == "TRBV18"
    assert cell1["IR_VDJ_1_d_call"] == "TRBD1"
    assert cell1["IR_VDJ_1_j_call"] == "TRBJ2-2"
    assert cell1["IR_VDJ_1_c_call"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])
    assert np.all(
        _is_na(
            cell1[["IR_VJ_1_junction_aa", "IR_VDJ_2_junction_aa", "IR_VJ_1_np1_length"]]
        )
    )

    assert cell2.name == "AAACCTGAGTACGCCC-1"
    assert cell2["IR_VJ_1_junction_aa"] == "CAMRVGGSQGNLIF"
    assert cell2["IR_VJ_2_junction_aa"] == "CATDAKDSNYQLIW"
    assert cell2["IR_VJ_1_duplicate_count"] == 9
    assert cell2["IR_VJ_2_duplicate_count"] == 4
    assert np.all(_is_na(cell2[["IR_VDJ_1_junction_aa", "IR_VDJ_2_junction_aa"]]))
    assert cell2["IR_VJ_1_np1_length"] == 4
    assert _is_na(cell2["IR_VJ_1_np2_length"])
    assert cell2["IR_VJ_2_np1_length"] == 4
    assert _is_na(cell2["IR_VJ_2_np2_length"])

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
    assert anndata.obs.shape[0] == 3

    cell1 = anndata.obs.loc["cell1", :]
    cell2 = anndata.obs.loc["cell2", :]

    assert cell1.name == "cell1"
    assert cell1["IR_VJ_1_junction_aa"] == "AESTGTSGTYKYI"
    assert cell1["IR_VDJ_1_junction_aa"] == "ASSYSVSRSGELF"

    assert cell2.name == "cell2"
    assert cell2["IR_VJ_1_junction_aa"] == "ALSEAEGGSEKLV"
    assert cell2["IR_VDJ_1_junction_aa"] == "ASSYNRGPGGTQY"
    assert cell2["IR_VDJ_1_j_call"] == "TRBJ2-5"


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
        "IR_VJ_1_junction_aa",
        "IR_VJ_1_junction",
        "IR_VJ_1_v_call",
        "IR_VJ_1_d_call",
        "IR_VJ_1_j_call",
        "IR_VJ_1_c_call",
        "IR_VJ_1_consensus_count",
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
    assert cell1["IR_VJ_1_junction_aa"] == "CTRPKWESPMVDAFDIW"
    assert cell1["IR_VDJ_2_junction_aa"] == "CQQYDNLQITF"
    assert cell1["IR_VDJ_1_junction_aa"] == "CQQYYHTPYSF"
    assert cell1["IR_VJ_1_locus"] == "TRA"
    assert cell1["IR_VDJ_1_locus"] == "TRB"

    assert cell2.name == "cell2"

    assert cell3["IR_VJ_1_locus"] == "IGL"
    assert cell3["IR_VDJ_1_locus"] == "IGH"


@pytest.mark.conda
def test_read_bracer():
    anndata = read_bracer(TESTDATA / "bracer/changeodb.tab")
    assert "SRR10788834" in anndata.obs.index
    assert anndata.obs.shape[0] == 6

    cell1 = anndata.obs.loc["SRR10779208", :]
    cell2 = anndata.obs.loc["SRR10788834", :]

    assert cell1.name == "SRR10779208"
    assert cell1["IR_VJ_1_locus"] == "IGK"
    assert cell1["IR_VDJ_1_locus"] == "IGH"
    assert cell1["IR_VDJ_1_j_call"] == "IGHJ4"
    assert cell1["IR_VDJ_1_junction"] == "TGTGCGACGATGACGGGGGGTGACCTTGACTACTGG"
    assert cell1["IR_VDJ_1_junction_aa"] == "CATMTGGDLDYW"
    assert cell1["IR_VJ_1_np1_length"] == 1
    assert _is_na(cell1["IR_VJ_1_np2_length"])

    assert cell2.name == "SRR10788834"
    assert cell2["IR_VDJ_1_junction_aa"] == "CARDHIVVLEPTPKRYGMDVW"
    assert (
        cell2["IR_VDJ_1_junction"]
        == "TGTGCGAGAGATCATATTGTAGTCTTGGAACCTACCCCTAAGAGATACGGTATGGACGTCTGG"
    )
    assert cell2["IR_VDJ_1_np1_length"] == 2
    assert cell2["IR_VDJ_1_np2_length"] == 22
