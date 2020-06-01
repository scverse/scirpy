from scirpy.io import read_10x_vdj, read_tracer, read_airr
from scirpy.util import _is_na, _is_false
import numpy as np
import pytest
import pandas.testing as pdt


def test_read_10x_example():
    """Test that a full 10x CSV table can be imported without errors.

    Test-dataset from https://support.10xgenomics.com/single-cell-vdj/datasets/3.1.0/vdj_nextgem_hs_pbmc3
    under CC-BY-4.0
    """
    anndata = read_10x_vdj(
        "tests/data/10x/vdj_nextgem_hs_pbmc3_t_filtered_contig_annotations.csv.gz"
    )


# def test_read_tracer_example():
#     anndata = read_tracer("tutorial/example_data/tracer/tracer_100")


def test_read_10x_csv():
    anndata = read_10x_vdj("tests/data/10x/filtered_contig_annotations.csv")
    obs = anndata.obs
    assert obs.shape[0] == 4
    cell1 = obs.iloc[1, :]
    cell2 = obs.iloc[3, :]

    assert cell1.name == "AAACCTGAGTACGCCC-1"
    assert cell1["TRB_1_cdr3"] == "CASSLGPSTDTQYF"
    assert cell1["TRB_1_cdr3_nt"] == "TGTGCCAGCAGCTTGGGACCTAGCACAGATACGCAGTATTTT"
    assert _is_na(cell1["TRB_1_junction_ins"])
    assert cell1["TRB_1_expr"] == 55
    assert cell1["TRB_1_v_gene"] == "TRBV7-2"
    assert cell1["TRB_1_d_gene"] == "TRBD2"
    assert cell1["TRB_1_j_gene"] == "TRBJ2-3"
    assert cell1["TRB_1_c_gene"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])

    assert cell2.name == "AAACCTGGTCCGTTAA-1"
    assert cell2["TRA_1_cdr3"] == "CALNTGGFKTIF"
    assert cell2["TRA_2_cdr3"] == "CAVILDARLMF"
    assert cell2["TRA_1_expr"] == 5
    assert cell2["TRA_2_expr"] == 5
    assert _is_na(cell2["TRB_2_cdr3"])


def test_read_10x():
    anndata = read_10x_vdj("tests/data/10x/all_contig_annotations.json")
    obs = anndata.obs
    # this has `is_cell=false` and should be filtered out
    assert "AAACCTGAGACCTTTG-1" not in anndata.obs_names
    assert obs.shape[0] == 2
    cell1 = obs.iloc[0, :]
    cell2 = obs.iloc[1, :]

    assert cell1.name == "AAACCTGAGACCTTTG-2"
    assert cell1["TRB_1_cdr3"] == "CASSPPSQGLSTGELFF"
    assert (
        cell1["TRB_1_cdr3_nt"] == "TGTGCCAGCTCACCACCGAGCCAGGGCCTTTCTACCGGGGAGCTGTTTTTT"
    )
    assert cell1["TRB_1_junction_ins"] == 4 + 7
    assert cell1["TRB_1_expr"] == 1
    assert cell1["TRB_1_v_gene"] == "TRBV18"
    assert cell1["TRB_1_d_gene"] == "TRBD1"
    assert cell1["TRB_1_j_gene"] == "TRBJ2-2"
    assert cell1["TRB_1_c_gene"] == "TRBC2"
    assert _is_false(cell1["multi_chain"])
    assert np.all(_is_na(cell1[["TRA_1_cdr3", "TRB_2_cdr3", "TRA_1_junction_ins"]]))

    assert cell2.name == "AAACCTGAGTACGCCC-1"
    assert cell2["TRA_1_cdr3"] == "CAMRVGGSQGNLIF"
    assert cell2["TRA_2_cdr3"] == "CATDAKDSNYQLIW"
    assert cell2["TRA_1_expr"] == 9
    assert cell2["TRA_2_expr"] == 4
    assert np.all(_is_na(cell2[["TRB_1_cdr3", "TRB_2_cdr3"]]))
    assert cell2["TRA_1_junction_ins"] == 4
    assert cell2["TRA_2_junction_ins"] == 4


def test_read_tracer():
    with pytest.raises(IOError):
        anndata = read_tracer("scirpy")

    anndata = read_tracer("tests/data/tracer")
    assert "cell1" in anndata.obs_names and "cell2" in anndata.obs_names
    assert anndata.obs.shape[0] == 2

    cell1 = anndata.obs.loc["cell1", :]
    cell2 = anndata.obs.loc["cell2", :]

    assert cell1.name == "cell1"
    assert cell1["TRA_1_cdr3"] == "AESTGTSGTYKYI"
    assert cell1["TRB_1_cdr3"] == "ASSYSVSRSGELF"

    assert cell2.name == "cell2"
    assert cell2["TRA_1_cdr3"] == "ALSEAEGGSEKLV"
    assert cell2["TRB_1_cdr3"] == "ASSYNRGPGGTQY"
    assert cell2["TRB_1_j_gene"] == "TRBJ2-5"


def test_read_airr():
    # Test that reading the files one-by-one or at once yields the same results
    anndata_tra = read_airr("tests/data/airr/rearrangement_tra.tsv")
    anndata_trb = read_airr("tests/data/airr/rearrangement_trb.tsv")
    anndata = read_airr(
        [
            "tests/data/airr/rearrangement_tra.tsv",
            "tests/data/airr/rearrangement_trb.tsv",
        ]
    )
    tra_cols = [
        "TRA_1_cdr3",
        "TRA_1_cdr3_nt",
        "TRA_1_v_gene",
        "TRA_1_d_gene",
        "TRA_1_j_gene",
        "TRA_1_c_gene",
        "TRA_1_expr",
    ]
    trb_cols = [x.replace("TRA", "TRB") for x in tra_cols]
    pdt.assert_frame_equal(anndata.obs[tra_cols], anndata_tra.obs[tra_cols])
    pdt.assert_frame_equal(anndata.obs[trb_cols], anndata_trb.obs[trb_cols])

    # test some fundamental values
    assert "cell1" in anndata.obs_names and "cell2" in anndata.obs_names
    assert anndata.obs.shape[0] == 3

    cell1 = anndata.obs.loc["cell1", :]
    cell2 = anndata.obs.loc["cell2", :]

    assert cell1.name == "cell1"
    assert cell1["TRA_1_cdr3"] == "CTRPKWESPMVDAFDIW"
    assert cell1["TRB_2_cdr3"] == "CQQYDNLQITF"
    assert cell1["TRB_1_cdr3"] == "CQQYYHTPYSF"

    assert cell2.name == "cell2"
