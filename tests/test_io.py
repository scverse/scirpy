from sctcrpy._io import read_10x


def test_read_10x():
    anndata = read_10x(
        "tests/data/10x/vdj_v1_hs_pbmc3_b_filtered_contig_annotations.csv"
    )
