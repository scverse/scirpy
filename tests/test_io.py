from sctcrpy._io import read_10x


def test_read_10x():
    anndata = read_10x("tests/data/10x/vdbg_141_tcr_filtered_contig_annotations.csv")
