from sctcrpy import read_10x_vdj, read_tracer


def test_read_10x():
    anndata = read_10x_vdj > ("tests/data/10x/all_contig_annotations.json")


def test_read_tracer():
    anndata = read_tracer("tests/data/tracer/tracer_100")
