from sctcrpy import read_10x_vdj, read_tracer


def test_read_10x_example():
    anndata = read_10x_vdj("tutorial/example_data/10x/all_contig_annotations.json")


def test_read_tracer_example():
    anndata = read_tracer("tutorial/example_data/tracer/tracer_100")
