import scirpy as ir


def test_vdjdb():
    adata = ir.datasets.vdjdb()
    assert len(adata.obsm["airr"]) > 1000


def test_iedb():
    adata = ir.datasets.iedb()
    assert len(adata.obsm["airr"]) > 1000
