import scirpy as ir


def test_vdjdb():
    adata = ir.datasets.vdjdb()
    assert adata.shape[0] > 1000


def test_iedb():
    adata = ir.datasets.iedb()
    assert adata.shape[0] > 1000
