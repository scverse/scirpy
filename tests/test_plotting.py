import pandas as pd
from anndata import AnnData
import pytest
from sctcrpy import pl


@pytest.fixture
def adata_clonotype():
    obs = pd.DataFrame.from_records(
        [
            ["cell1", "A", "ct1"],
            ["cell2", "A", "ct1"],
            ["cell3", "A", "ct1"],
            ["cell3", "A", "NaN"],
            ["cell4", "B", "ct1"],
            ["cell5", "B", "ct2"],
            ["cell6", "B", "ct3"],
            ["cell7", "B", "ct4"],
            ["cell8", "B", "ct4"],
        ],
        columns=["cell_id", "group", "clonotype"],
    ).set_index("cell_id")
    adata = AnnData(obs=obs)
    return adata


def test_alpha_diversity(adata_clonotype):
    pl.alpha_diversity(adata_clonotype, groupby="group")


def test_clonal_expansion(adata_clonotype):
    pl.clonal_expansion(adata_clonotype, groupby="group")

def test_group_abundance(adata_clonotype):
    pl.group_abundance(adata_clonotype, groupby="group")

def test_group_abundance_complicated(adata_clonotype):
    pl.group_abundance_complicated(adata_clonotype, groupby="group")

def test_group_abundance_lazy(adata_clonotype):
    pl.group_abundance_lazy(adata_clonotype, groupby="group")
