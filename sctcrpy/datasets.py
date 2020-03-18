from anndata import AnnData


def wu2020() -> AnnData:
    """
    Return the dataset from [Wu2020]_ as AnnData object. 

    Downloads the data from GSE139555. 
    """
    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE139555&format=file"
