import pandas as pd
from typing import Iterable


def concat(objs: Iterable[pd.DataFrame], names: Iterable[str] = None):
    """Concatenate scTCRpy sample dataframes
    
    Parameters
    ----------
    objs
        DataFrames to concatenate
    names
        List of sample names associated with the DataFrames. 
        If none are provided, ascending numeric ids are used. 

    Returns
    -------
    pd.DataFrame 
        concatenated DataFrames with 'sample' column added. 
    """
    if names is None:
        names = range(len(objs))

    def _add_sample_names(_objs, _names):
        for name, obj in zip(_names, _objs):
            # operate on a copy in order not to change input data
            obj2 = obj.copy()
            obj2.insert(0, "sample", str(name))
            yield obj2

    return pd.concat(_add_sample_names(objs, names))


def define_clonotypes(clone_df, flavor="paired"):
    """Define clonotypes based on CDR3 region. 
    
    Parameters
    ----------
    clone_df : [type]
        [description]
    flavor : str, optional
        [description], by default "paired"
    
    Returns
    -------
    [type]
        [description]
    """
    assert flavor == "paired", "Other flavors currently not supported"
    clone_df = clone_df.loc[clone_df.chain.isin(["TRA", "TRB"]), :]

    def _apply(df):
        df_a = df[df["chain"] == "TRA"].sort_values("umis", ascending=False)
        df_b = df[df["chain"] == "TRB"].sort_values("umis", ascending=False)

        return pd.Series(
            {
                "dominant_alpha": df_a["cdr3"].values[0]
                if df_a.shape[0] >= 1
                else None,
                "dominant_beta": df_b["cdr3"].values[0] if df_b.shape[0] >= 1 else None,
                "secondary_alpha": df_a["cdr3"].values[1]
                if df_a.shape[0] >= 2
                else None,
                "secondary_beta": df_b["cdr3"].values[1]
                if df_b.shape[0] >= 2
                else None,
            }
        )

    return clone_df.groupby(["sample", "barcode"]).apply(_apply)
