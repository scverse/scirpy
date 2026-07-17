# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

import muon as mu
import pandas as pd

# Use this list of 3k barcodes for consistency with previous versions
barcodes = pd.read_csv("./3k_barcodes.csv", header=None)[0].values

barcodes = pd.Series(barcodes).str.replace("-\\d+$", "", regex=True).values

mdata = mu.read_h5mu("wu2020.h5mu")

assert mdata.obs_names.is_unique

mdata = mdata[barcodes, :].copy()

mdata

mdata.write_h5mu("wu2020_3k.h5mu", compression="lzf")
