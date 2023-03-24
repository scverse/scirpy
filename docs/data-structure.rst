.. _data-structure:

Data structure
==============

.. note:: 

    Scirpy's datastructure was fundamentally changed in version 0.13.0. While previously, immune receptor
    data was expanded into columns in `adata.obs`, they are now stored as :term:`awkward array` in `adata.obsm`. 
    Fore more details ... # TODO

For instructions how to load data into scirpy, see :ref:`importing-data`.

Scirpy combines the `AIRR Rearrangement standard <https://docs.airr-community.org/en/latest/datarep/rearrangements.html>`_ 
for representing adaptive immune receptor repertoire data with scverse's `AnnData <https://anndata.readthedocs.io/en/latest/>`_ data structure.

AnnData combines a gene expression matrix (`.X`), gene-level annotations (`.var`) and
cell-level annotations (`.obs`) into a single object. Additionally, matrices aligned to the cells can be stored in `.obsm`.

.. figure:: img/anndata.svg
   :width: 350px

The AIRR rearrangement standard defines a set of fields to describe a single receptor chain. One cell can have 
multiple receptor chains. This relationship is represented as an :term:`awkward array` stored in `adata.obsm["airr"]`.

The first dimension of the array represents the cells and is aligned to the `obs` axis of the `AnnData` object. 
The second dimension represents the number of chains per cell and is of variable length. The third dimension 
is a :ref:~akward.RecordType` and represents fields defined in the rearrangement standard. 

.. code-block:: python

    # adata.obsm["airr"]
    [
        # cell0: 2 chains
        [{"locus": "TRA", "junction_aa": "CADASGT..."}, {"locus": "TRB", "junction_aa": "CTFDD..."}],
        # cell1: 1 chain
        [{"locus": "IGH", "junction_aa": "CDGFFA..."}],
        # cell2: 0 chains
        [],
    ]

This allows to losslessly store a complete AIRR rearrangement table in AnnData. The purpose of scirpy's :ref:`IO module <api-io>`
is to create AnnData objects of this structure. At this point, chains are neither filtered, nor separated by locus. 
This allows adopting the datastructure and reusing the IO functions even for packages that do not adhere to the 
:ref:`scirpy receptor model <receptor-model>`. 

chain indices
-------------


Working with multimodal data
----------------------------