

Usage principles
=========================

Import scirpy as

.. code-block:: python

   import scanpy as sc
   import scirpy as ir


Workflow
--------
Scirpy is an extension to `Scanpy <https://scanpy.readthedocs.io>`_ and adheres to its
`workflow principles <https://scanpy.readthedocs.io/en/stable/usage-principles.html>`_:

 * The :ref:`API <API>` is divided into *preprocessing* (`pp`), *tools* (`tl`),
   and *plotting* (`pl`).
 * All functions work on :class:`~anndata.AnnData` objects.
 * The :class:`~anndata.AnnData` instance is modified inplace, unless the functions
   is called with the keyword argument `inplace=False`.

We decided to handle a few minor points differently to Scanpy:

 * Plotting functions with inexpensive computations (e.g. :func:`scirpy.pl.clonal_expansion`)
   call the corresponding tool (:func:`scirpy.tl.clonal_expansion`) on-the-fly and
   don't store the results in the :class:`~anndata.AnnData` object.
 * All plotting functions, by default, return a :class:`~matplotlib.axes.Axes` object,
   or a list of such.


.. _data-structure:

Data structure
--------------

For instructions how to load data into scirpy, see :ref:`importing-data`.

Scirpy leverages the `AnnData <https://github.com/theislab/anndata>`_ data structure
which combines a gene expression matrix (`.X`), gene-level annotations (`.var`) and
cell-level annotations (`.obs`) into a single object. :class:`~anndata.AnnData` forms the basis for the
`Scanpy analysis workflow <https://scanpy.readthedocs.io/en/stable/usage-principles.html>`_
for single-cell transcriptomics data.

.. figure:: img/anndata.svg
   :width: 350px

   Image by `F. Alex Wolf <http://falexwolf.de/>`__.


Scirpy adds the following :term:`IR`-related columns to `AnnData.obs`:

 * `IR_VJ_1_<attr>`/`IR_VJ_2_<attr>`: columns related to the primary and secondary
   :term:`VJ<V(D)J>`-chain of a receptor (`TRA`, `TRG`, `IGK`, or `IGL`)
 * `IR_VDJ_1_<attr>`/`IR_VDJ_2_<attr>`: columns related to the primary and secondary
   :term:`VDJ<V(D)J>`-chain of a receptor (`TRB`, `TRD`, or `IGH`)
 * `has_ir`: `True` for all cells with an adaptive immune receptor
 * `extra_chains`: Contains non-productive chains (if not filtered out), and extra chains
   that do not fit into the 2 `VJ` + 2 `VDJ` chain model encoded as JSON. Scirpy does
   not use this information except for writing it back to AIRR format using 
   :func:`scirpy.io.write_airr`. 
 * `multi_chain`: `True` for all cells with more than two productive `VJ` cells or 
   two or more productive `VDJ` cells. 

Where `<attr>` can be any field of the `AIRR Rearrangement Schema <https://docs.airr-community.org/en/latest/datarep/rearrangements.html#fields>`__. 
For Scirpy the following fields are relevant: 

 * `locus`: The :term:`IGMT locus name<Chain locus>` of the chain (`TRA`, `IGH`, etc.)
 * `c_call`, `v_call`, `d_call`, `j_call`: The gene symbols of the respective genes
 * `junction_aa` and `junction`: The amino acid and nucleotide sequences of the CDR3 regions
