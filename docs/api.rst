.. _api:

API
===

Import scirpy together with scanpy as

.. code-block:: python

   import scanpy as sc
   import scirpy as ir

For consistency, the scirpy API tries to follow the `scanpy API <https://scanpy.readthedocs.io/en/stable/api/index.html>`__
as closely as possible.

.. _api-io:

Input/Output: `io`
------------------

.. module:: scirpy.io
.. currentmodule:: scirpy

.. note::
   **scirpy's data structure has been updated in v0.13.0.**

   Previously, receptor data was expanded into columns of `adata.obs`, now they are stored as an :term:`awkward array` in `adata.obsm["airr"]`.
   Moreover, we now use :class:`~mudata.MuData` to handle paired transcriptomics and :term:`AIRR` data.

   :class:`~anndata.AnnData` objects created with older versions of scirpy can be upgraded with :func:`scirpy.io.upgrade_schema` to be compatible with the latest version of scirpy.

   Please check out

   * the `release notes <https://github.com/scverse/scirpy/releases/tag/v0.13.0>`_ for details about the changes and
   * the documentation about :ref:`Scirpy's data structure <data-structure>`

   .. autosummary::
      :toctree: ./generated

      io.upgrade_schema


The following functions allow to import :term:`V(D)J` information from various
formats.

.. autosummary::
   :toctree: ./generated

   io.read_h5mu
   io.read_h5ad
   io.read_10x_vdj
   io.read_tracer
   io.read_bracer
   io.read_bd_rhapsody
   io.read_airr
   io.from_dandelion

Scirpy can export data to the following formats:

.. autosummary::
   :toctree: ./generated

   io.write_airr
   io.to_dandelion


To convert own formats into the scirpy :ref:`data-structure`, we recommend building
a list of :class:`~scirpy.io.AirrCell` objects first, and then converting them into
an :class:`~anndata.AnnData` object using :func:`~scirpy.io.from_airr_cells`.
For more details, check the :ref:`Data loading tutorial <importing-data>`.

.. autosummary::
   :toctree: ./generated

   io.AirrCell
   io.from_airr_cells
   io.to_airr_cells


Preprocessing: `pp`
-------------------

.. module:: scirpy.pp
.. currentmodule:: scirpy

.. autosummary::
   :toctree: ./generated

   pp.index_chains
   pp.merge_airr
   pp.ir_dist

Get: `get`
----------

The `get` module allows retrieving :term:`AIRR` data stored in `adata.obsm["airr"]` as a per-cell :class:`~pandas.DataFrame`
or :class:`~pandas.Series`.

.. module:: scirpy.get
.. currentmodule:: scirpy

.. autosummary::
   :toctree: ./generated

   get.airr
   get.obs_context
   get.airr_context

Tools: `tl`
-----------

.. module:: scirpy.tl
.. currentmodule:: scirpy

Tools add an interpretable annotation to the :class:`~anndata.AnnData` object
which usually can be visualized by a corresponding plotting function.

Generic
^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.group_abundance

Quality control
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.chain_qc

Define and visualize clonotypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.define_clonotypes
   tl.define_clonotype_clusters
   tl.clonotype_convergence
   tl.clonotype_network
   tl.clonotype_network_igraph

Analyse clonal diversity
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.clonal_expansion
   tl.summarize_clonal_expansion
   tl.alpha_diversity
   tl.repertoire_overlap
   tl.clonotype_modularity
   tl.clonotype_imbalance

Query reference databases
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.ir_query
   tl.ir_query_annotate
   tl.ir_query_annotate_df

V(D)J gene usage
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.spectratype

Calculating mutations
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   tl.mutational_load

Plotting: `pl`
--------------

.. module:: scirpy.pl
.. currentmodule:: scirpy


Generic
^^^^^^^
.. autosummary::
   :toctree: ./generated

   pl.embedding


Tools
^^^^^
Every of these plotting functions has a corresponding *tool* in the :mod:`scirpy.tl`
section. Depending on the computational load, tools are either invoked on-the-fly
when calling the plotting function or need to be precomputed and stored in
:class:`~anndata.AnnData` previously.

.. autosummary::
   :toctree: ./generated

   pl.alpha_diversity
   pl.clonal_expansion
   pl.group_abundance
   pl.spectratype
   pl.vdj_usage
   pl.repertoire_overlap
   pl.clonotype_modularity
   pl.clonotype_network
   pl.clonotype_imbalance
   pl.logoplot_cdr3_motif


Base plotting functions: `pl.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autosummary::
   :toctree: ./generated

   pl.base.bar
   pl.base.line
   pl.base.barh
   pl.base.curve


Plot styling: `pl.styling`
^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autosummary::
   :toctree: ./generated

   pl.styling.apply_style_to_axes
   pl.styling.style_axes


Datasets: `datasets`
--------------------

.. module:: scirpy.datasets
.. currentmodule:: scirpy

Example datasets
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   datasets.wu2020
   datasets.wu2020_3k
   datasets.maynard2020
   datasets.stephenson2021_5k

Reference databases
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   datasets.vdjdb
   datasets.iedb

A reference database is also just a :ref:`Scirpy-formatted AnnData object<data-structure>`.
This means you can follow the instructions in the :ref:`data loading tutorial <importing-custom-formats>`
to build a custom reference database.


Utility functions: `util`
-------------------------

.. module:: scirpy.util
.. currentmodule:: scirpy

.. autosummary::
   :toctree: ./generated

   util.DataHandler
   util.graph.layout_components
   util.graph.layout_fr_size_aware
   util.graph.igraph_from_sparse_matrix


IR distance utilities: `ir_dist`
-----------------------------------

.. module:: scirpy.ir_dist
.. currentmodule:: scirpy

.. autosummary::
   :toctree: ./generated

   ir_dist.sequence_dist


distance metrics
^^^^^^^^^^^^^^^^


.. autosummary::
   :toctree: ./generated

   ir_dist.metrics.DistanceCalculator
   ir_dist.metrics.ParallelDistanceCalculator
   ir_dist.metrics.IdentityDistanceCalculator
   ir_dist.metrics.LevenshteinDistanceCalculator
   ir_dist.metrics.HammingDistanceCalculator
   ir_dist.metrics.GPUHammingDistanceCalculator
   ir_dist.metrics.AlignmentDistanceCalculator
   ir_dist.metrics.FastAlignmentDistanceCalculator
   ir_dist.metrics.TCRdistDistanceCalculator
