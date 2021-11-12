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
   In scirpy v0.7.0 the way VDJ data is stored in `adata.obs` has changed to
   be fully compliant with the `AIRR Rearrangement <https://docs.airr-community.org/en/latest/datarep/rearrangements.html#productive>`__
   schema. Please use :func:`~scirpy.io.upgrade_schema` to make `AnnData` objects
   from previous scirpy versions compatible with the most recent scirpy workflow.

   .. autosummary::
      :toctree: ./generated

      io.upgrade_schema


The following functions allow to import :term:`V(D)J` information from various
formats.

.. autosummary::
   :toctree: ./generated

   io.read_h5ad
   io.read_10x_vdj
   io.read_tracer
   io.read_bracer
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

   pp.merge_with_ir
   pp.merge_airr_chains
   pp.ir_dist


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

.. autosummary::
   :toctree: ./generated

   datasets.wu2020
   datasets.wu2020_3k
   datasets.maynard2020

Reference databases
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   datasets.vdjdb

A reference database is also just a :ref:`Scirpy-formatted AnnData object<data-structure>`.
This means you can follow the instructions in the :ref:`data loading tutorial <importing-custom-formats>`
to build a custom reference database.


Utility functions: `util`
-------------------------

.. module:: scirpy.util
.. currentmodule:: scirpy

.. autosummary::
   :toctree: ./generated

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
   ir_dist.metrics.AlignmentDistanceCalculator

