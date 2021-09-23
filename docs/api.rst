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

.. note::
   In scirpy v0.7.0 the way VDJ data is stored in `adata.obs` has changed to 
   be fully compliant with the `AIRR Rearrangement <https://docs.airr-community.org/en/latest/datarep/rearrangements.html#productive>`__ 
   schema. Please use :func:`~scirpy.io.upgrade_schema` to make `AnnData` objects
   from previous scirpy versions compatible with the most recent scirpy workflow. 

   .. autosummary::
      :toctree: ./generated
      
      upgrade_schema


The following functions allow to import :term:`V(D)J` information from various
formats.

.. autosummary::
   :toctree: ./generated

   read_h5ad
   read_10x_vdj
   read_tracer
   read_bracer
   read_airr
   from_dandelion

Scirpy can export data to the following formats:

.. autosummary::
   :toctree: ./generated

   write_airr
   to_dandelion

To convert own formats into the scirpy :ref:`data-structure`, we recommend building
a list of :class:`~scirpy.io.AirrCell` objects first, and then converting them into
an :class:`~anndata.AnnData` object using :func:`~scirpy.io.from_airr_cells`.
For more details, check the :ref:`Data loading tutorial <importing-data>`.

.. autosummary::
   :toctree: ./generated

   AirrCell
   from_airr_cells
   to_airr_cells


Preprocessing: `pp`
-------------------

.. module:: scirpy.pp

.. autosummary::
   :toctree: ./generated

   merge_with_ir
   merge_airr_chains
   ir_dist


Tools: `tl`
-----------

Tools add an interpretable annotation to the :class:`~anndata.AnnData` object
which usually can be visualized by a corresponding plotting function.

.. module:: scirpy.tl

Generic
^^^^^^^
.. autosummary::
   :toctree: ./generated

   group_abundance

Quality control
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   chain_qc

Define and visualize clonotypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   define_clonotypes
   define_clonotype_clusters
   clonotype_convergence
   clonotype_network
   clonotype_network_igraph

Analyse clonal diversity
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   clonal_expansion
   summarize_clonal_expansion
   alpha_diversity
   repertoire_overlap
   clonotype_modularity
   clonotype_imbalance

V(D)J gene usage
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   spectratype



Plotting: `pl`
--------------

.. module:: scirpy.pl

Generic
^^^^^^^
.. autosummary::
   :toctree: ./generated

   embedding


Tools
^^^^^
Every of these plotting functions has a corresponding *tool* in the :mod:`scirpy.tl`
section. Depending on the computational load, tools are either invoked on-the-fly
when calling the plotting function or need to be precomputed and stored in
:class:`~anndata.AnnData` previously.

.. autosummary::
   :toctree: ./generated

   alpha_diversity
   clonal_expansion
   group_abundance
   spectratype
   vdj_usage
   repertoire_overlap
   clonotype_modularity
   clonotype_network
   clonotype_imbalance



Base plotting functions: `pl.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: scirpy.pl.base

.. autosummary::
   :toctree: ./generated

   bar
   line
   barh
   curve


Plot styling: `pl.styling`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: scirpy.pl.styling

.. autosummary::
   :toctree: ./generated

   apply_style_to_axes
   style_axes


Datasets: `datasets`
--------------------

.. module:: scirpy.datasets

.. autosummary::
   :toctree: ./generated

   wu2020
   wu2020_3k
   maynard2020



Utility functions: `util`
-------------------------

.. module:: scirpy.util

.. autosummary::
   :toctree: ./generated

   graph.layout_components
   graph.layout_fr_size_aware
   graph.igraph_from_sparse_matrix


IR distance utilities: `ir_dist`
-----------------------------------

.. module:: scirpy.ir_dist

.. autosummary::
   :toctree: ./generated

   sequence_dist


distance metrics
^^^^^^^^^^^^^^^^

.. module:: scirpy.ir_dist.metrics

.. autosummary::
   :toctree: ./generated

   DistanceCalculator
   ParallelDistanceCalculator
   IdentityDistanceCalculator
   LevenshteinDistanceCalculator
   HammingDistanceCalculator
   AlignmentDistanceCalculator

