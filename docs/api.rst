API
===

.. _api-io:

Input/Output
------------

.. module:: scirpy

.. autosummary::
   :toctree: .

   read_h5ad
   read_10x_vdj
   read_tracer


Datasets: `datasets`
--------------------

.. module:: scirpy.datasets

.. autosummary::
   :toctree: .

   wu2020
   wu2020_3k



Preprocessing: `pp`
-------------------

.. module:: scirpy.pp

.. autosummary::
   :toctree: .

   merge_with_tcr
   tcr_neighbors


Tools: `tl`
-----------

.. module:: scirpy.tl

.. autosummary::
   :toctree: .

   define_clonotypes
   clonotype_network
   chain_pairing
   clonal_expansion
   summarize_clonal_expansion
   alpha_diversity
   group_abundance
   spectratype
   repertoire_overlap
   clonotype_imbalance


Plotting: `pl`
--------------

.. module:: scirpy.pl

.. autosummary::
   :toctree: . 

   alpha_diversity
   clonal_expansion
   group_abundance
   spectratype
   vdj_usage
   repertoire_overlap
   clonotype_imbalance
   clonotype_network
   clonotype_network_igraph
   embedding

   
Base plotting functions: `pl.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: scirpy.pl.base

.. autosummary::
   :toctree: .

   bar
   line
   barh
   curve


Plot styling: `pl.styling`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: scirpy.pl.styling

.. autosummary::
   :toctree: . 

   apply_style_to_axes
   style_axes


Utility functions: `util`
-------------------------

.. module:: scirpy.util

.. autosummary::
   :toctree: . 

   graph.layout_components


TCR distance metrics: `tcr_dist`
-----------------------------------

.. module:: scirpy.tcr_dist

.. autosummary::
   :toctree: .

   tcr_dist
   DistanceCalculator
   IdentityDistanceCalculator
   LevenshteinDistanceCalculator
   AlignmentDistanceCalculator
..   DistanceCalculator.calc_dist_mat


