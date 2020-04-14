API
===


.. module:: scirpy 

.. automodule:: scirpy 
   :noindex:

.. _api-io:

Input/Output
------------

.. autosummary::
   :toctree: .

   read_h5ad
   read_10x_vdj
   read_tracer


Datasets
--------

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
   tcr_dist
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
   clonotype_network

   
Base plotting functions: `pl.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: scirpy.pl.base

.. autosummary::
   :toctree: .

   bar
   line
   barh
   curve
