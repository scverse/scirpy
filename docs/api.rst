API
===


.. module:: sctcrpy 

.. automodule:: sctcrpy 
   :noindex:


Input/Output
------------

.. autosummary::
   :toctree: .

   read_h5ad
   read_10x_vdj
   read_10x_vdj_csv
   read_tracer


Datasets
--------

.. module:: sctcrpy.datasets

.. autosummary::
   :toctree: .

   wu2020
   wu2020_3k



Preprocessing: `pp`
-------------------

.. module:: sctcrpy.pp

.. autosummary::
   :toctree: .

   merge_with_tcr


Tools: `tl`
-----------

.. module:: sctcrpy.tl

.. autosummary::
   :toctree: .

   define_clonotypes
   clonotype_network
   tcr_dist
   chain_pairing
   clip_and_count
   alpha_diversity
   group_abundance
   spectratype


Plotting: `pl`
--------------

Base plotting functions: `pl.base`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: sctcrpy.pl.base

.. autosummary::
   :toctree: .
   bar
   line
   barh
   curve


.. module:: sctcrpy.pl

.. autosummary::
   :toctree: . 

   alpha_diversity
   clip_and_count
   clonal_expansion
   group_abundance
   spectratype
   clonotype_network

   
