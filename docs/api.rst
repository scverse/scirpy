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
   read_tracer


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
   decide_chain_cat
   chain_pairing
   tcr_dist 
   alpha_diversity
   clonal_expansion
   cdr_convergence
   spectratype
   group_abundance


Plotting: `pl`
--------------

.. module:: sctcrpy.pl

.. autosummary::
   :toctree: . 

   alpha_diversity
   clonal_expansion
   cdr_convergence
   spectratype
   group_abundance
   nice_bar_plain
   nice_line_plain
   nice_curve_plain
   nice_stripe_plain
   check_for_plotting_profile
   reset_plotting_profile

   
