

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

We decided to handle a few minor points differenlty than Scanpy:

 * Plotting functions with inexpensive computations (e.g. :func:`scirpy.pl.clonal_expansion`)
   call the corresponding tool (:func:`scirpy.tl.clonal_expansion`) on-the-fly and
   don't store the results in the :class:`~anndata.AnnData` object. 
 * All plotting functions, by default, return a :class:`~matplotlib.axes.Axes` object, 
   or a list of such. 


.. _importing-data:

Reading in data
---------------

TODO


.. _data-structure:

Data structure
--------------

TODO