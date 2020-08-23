Scirpy: A Scanpy extension for analyzing single-cell T-cell receptor sequencing data
====================================================================================
|tests| |docs| |pypi| |bioconda| |black|

.. |tests| image:: https://github.com/icbi-lab/scirpy/workflows/tests/badge.svg
    :target: https://github.com/icbi-lab/scirpy/actions?query=workflow%3Atests
    :alt: Build Status

.. |docs| image::  https://github.com/icbi-lab/scirpy/workflows/docs/badge.svg
    :target: https://icbi-lab.github.io/scirpy
    :alt: Documentation Status
    
.. |pypi| image:: https://img.shields.io/pypi/v/scirpy?logo=PyPI
    :target: https://pypi.org/project/scirpy/
    :alt: PyPI
    
.. |bioconda| image:: https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat
     :target: http://bioconda.github.io/recipes/scirpy/README.html
     :alt: Bioconda
    
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: The uncompromising python formatter
    
Scirpy is a scalable python-toolkit to analyse  T cell receptor (TCR) repertoires from
single-cell RNA sequencing (scRNA-seq) data. It seamlessly integrates with the popular
`scanpy <https://scanpy.readthedocs.io/en/stable/index.html>`_ library and
provides various modules for data import, analysis and visualization.

.. image:: img/workflow.png
    :align: center
    :alt: The scirpy workflow 

Getting started
^^^^^^^^^^^^^^^
Please refer to the `documentation <https://icbi-lab.github.io/scirpy>`_. In particular, the

- `Tutorial <https://icbi-lab.github.io/scirpy/tutorials/tutorial_3k_tcr.html>`_, and the 
- `API documentation <https://icbi-lab.github.io/scirpy/api.html>`_.
  
In the documentation, you can also learn more about our `T-cell receptor model <https://icbi-lab.github.io/scirpy/tcr-biology.html>`_.

Case-study
~~~~~~~~~~
The case study from our preprint is available `here <https://icbi-lab.github.io/scirpy-paper/wu2020.html>`_. 
    
Installation
^^^^^^^^^^^^
You need to have Python 3.6 or newer installed on your system. If you don't have 
Python installed, we recommend installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. 

Install the latest release of `scirpy` from `PyPI <https://pypi.org/project/scirpy/>`_: 

.. code-block::

    pip install scirpy
    

Or, get it from `Bioconda <http://bioconda.github.io/recipes/scirpy/README.html>`_:

.. code-block::

    conda install -c conda-forge -c bioconda scirpy


Alternatively, install the latest development version:

.. code-block::

    pip install git+https://github.com/icbi-lab/scirpy.git@master


Release notes
^^^^^^^^^^^^^
See the `release section <https://github.com/icbi-lab/scirpy/releases>`_. 

Contact
^^^^^^^
Please use the `issue tracker <https://github.com/icbi-lab/scirpy/issues>`_. 

Citation
^^^^^^^^

    Sturm, G. Tamas, GS, ..., Finotello, F. (2020). Scirpy: A Scanpy extension for analyzing single-cell T-cell receptor sequencing data. Bioinformatics. doi:`10.1093/bioinformatics/btaa611 <https://doi.org/10.1093/bioinformatics/btaa611>`_
