Scirpy: A Scanpy extension for analyzing single-cell T-cell receptor sequencing data
====================================================================================
|tests| |docs| |pypi| |black|

.. |tests| image:: https://github.com/grst/scirpy/workflows/tests/badge.svg
    :target: https://github.com/icbi-lab/scirpy/actions?query=workflow%3Atests
    :alt: Build Status

.. |docs| image::  https://github.com/grst/scirpy/workflows/docs/badge.svg
    :target: https://icbi-lab.github.io/scirpy
    :alt: Documentation Status
    
.. |pypi| image:: https://img.shields.io/pypi/v/scirpy?logo=PyPI
    :target: https://pypi.org/project/scirpy/
    :alt: PyPI
    
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: The uncompromising python formatter
    
Scirpy is a scalable python-toolkit to analyse single-cell T-cell receptor sequencing (scTCR-seq) data. It 
seamlessly integrates with the popular `scanpy <https://scanpy.readthedocs.io/en/stable/index.html>`_ library and
provides various modules for data import, analysis and visualization. 

Getting started
^^^^^^^^^^^^^^^
Please refer to the `Documentation <https://icbi-lab.github.io/scirpy>`_. In particular, the

  * `Tutorial <https://icbi-lab.github.io/scirpy/tutorials/tutorial_3k_tcr.html>`_ and the 
  * `API documentation <https://icbi-lab.github.io/scirpy/api.html>`_
  
In the documentation, you can also learn more about our `T-cell receptor model <https://icbi-lab.github.io/scirpy/tcr-biology.html>`_.
    
Installation
^^^^^^^^^^^^
You need to have Python 3.6 or newer installed on your system. If you don't have 
python installed, we recommend installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. 

Install the latest release of `scirpy` from `PyPI <https://pypi.org/project/scirpy/>`_. 

.. code-block::

    pip install scirpy


Alternatively, install the latest development version

.. code-block::

    git clone git@github.com:icbi-lab/scirpy.git
    cd scirpy
    pip install flit
    flit install


Bioconda coming soon. 

Release notes
^^^^^^^^^^^^^
See the `release section <https://github.com/grst/scirpy/releases>`_. 

Contact
^^^^^^^
Please use the `issue tracker <https://github.com/icbi-lab/scirpy/issues)>`_. 

Citation
^^^^^^^^
*Preprint coming soon*
