.. _developer-docs:

Developer Documentation
=======================

The Scanpy documentation provides extensive `developer documentation <https://scanpy.readthedocs.io/en/latest/dev/index.html>`_. 
Since Scirpy's design closely follows Scanpy's, most of its content is applicable also to Scirpy. 

This document will not repeat the content from there, but aim at summarizing the most important information to get you started
on contributing to scirpy and on pointing out differences to the Scanpy workflow. 


Getting set-up
^^^^^^^^^^^^^^

We assume that you are already familiar with Git and making Pull requests on GitHub. If not please, refer to the Scanpy 
developer documentation.

Installing additional dependencies
-------------------------------

In addition to Scipy's runtime dependencies you need additional python 
packages to run the tests and building the documentation. It's easy to 
install them using pip: 

.. code:
   
   pip install scirpy[test,doc]


Formatting code
---------------

All Python code needs to be formatted using `black <https://github.com/psf/black>`_. 
If the code is not formatted correctly, the CI checks will fail. 

We recommend setting up the pre-commit hook that automatically formats
the code on every commit.

.. code: 

   # inside root of scirpy repository
   pre-commit install

Alternatively, you can manually run black by running 

.. code:

   black .

Most IDEs also have an "autoformat on save" feature which can be enabled.  


Running tests
-------------

Scirpy uses automated testing with `pytest <https://docs.pytest.org>`_. 
All tests need to pass before we can merge a pull request. If you add
new functionality, `please add tests <https://scanpy.readthedocs.io/en/latest/dev/testing.html#writing-tests>`_

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively, 
you can run all tests from the command line by executing 

.. code: 

   pytest

in the root of the scirpy repository. 


Previewing the docs
^^^^^^^^^^^^^^^^^^^

Updates to the documentation are automatically built 
by the CI for every PR. Once the PR is merged they will
be automatically published on the documentation website. 

The easiest way to preview changes to the documentation 
is to download the CI artifact:

TODO add screenshot

Alternatively, you can build and preview the documentation locally: 

 ..code: 
        
   cd docs
   make html
   # open the docs in the web browser, e.g. 
   firefox _build/html/index.html


