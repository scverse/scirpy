Features
========

Execution engines
-----------------
Reportsrender comes with two execution engines:

* **Rmarkdown.** This engine makes use of the `Rmarkdown package <https://rmarkdown.rstudio.com/>`_
  implemented in R. Essentially, this engine calls
  `Rscript -e "rmarkdown::render()"`. It supports 
  Rmarkdown notebooks (`Rmd` format) and python notebooks
  through `reticulate <https://rstudio.github.io/reticulate/>`_.

* **Papermill.** This engine combines `papermill <https://github.com/nteract/papermill>`_
  and `nbconvert <https://nbconvert.readthedocs.io/en/latest/>`_ to parametrize and 
  execute notebooks. It supports any programming language for which a jupyter
  kernel is installed. 


Supported notebook formats
--------------------------
Reportsrender uses `jupytext <https://github.com/mwouts/jupytext>`_
to convert between input formats. 
Here is the full list of `supported formats <https://jupytext.readthedocs.io/en/latest/formats.html>`_.

So no matter if you want to run an `Rmd` file with papermill, an `ipynb` with Rmarkdown or a
`Hydrogen percent script <https://atom.io/packages/hydrogen>`_, reportsrender
has got you covered. 



Hiding cell inputs/outputs
--------------------------
You can hide inputs and or outputs of individual cells:

Papermill engine:
^^^^^^^^^^^^^^^^^

Within a jupyter notebook:

* edit cell metadata
* add one of the following `tags`: `hide_input`, `hide_output`, `remove_cell`

::

    {
        "tags": [
            "remove_cell"
        ]
    }

Rmarkdown engine:
^^^^^^^^^^^^^^^^^

* all native input control options
  (e.g. `results='hide'`, `include=FALSE`, `echo=FALSE`) are supported. See the
  `Rmarkdown documentation <https://bookdown.org/yihui/rmarkdown/r-code.html>`_ for more details.

`Jupytext <https://github.com/mwouts/jupytext>`_ automatically converts the
tags to Rmarkdown options for all supported formats.



Parametrized notebooks
----------------------

 
Papermill engine:
^^^^^^^^^^^^^^^^^

* See the `Papermill documentation <https://papermill.readthedocs.io/en/latest/usage-parameterize.html>`_

Example:

* Add the tag `parameters` to the metadata of a cell in a jupyter notebook.
* Declare default parameters in that cell:

::

    input_file = '/path/to/default_file.csv'


* Use the variable as any other:

::

    import pandas as pd
    pd.read_csv(input_file)



Rmarkdown engine:
^^^^^^^^^^^^^^^^^

* See the `documentation <https://bookdown.org/yihui/rmarkdown/params-declare.html>`_.

Example:

* Declare the parameter in the yaml frontmatter.
* You can set default parameters that will be used when
  the notebook is executed interactively in Rstudio. They will be overwritten
  when running through `reportsrender`.

::

    ---
    title: My Document
    output: html_document
    params:

      input_file: '/path/to/default_file.csv'
    ---

* Access the parameters from the code:

::

    read_csv(params$input_file)


Be compatible with both engines:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes it's possible! You can execute the same notebook with both engines.
Adding parameters is a bit more cumbersome though.

Example (Python notebook stored as `.Rmd` file using *jupytext*):

::

    ---
    title: My Document
    output: html_document
    params:
      input_file: '/path/to/default_file.csv'
    ---

    ```{python tags=c("parameters")}
    try:
        # try to get param from Rmarkdown using reticulate.
        input_file = r.params["input_file"]
    except:
        # won't work if running papermill. Re-declare default parameters.
        input_file = "/path/to/default_file.csv"
    ```


Sharing reports
---------------
Reportsrender create self-contained HTML files 
that can be easily shared, e.g. via email. 

I do, however, recommend using `github pages <https://pages.github.com/>`_
to upload and share your reports. A central website serves 
as a *single point of truth* and elimiates the problem of 
different versions of your reports being emailed around. 

You can make use of `reportsrender index` to automatically generate 
an index page listing multiple reports: 

Say, you generated several reports and already put them into your 
github-pages directory:

::

    gh-pages
    ├── 01_preprocess_data.html
    ├── 02_analyze_data.html
    └── 03_visualize_data.htmlp

Then you can generate an index file listing and linking to your reports by running

::

    reportsrender index --index gh-pages/index.md gh-pages/*.html

For more details see :ref:`cli` and :meth:`reportsrender.build_index` 


Password protection
^^^^^^^^^^^^^^^^^^^
Not all analyses can be shared publicly. Unfortunately, 
github-pages does not support password protection. 

There is `a workaround <https://stackoverflow.com/questions/27065192/how-do-i-protect-a-directory-within-github-pages>`_,
though:

As github-pages doesn't list directories, you can simply create
a long, cryptic subdirectory, e.g. `t8rry6poj7ua6eujqpb57`
and put your reports within. Only people with whom 
you share the exact link will be able to access the site. 


Combine notebooks into a pipeline
---------------------------------
Reportsrender is built with pipelines in mind. 
You can easily combine individual analysis steps into a fully reproducible 
pipeline using workflow engines such as `Nextflow <https://www.nextflow.io/>`_
or `Snakemake <https://snakemake.readthedocs.io/en/stable/>`_. 

A full example how such a pipeline might look like is available in 
a dedicated GitHub repository: `universal_analysis_pipeline <https://github.com/grst/universal_analysis_pipeline/>`_. 
It's based on Nextflow, but could easily be adapted to other pipelining engines. 

