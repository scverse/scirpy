# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))

# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("scirpy")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

api_dir = HERE / "_static" / "api"
api_rel_dir = "_static/api"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "scverse",  # Username
    "github_repo": project_name,  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "cycler": ("https://matplotlib.org/cycler", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable", None),
    "leidenalg": ("https://leidenalg.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "seaborn": ("https://seaborn.pydata.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "networkx": ("https://networkx.org/documentation/networkx-1.10", None),
    "dandelion": ("https://sc-dandelion.readthedocs.io/en/latest", None),
    "muon": ("https://muon.readthedocs.io/en/latest", None),
    "mudata": ("https://mudata.readthedocs.io/en/latest", None),
    "awkward": ("https://awkward-array.org/doc/main", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "logomaker": ("https://logomaker.readthedocs.io/en/latest/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = project_name
html_logo = "img/scirpy_logo.png"
html_css_files = ["css/custom.css"]

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

# -- nbsphinx Tutorials ----------------------------------------------------------------

# # Enable jupytext notebooks
# nbsphinx_custom_formats = {
#     ".md": lambda s: jupytext.reads(s, ".md"),
# }
# # nbsphinx_execute = "always"
# nbsphinx_execute_arguments = [
#     "--InlineBackend.figure_formats={'svg'}",
#     "--InlineBackend.rc={'figure.dpi': 96}",
# ]
# nbsphinx_timeout = 300


nitpick_ignore = [
    ("py:class", "igraph.Graph"),
    ("py:class", "igraph.Layout"),
    ("py:class", "igraph.layout.Layout"),
    # the following entries are because the `MutableMapping` base class does not
    # use type hints.
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "D[k] if k in D, else d.  d defaults to None."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "v, remove specified key and return the corresponding value."),
    ("py:class", "(k, v), remove and return some (key, value) pair"),
    ("py:class", "D.get(k,d), also set D[k]=d if k not in D"),
    ("py:class", "None.  Update D from mapping/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
    # don't know why these are not working
    ("py:class", "seaborn.matrix.ClusterGrid"),
    ("py:meth", "mudata.MuData.update"),
    ("py:class", "awkward.highlevel.Array"),
    ("py:class", "logomaker.src.Logo.Logo"),
]
