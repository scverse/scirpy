import sys
from pathlib import Path
from datetime import datetime
import jupytext

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from scirpy import __author__, __version__

# General information
project = "scirpy"
author = __author__
copyright = f"{datetime.now():%Y}, {author}."
version = __version__

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# add custom stylesheet
# https://stackoverflow.com/a/43186995/2340703
html_static_path = ["_static"]


def setup(app):
    app.add_stylesheet("custom.css")


nitpicky = True  # Warn about broken links
nitpick_ignore = [
    ("py:data", "typing.Optional"),
    ("py:class", "typing.Collection"),
    ("py:class", "typing.List"),
    ("py:class", "str"),
    ("py:class", "dict"),
]
needs_sphinx = "2.0"  # Nicer param docs


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "scanpydoc",
    "nbsphinx",
]

autosummary_generate = True
autodoc_member_order = "bysource"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
todo_include_todos = False

# Enable jupytext notebooks
nbsphinx_custom_formats = {
    ".md": lambda s: jupytext.reads(s, ".md"),
}
# nbsphinx_execute = "always"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_timeout = 300

# styling
html_theme = "sphinx_rtd_theme"
pygments_style = "sphinx"
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="grst",  # Username
    github_repo=project,  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/",  # Path in the checkout to the docs root
)

intersphinx_mapping = dict(
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    h5py=("http://docs.h5py.org/en/stable/", None),
    ipython=("https://ipython.readthedocs.io/en/stable/", None),
    leidenalg=("https://leidenalg.readthedocs.io/en/latest/", None),
    matplotlib=("https://matplotlib.org/", None),
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
    seaborn=("https://seaborn.pydata.org/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
)
